#!/usr/bin/env python3
"""
GalaxySD - DAPS 物理引导推理脚本 (40k 开盖验丹版)
特性：支持 GPU Batch 并行、复合归一化、微批次防 OOM、DAPS 退火解耦步进
"""

import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from hcpdiff.visualizer import Visualizer
from omegaconf import OmegaConf
from accelerate.hooks import remove_hook_from_module
import torch.nn as nn
from torchvision import models
from diffusers.models import AutoencoderKL
import warnings
from torchvision.transforms import functional as TF
warnings.filterwarnings('ignore')

# =====================================================================
# 🔧 [配置区] 
# =====================================================================
CFG_PATH = "cfgs/infer/text2img_galaxy_catalog.yaml"  
CATALOG_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/labeled_catalog_cleaned_v3.csv"
INDEX_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/validation/validation_index.txt" 
REAL_IMG_ROOT = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/filtered_data" 
OUT_DIR = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/validation/val_results_150k_dps" # 建议改名防混淆
STATS_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/normalization_stats_v3.csv"
REGRESSOR_CKPT = "regressor_output_v3/best_regressor_v3.pth" 

BATCH_SIZE = 16    
SEED = 114514      
LAMBDA_GUIDE = 10.0 
SKIP_RATIO = 0.3    # 🚀 [修改 1] DAPS 全局生效，真正测试 40k 基座对推力的承受度
MICRO_BATCH = 4 
# =====================================================================

class GalaxyRegressor(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        return self.output_act(self.backbone(x))

class CustomVisualizer(Visualizer):
    def load_model(self, pretrained_model):
        custom_te = None
        if 'new_components' in self.cfgs and 'text_encoder' in self.cfgs.new_components:
            custom_te = self.cfgs.new_components.text_encoder
            OmegaConf.set_struct(self.cfgs, False)
            self.cfgs.new_components.pop('text_encoder')
        pipe = super().load_model(pretrained_model)
        if custom_te is not None:
            pipe.text_encoder = custom_te
            self.cfgs.new_components.text_encoder = custom_te
        return pipe

class CatalogInferencerDPS:
    def __init__(self, cfg_path, catalog_file, index_file, real_img_root, stats_file, regressor_ckpt):
        self.cfg = OmegaConf.load(cfg_path)
        self._fill_default_cfg()
        self.real_img_root = real_img_root

        self.feature_columns = [
            'DESIDR1_FLUX_G', 'DESIDR1_FLUX_R', 'DESIDR1_FLUX_Z', 'DESIDR1_FLUX_W1', 'DESIDR1_FLUX_W2',
            'DESIDR1_SHAPE_R', 'DESIDR1_SHAPE_E1', 'DESIDR1_SHAPE_E2', 'DESIDR1_SERSIC',
            'mapped_smooth_combined_score', 'mapped_barred_spirals_combined_score',
            'mapped_unbarred_spirals_combined_score', 'mapped_edge_on_with_bulge_combined_score',
            'mapped_mergers_combined_score', 'mapped_irregular_combined_score', 'DESIDR1_Z',
            'MassEMLine_MASS_CG', 'MassEMLine_SFR_CG', 'MassEMLine_AGE_CG',
            'MassEMLine_HALPHA_FLUX', 'MassEMLine_OIII5007_FLUX', 'MassEMLine_EBV',
        ]

        self.stats_dict = self._load_normalization_stats(stats_file)
        
        with open(index_file, 'r') as f:
            self.target_indexs = [line.strip() for line in f.readlines() if line.strip()]
        target_set = set(self.target_indexs)
        print(f"[*] 从 txt 文件中读取到 {len(self.target_indexs)} 个目标 index...")

        self.catalog_dict = self._load_catalog_by_indexs(catalog_file, target_set)
        
        print("[*] 正在加载生成模型与 Pipeline...")
        self.visualizer = CustomVisualizer(self.cfg)
        self.visualizer.pipe.set_progress_bar_config(disable=True)
        self.device = self.visualizer.pipe.device

        print("[*] 正在加载物理裁判 (Physics Regressor)...")
        self.physics_regressor = GalaxyRegressor(22) 
        if regressor_ckpt and os.path.exists(regressor_ckpt):
            self.physics_regressor.load_state_dict(torch.load(regressor_ckpt, map_location='cpu'))
            print("  ✅ 成功加载回归器权重！")
        
        self.physics_regressor.to(self.device, dtype=torch.float32)
        self.physics_regressor.eval()
        for param in self.physics_regressor.parameters():
            param.requires_grad = False

    def _load_normalization_stats(self, stats_file):
        stats_dict = {}
        df_stats = pd.read_csv(stats_file)
        for _, row in df_stats.iterrows():
            stats_dict[row['feature']] = {
                'method': row['method'], 'min': float(row['transform_min']), 'max': float(row['transform_max'])
            }
        return stats_dict

    def _load_catalog_by_indexs(self, catalog_file, target_set):
        catalog_dict = {}
        with open(catalog_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row.get('index', '')
                if idx in target_set:
                    catalog_dict[idx] = row
        return catalog_dict

    def _fill_default_cfg(self):
        defaults = {
            'dtype': 'fp16', 'amp': True, 'condition': None, 'emb_dir': 'embs/', 
            'clip_skip': 0, 'clip_final_norm': True, 'encoder_attention_mask': True, 
            'seed': None, 'offload': None, 'vae_optimize': {'tiling': False, 'slicing': False},
            'save': {'out_dir': 'output/', 'save_cfg': False, 'image_type': 'png', 'quality': 95},
            'ex_input': {},
        }
        for key, val in defaults.items():
            if key not in self.cfg:
                self.cfg[key] = val

    def extract_raw_features(self, row):
        feature_vector = []
        for col in self.feature_columns:
            val_str = row.get(col, '')
            if val_str in ('', 'NA_Value', '-99.0', 'NaN', 'nan', 'None'):
                val = np.nan
            else:
                try: val = float(val_str)
                except ValueError: val = np.nan
            if np.isnan(val) or np.isinf(val):
                feature_vector.append(0.5)
                continue
            feature_vector.append(val)
        return np.array(feature_vector, dtype=np.float32)

    def extract_normalized_features(self, row):
        feature_vector = []
        for col in self.feature_columns:
            val_str = row.get(col, '')
            if val_str in ('', 'NA_Value', '-99.0', 'NaN', 'nan', 'None'):
                val = np.nan
            else:
                try: val = float(val_str)
                except ValueError: val = np.nan

            if np.isnan(val) or np.isinf(val):
                feature_vector.append(0.5)
                continue

            if col in self.stats_dict:
                method = self.stats_dict[col]['method']
                t_min = self.stats_dict[col]['min']
                t_max = self.stats_dict[col]['max']

                if method == 'log10_minmax':
                    val = np.clip(val, 1e-8, None)
                    t_val = np.log10(val)
                elif method == 'asinh_minmax':
                    t_val = np.arcsinh(val)
                else: 
                    t_val = val

                norm_val = (t_val - t_min) / (t_max - t_min + 1e-8)
                norm_val = np.clip(norm_val, 0.0, 1.0)
                feature_vector.append(norm_val)
            else:
                feature_vector.append(0.5)
        return np.array(feature_vector, dtype=np.float32)

    def save_compare_image(self, idx_str, fake_img, compare_dir):
        try:
            real_img_path = os.path.join(self.real_img_root, f"{idx_str}.jpg")
            if os.path.exists(real_img_path):
                real_img = Image.open(real_img_path)
                size = real_img.size
            else:
                size = (192, 192)  
                real_img = Image.new('RGB', size, (0, 0, 0))
            
            fake_img = fake_img.resize(size)
            width, height = size
            compare_img = Image.new('RGB', (width * 2, height))
            compare_img.paste(real_img, (0, 0))
            compare_img.paste(fake_img, (width, 0))
            
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(compare_img)
            try: font = ImageFont.truetype("arial.ttf", 16)
            except: font = ImageFont.load_default()
            
            draw.text((5, 5), "Real", fill=(255, 255, 255), font=font)
            draw.text((width + 5, 5), f"Fake(DAPS L={LAMBDA_GUIDE})", fill=(255, 255, 255), font=font)
            compare_img.save(os.path.join(compare_dir, f"{idx_str}.png"))
        except Exception as e:
            print(f"[!] 保存对比图失败: {e}")

    def batch_infer(self, output_dir, batch_size, seed):
        fake_dir = os.path.join(output_dir, "fake_only")
        compare_dir = os.path.join(output_dir, "compare")
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(compare_dir, exist_ok=True)

        pipe = self.visualizer.pipe
        device = self.device
        
        try:
            remove_hook_from_module(pipe.vae, recurse=True)
            remove_hook_from_module(pipe.unet, recurse=True)
            remove_hook_from_module(pipe.text_encoder, recurse=True)
        except: pass 
        
        pipe.to(device)
        for module in [pipe.unet, pipe.text_encoder, pipe.vae]:
            for param in module.parameters():
                if param.device != device: param.data = param.data.to(device)
            for buffer in module.buffers():
                if buffer.device != device: buffer.data = buffer.data.to(device)
                    
        pipe_dtype = next(pipe.unet.parameters()).dtype
        pipe.vae.to(device, dtype=torch.float32)
        
        infer_args = OmegaConf.to_container(self.cfg.infer_args, resolve=True)
        num_inference_steps = infer_args.get("num_inference_steps", 50)
        height, width = infer_args.get("height", 192), infer_args.get("width", 192)
        guidance_scale = infer_args.get("guidance_scale", 7.5)
        do_classifier_free_guidance = guidance_scale > 1.0
        start_guide_step_idx = int(num_inference_steps * SKIP_RATIO)

        valid_indexs = [idx for idx in self.target_indexs if idx in self.catalog_dict]

        print(f"\n[*] 🚀 开始 DAPS 物理约束高并发生成 (Batch={batch_size}, CFG={guidance_scale}, Base Lambda={LAMBDA_GUIDE})")
        generator = torch.Generator(device=device).manual_seed(seed)

        for i in tqdm(range(0, len(valid_indexs), batch_size), desc="Batch Inference"):
            batch_idxs = valid_indexs[i : i + batch_size]
            current_bs = len(batch_idxs)
            
            raw_features_list = [self.extract_raw_features(self.catalog_dict[idx]) for idx in batch_idxs]
            norm_features_list = [self.extract_normalized_features(self.catalog_dict[idx]) for idx in batch_idxs]
            
            raw_feature_tensor = torch.tensor(np.array(raw_features_list), dtype=pipe_dtype, device=device)
            target_norm_tensor = torch.tensor(np.array(norm_features_list), dtype=torch.float32, device=device)

            try:
                with torch.no_grad():
                    emb, _ = pipe.text_encoder(raw_feature_tensor) 
                    if do_classifier_free_guidance:
                        null_feature_tensor = torch.zeros_like(raw_feature_tensor)
                        negative_emb, _ = pipe.text_encoder(null_feature_tensor)
                        emb = torch.cat([negative_emb, emb])

                pipe.scheduler.set_timesteps(num_inference_steps, device=device)
                num_channels_latents = pipe.unet.config.in_channels
                vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
                
                latents_shape = (current_bs, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
                latents = torch.randn(latents_shape, device=device, dtype=pipe_dtype, generator=generator)
                latents = latents * pipe.scheduler.init_noise_sigma

                for step_idx, t in enumerate(pipe.scheduler.timesteps):
                    t_tensor = t.to(device, dtype=torch.long) if torch.is_tensor(t) else torch.tensor(t, device=device, dtype=torch.long)
                    
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t_tensor)
                    
                    with torch.no_grad():
                        noise_pred = pipe.unet(latent_model_input, t_tensor, encoder_hidden_states=emb).sample
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
                    alpha_prod_t = alphas_cumprod[t_tensor]
                    beta_prod_t = 1 - alpha_prod_t
                    daps_weight = LAMBDA_GUIDE * alpha_prod_t

                    grad = 0.0
                    if step_idx >= start_guide_step_idx:
                        with torch.enable_grad():
                            latents_g = latents.detach().requires_grad_(True)
                            x0_hat_latent = (latents_g - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                            x0_hat_scaled = 1 / pipe.vae.config.scaling_factor * x0_hat_latent
                            x0_hat_scaled = x0_hat_scaled.to(device, dtype=pipe_dtype)
                            
                            total_loss = 0.0
                            grad_accumulator = torch.zeros_like(latents_g)
                            
                            for mb_start in range(0, current_bs, MICRO_BATCH):
                                mb_end = min(mb_start + MICRO_BATCH, current_bs)
                                is_last = (mb_end == current_bs) # 🚀 [修改 2] 动态释放计算图
                                
                                mb_x0_scaled_fp32 = x0_hat_scaled[mb_start:mb_end].to(torch.float32)
                                mb_target_fp32 = target_norm_tensor[mb_start:mb_end] 
                                
                                mb_pred_images = AutoencoderKL.decode(pipe.vae, mb_x0_scaled_fp32).sample
                                mb_pred_images = (mb_pred_images / 2 + 0.5).clamp(0, 1) 
                                
                                mb_pred_images_blurred = TF.gaussian_blur(mb_pred_images, kernel_size=5, sigma=2.0)
                                mb_pred_catalog = self.physics_regressor(mb_pred_images_blurred)
                                
                                mb_loss = F.mse_loss(mb_pred_catalog, mb_target_fp32, reduction='sum')
                                total_loss += mb_loss.item()
                                
                                # 使用 retain_graph=not is_last 防泄漏
                                mb_grad_full = torch.autograd.grad(mb_loss, latents_g, retain_graph=not is_last)[0]
                                grad_accumulator += mb_grad_full
                            
                            # 🚀 [修改 3] 严格数学对齐：除以总样本数与特征维度(22)，防止梯度暴增
                            grad = grad_accumulator / (current_bs * 22.0)

                    with torch.no_grad():
                        step_output = pipe.scheduler.step(noise_pred, t, latents)
                        prev_latents = step_output.prev_sample
                        
                        if isinstance(grad, torch.Tensor):
                            grad_fp32 = grad.to(torch.float32)
                            grad_norm = torch.norm(grad_fp32) + 1e-8
                            grad_dir = grad_fp32 / grad_norm
                            latent_std = prev_latents.std().to(torch.float32)
                            
                            physics_push = daps_weight * grad_dir * latent_std * 0.5
                            
                            # if step_idx % 10 == 0:
                            #     print(f"    Step {step_idx:02d} | Physics Loss: {(total_loss / (current_bs * 22)):.4f} | Push: {torch.norm(physics_push).item():.4f}")
                                
                            latents = prev_latents - physics_push.to(pipe_dtype)
                        else:
                            latents = prev_latents

                with torch.no_grad():
                    latents = 1 / pipe.vae.config.scaling_factor * latents
                    latents_fp32 = latents.to(torch.float32)
                    
                    images_tensor = []
                    for mb_start in range(0, current_bs, MICRO_BATCH):
                        mb_end = min(mb_start + MICRO_BATCH, current_bs)
                        mb_img = AutoencoderKL.decode(pipe.vae, latents_fp32[mb_start:mb_end]).sample
                        images_tensor.append(mb_img)
                    images_tensor = torch.cat(images_tensor)
                    
                    images_tensor = (images_tensor / 2 + 0.5).clamp(0, 1)
                    images_numpy = images_tensor.cpu().permute(0, 2, 3, 1).numpy()
                    
                from diffusers.utils import numpy_to_pil
                pil_images = numpy_to_pil(images_numpy)
                
                for idx_str, img in zip(batch_idxs, pil_images):
                    img.save(os.path.join(fake_dir, f"{idx_str}.png"))
                    self.save_compare_image(idx_str, img, compare_dir)

            except Exception as e:
                print(f"\n[!] Batch {i} 生成失败: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    inferencer = CatalogInferencerDPS(CFG_PATH, CATALOG_FILE, INDEX_FILE, REAL_IMG_ROOT, STATS_FILE, REGRESSOR_CKPT)
    inferencer.batch_infer(OUT_DIR, batch_size=BATCH_SIZE, seed=SEED)
    print("\n[√] 物理注入推理任务圆满完成！")