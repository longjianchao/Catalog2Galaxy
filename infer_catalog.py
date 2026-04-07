#!/usr/bin/env python3
"""
GalaxySD - DPS 物理引导推理脚本 (终极架构版)
融合了 AdaGN 物理场兼容接口、高斯低通滤波防作弊、以及潜变量方差自适应步长。
"""

import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from hcpdiff.visualizer import Visualizer
from omegaconf import OmegaConf
from accelerate.hooks import remove_hook_from_module
import torch.nn as nn
from torchvision import models
from diffusers.models import AutoencoderKL

# ==============================================================================
# 🔴 GalaxySD 物理教练网络架构
# ==============================================================================
class GalaxyRegressor(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        return self.output_act(self.backbone(x))
# ==============================================================================

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
    def __init__(self, cfg_path, catalog_file, target_indexs, regressor_ckpt_path, stats_file):
        self.cfg = OmegaConf.load(cfg_path)
        self._fill_default_cfg()

        self.feature_columns = [
            'DESIDR1_FLUX_G', 'DESIDR1_FLUX_R', 'DESIDR1_FLUX_Z',
            'DESIDR1_FLUX_W1', 'DESIDR1_FLUX_W2',
            'DESIDR1_SHAPE_R', 'DESIDR1_SHAPE_E1', 'DESIDR1_SHAPE_E2', 'DESIDR1_SERSIC',
            'mapped_smooth_combined_score', 'mapped_barred_spirals_combined_score',
            'mapped_unbarred_spirals_combined_score', 'mapped_edge_on_with_bulge_combined_score',
            'mapped_mergers_combined_score', 'mapped_irregular_combined_score',
            'DESIDR1_Z',
            'MassEMLine_MASS_CG', 'MassEMLine_SFR_CG', 'MassEMLine_AGE_CG',
            'MassEMLine_HALPHA_FLUX', 'MassEMLine_OIII5007_FLUX', 'MassEMLine_EBV',
        ]
        
        self.stats_dict = self._load_normalization_stats(stats_file)

        target_set = {str(i) for i in target_indexs}
        print(f"[*] 从星表中提取 {len(target_set)} 条目标记录...")
        self.catalog_dict = self._load_catalog_by_indexs(catalog_file, target_set)

        self.visualizer = CustomVisualizer(self.cfg)
        self.device = self.visualizer.pipe.device

        print("[*] 正在加载物理回归器 (Physics Regressor)...")
        self.physics_regressor = GalaxyRegressor(22) 
        if regressor_ckpt_path and os.path.exists(regressor_ckpt_path):
            self.physics_regressor.load_state_dict(torch.load(regressor_ckpt_path, map_location='cpu'))
            print("  ✅ 成功加载回归器权重！")
        self.physics_regressor.to(self.device)
        self.physics_regressor.eval()
        for param in self.physics_regressor.parameters():
            param.requires_grad = False

    def _load_normalization_stats(self, stats_file):
        stats_dict = {}
        df_stats = pd.read_csv(stats_file)
        for _, row in df_stats.iterrows():
            stats_dict[row['feature']] = {
                'method': row['method'],
                'min': float(row['transform_min']),
                'max': float(row['transform_max'])
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
            'dtype': 'fp32', 'amp': True, 'condition': None,
            'emb_dir': 'embs/', 'clip_skip': 0, 'clip_final_norm': True,
            'encoder_attention_mask': True, 'seed': None, 'offload': None,
            'vae_optimize': {'tiling': False, 'slicing': False},
            'save': {'out_dir': 'output/', 'save_cfg': True, 'image_type': 'png', 'quality': 95},
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
                val = 0.0
            else:
                try:
                    val = float(val_str)
                    if np.isnan(val) or np.isinf(val): val = 0.0
                except:
                    val = 0.0
            feature_vector.append(val)
        return np.array(feature_vector, dtype=np.float32)

    def extract_normalized_features(self, row):
        feature_vector = []
        for col in self.feature_columns:
            val_str = row.get(col, '')
            if val_str in ('', 'NA_Value', '-99.0', 'NaN', 'nan', 'None'):
                val = np.nan
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    val = np.nan

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

    def infer_from_indexs(self, indexs, output_dir, lambda_guide=10.0, skip_ratio=0.3):
        os.makedirs(output_dir, exist_ok=True)
        infer_args = OmegaConf.to_container(self.cfg.infer_args, resolve=True)
        num_inference_steps = infer_args.get("num_inference_steps", 50)
        height, width = infer_args.get("height", 192), infer_args.get("width", 192)
        start_guide_step_idx = int(num_inference_steps * skip_ratio)

        for index in indexs:
            index_str = str(index)
            print(f"\n{'='*50}")
            print(f"🚀 正在生成 index: {index_str} (DPS Lambda: {lambda_guide})")

            if index_str not in self.catalog_dict:
                continue

            row = self.catalog_dict[index_str]
            raw_features = self.extract_raw_features(row)
            norm_features = self.extract_normalized_features(row)

            try:
                pipe = self.visualizer.pipe
                try:
                    remove_hook_from_module(pipe.vae, recurse=True)
                    remove_hook_from_module(pipe.unet, recurse=True)
                    remove_hook_from_module(pipe.text_encoder, recurse=True)
                except:
                    pass 
                
                pipe.to(self.device)
                for module in [pipe.unet, pipe.text_encoder, pipe.vae]:
                    for param in module.parameters():
                        if param.device != self.device: param.data = param.data.to(self.device)
                    for buffer in module.buffers():
                        if buffer.device != self.device: buffer.data = buffer.data.to(self.device)
                            
                pipe_dtype = next(pipe.unet.parameters()).dtype

                raw_feature_tensor = torch.tensor(raw_features, dtype=pipe_dtype, device=self.device).unsqueeze(0)
                target_norm_tensor = torch.tensor(norm_features, dtype=pipe_dtype, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    # 🔴 完美对接新架构：emb 的形状将是纯净的 [1, 22, 768]
                    # 我们只需要索引 0 的 feature_embeddings，不需要重构特征
                    emb_output = pipe.text_encoder(raw_feature_tensor)
                    emb = emb_output[0].to(device=self.device, dtype=pipe_dtype)
                
                pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
                num_channels_latents = pipe.unet.config.in_channels
                vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
                
                latents_shape = (1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
                latents = torch.randn(latents_shape, device=self.device, dtype=pipe_dtype)
                latents = latents * pipe.scheduler.init_noise_sigma

                for i, t in enumerate(pipe.scheduler.timesteps):
                    t_tensor = t.to(self.device, dtype=torch.long) if torch.is_tensor(t) else torch.tensor(t, device=self.device, dtype=torch.long)
                    latent_model_input = pipe.scheduler.scale_model_input(latents, t_tensor)
                    
                    with torch.enable_grad():
                        latents_g = latents.detach().requires_grad_(True)
                        latent_model_input_g = pipe.scheduler.scale_model_input(latents_g, t_tensor)
                        
                        noise_pred = pipe.unet(latent_model_input_g, t_tensor, encoder_hidden_states=emb).sample
                        
                        grad = 0.0
                        if i >= start_guide_step_idx:
                            # 🚀 DAPS 核心机制植入
                            alphas_cumprod = pipe.scheduler.alphas_cumprod.to(self.device)
                            alpha_prod_t = alphas_cumprod[t_tensor]
                            beta_prod_t = 1 - alpha_prod_t
                            daps_weight = lambda_guide * alpha_prod_t
                            
                            x0_hat_latent = (latents_g - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                            x0_hat_scaled = 1 / pipe.vae.config.scaling_factor * x0_hat_latent
                            x0_hat_scaled = x0_hat_scaled.to(self.device, dtype=pipe_dtype)
                            
                            pred_images = AutoencoderKL.decode(pipe.vae, x0_hat_scaled).sample
                            pred_images = (pred_images / 2 + 0.5).clamp(0, 1) 
                            
                            # 🔴 高通对抗粉碎：用高斯模糊逼迫网络学习低频真实物理轮廓
                            pred_images_blurred = TF.gaussian_blur(pred_images, kernel_size=5, sigma=2.0)
                            
                            pred_catalog = self.physics_regressor(pred_images_blurred)
                            loss_physics = F.mse_loss(pred_catalog, target_norm_tensor)
                            grad = torch.autograd.grad(loss_physics, latents_g)[0]
                            
                            if i % 10 == 0:
                                print(f"  Step {i:02d} | Physics Loss: {loss_physics.item():.4f}")

                    with torch.no_grad():
                        step_output = pipe.scheduler.step(noise_pred, t, latents)
                        prev_latents = step_output.prev_sample
                        
                        if isinstance(grad, torch.Tensor):
                            grad_fp32 = grad.to(torch.float32)
                            # 🔴 自适应流形保护步长：按潜变量当前方差动态控制推力
                            grad_norm = grad_fp32.view(1, -1).norm(dim=1).view(1, 1, 1, 1) + 1e-8
                            grad_dir = grad_fp32 / grad_norm
                            latent_std = prev_latents.to(torch.float32).view(1, -1).std(dim=1).view(1, 1, 1, 1)
                            
                            physics_push = daps_weight * grad_dir * latent_std * 0.05
                            latents = prev_latents - physics_push.to(pipe_dtype)
                        else:
                            latents = prev_latents

                with torch.no_grad():
                    latents = 1 / pipe.vae.config.scaling_factor * latents
                    pipe.vae.to(self.device)
                    image = AutoencoderKL.decode(pipe.vae, latents).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()
                    
                from diffusers.utils import numpy_to_pil
                images = numpy_to_pil(image)
                
                save_path = os.path.join(output_dir, f"{index_str}.png")
                images[0].save(save_path)
                print(f"  ✅ 已保存至: {save_path}")

            except Exception as e:
                print(f"  ❌ 生成失败: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--catalog", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--indexs", type=str, nargs='+', required=True)
    parser.add_argument("--regressor_ckpt", type=str, default="")
    parser.add_argument("--stats_file", type=str, required=True)
    parser.add_argument("--lambda_guide", type=float, default=10.0)
    parser.add_argument("--skip_ratio", type=float, default=0.2)
    args = parser.parse_args()

    inferencer = CatalogInferencerDPS(args.cfg, args.catalog, args.indexs, args.regressor_ckpt, args.stats_file)
    inferencer.infer_from_indexs(args.indexs, args.out_dir, lambda_guide=args.lambda_guide, skip_ratio=args.skip_ratio)
    print("\n全部推理任务完成！")