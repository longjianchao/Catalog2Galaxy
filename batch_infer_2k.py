#!/usr/bin/env python3
"""
大规模验证集批量推理脚本：2.5k 星表样本生成 (复合物理归一化修复版)
特性：支持 GPU Batch 并行、复合 Z-score/MinMax 归一化、严谨的 CFG 双路推断
"""

import os
import csv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from hcpdiff.visualizer import Visualizer
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 🔧 [配置区] 
# =====================================================================
CFG_PATH = "cfgs/infer/text2img_galaxy_catalog.yaml"  
CATALOG_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/labeled_catalog_cleaned_v3.csv"
INDEX_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/validation/validation_index.txt" 
REAL_IMG_ROOT = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/filtered_data" 
OUT_DIR = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/validation/val_results_150k" 

# 🚀 指向你刚才发给我的那个 normv3 文件
STATS_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/normalization_stats_v3.csv"

BATCH_SIZE = 16    
SEED = 114514      
# =====================================================================


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


class CatalogInferencer:
    def __init__(self, cfg_path, catalog_file, index_file, real_img_root, stats_file):
        self.cfg = OmegaConf.load(cfg_path)
        self._fill_default_cfg()
        self.real_img_root = real_img_root

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

        # 1. 极其精准地加载你的 3 种归一化参数
        self.stats_dict = self._load_normalization_stats(stats_file)
        
        # 2. 从 txt 文件加载 Target Indexs
        with open(index_file, 'r') as f:
            self.target_indexs = [line.strip() for line in f.readlines() if line.strip()]
        
        target_set = set(self.target_indexs)
        print(f"[*] 从 txt 文件中读取到 {len(self.target_indexs)} 个目标 index...")

        # 3. 从 CSV 提取对应数据
        self.catalog_dict = self._load_catalog_by_indexs(catalog_file, target_set)
        
        # 4. 初始化 Pipeline
        print("[*] 正在加载模型权重与 Pipeline...")
        self.visualizer = CustomVisualizer(self.cfg)
        self.visualizer.pipe.set_progress_bar_config(disable=True)

    def _load_normalization_stats(self, stats_file):
        print(f"[*] 正在加载复合归一化参数: {stats_file}")
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
        """🚀 核心区：严格复刻训练时的多模态归一化逻辑"""
        feature_vector = []
        for col in self.feature_columns:
            val_str = row.get(col, '')
            
            # 解析原始数值
            if val_str in ('', 'NA_Value', '-99.0', 'NaN', 'nan', 'None'):
                val = np.nan
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    val = np.nan

            # 如果数据缺失，给个最安全的中心值（归一化后的 0.5）
            if np.isnan(val) or np.isinf(val):
                feature_vector.append(0.5)
                continue

            # 核心数学变换
            if col in self.stats_dict:
                method = self.stats_dict[col]['method']
                t_min = self.stats_dict[col]['min']
                t_max = self.stats_dict[col]['max']

                if method == 'log10_minmax':
                    # 【重要修复】防爆破：遇到 <= 0 的数值，强行拉平到极其微小的正数，防止 log10(负数) 报错 NaN
                    val = np.clip(val, 1e-8, None)
                    t_val = np.log10(val)
                elif method == 'asinh_minmax':
                    t_val = np.arcsinh(val)
                else: # linear_minmax
                    t_val = val

                # Min-Max 缩放
                norm_val = (t_val - t_min) / (t_max - t_min + 1e-8)
                
                # 防越界截断
                norm_val = np.clip(norm_val, 0.0, 1.0)
                
                # 💡 提示：如果你在 Dataset 里把 [0, 1] 又转成了 [-1, 1]，请把下面这行取消注释：
                # norm_val = norm_val * 2.0 - 1.0

                feature_vector.append(norm_val)
            else:
                feature_vector.append(0.5)
            
        return np.array(feature_vector, dtype=np.float32)

    def batch_infer(self, output_dir, batch_size, seed):
        fake_dir = os.path.join(output_dir, "fake_only")
        compare_dir = os.path.join(output_dir, "compare")
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(compare_dir, exist_ok=True)

        device = self.visualizer.pipe.device
        infer_args = OmegaConf.to_container(self.cfg.infer_args, resolve=True)

        self.visualizer.pipe.text_encoder.to(device, dtype=torch.float16)
        generator = torch.Generator(device=device).manual_seed(seed)
        infer_args['generator'] = generator

        valid_indexs = [idx for idx in self.target_indexs if idx in self.catalog_dict]

        print(f"\n[*] 🚀 开始物理约束的高并发生成，Batch Size = {batch_size}")
        
        for i in tqdm(range(0, len(valid_indexs), batch_size), desc="Batch Inference"):
            batch_idxs = valid_indexs[i : i + batch_size]
            
            # 这里拿到的特征已经是完美匹配训练集的归一化特征了！
            raw_features_list = [self.extract_raw_features(self.catalog_dict[idx]) for idx in batch_idxs]
            feature_tensor = torch.tensor(np.array(raw_features_list), dtype=torch.float16).to(device)

            try:
                with torch.no_grad():
                    # 1. 带着物理条件预测
                    emb, _ = self.visualizer.pipe.text_encoder(feature_tensor)
                    
                    # 2. 修复 CFG！真正的无条件基准线
                    null_feature_tensor = torch.zeros_like(feature_tensor).to(device)
                    negative_emb, _ = self.visualizer.pipe.text_encoder(null_feature_tensor)

                    # 3. 生成
                    images = self.visualizer.pipe(
                        prompt_embeds=emb,
                        negative_prompt_embeds=negative_emb,
                        **infer_args
                    ).images

                for idx_str, img in zip(batch_idxs, images):
                    # 保存生成的图像
                    img.save(os.path.join(fake_dir, f"{idx_str}.png"))
                    
                    # 保存对比图
                    self.save_compare_image(idx_str, img, compare_dir)

            except Exception as e:
                print(f"\n[!] Batch {i} 生成失败: {e}")
    
    def save_compare_image(self, idx_str, fake_img, compare_dir):
        """保存原图和生成图像的对比图"""
        try:
            # 加载原图
            real_img_path = os.path.join(self.real_img_root, f"{idx_str}.jpg")
            if os.path.exists(real_img_path):
                real_img = Image.open(real_img_path)
                # 获取原图的实际大小
                size = real_img.size
            else:
                # 如果原图不存在，使用黑色图像作为替代
                size = (192, 192)  # 默认大小
                real_img = Image.new('RGB', size, (0, 0, 0))
            
            # 调整生成图像的大小，确保与原图一致
            fake_img = fake_img.resize(size)
            
            # 创建对比图
            width, height = size
            compare_img = Image.new('RGB', (width * 2, height))
            compare_img.paste(real_img, (0, 0))
            compare_img.paste(fake_img, (width, 0))
            
            # 添加标记
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(compare_img)
            # 使用默认字体，设置合适的字体大小
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # 添加 Real 和 Fake 标记
            draw.text((5, 5), "Real", fill=(255, 255, 255), font=font)
            draw.text((width + 5, 5), "Fake", fill=(255, 255, 255), font=font)
            
            # 保存对比图
            compare_img.save(os.path.join(compare_dir, f"{idx_str}.png"))
        except Exception as e:
            print(f"[!] 保存对比图失败: {e}")

if __name__ == "__main__":
    inferencer = CatalogInferencer(CFG_PATH, CATALOG_FILE, INDEX_FILE, REAL_IMG_ROOT, STATS_FILE)
    inferencer.batch_infer(OUT_DIR, batch_size=BATCH_SIZE, seed=SEED)
    print("\n[√] 物理注入推理任务圆满完成！")