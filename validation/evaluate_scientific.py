#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 🔧 [配置区]
# =====================================================================
IMG_DIR = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/validation/val_results_100_dps/raw_only"
CAT_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/validation/validation_catalog_100.csv"
STATS_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/normalization_stats_v3.csv"
REGRESSOR_CKPT = "regressor_output/best_regressor_v4.pth" # 确保路径与你存放的物理裁判一致

FEATURE_COLUMNS = [
    'DESIDR1_FLUX_G', 'DESIDR1_FLUX_R', 'DESIDR1_FLUX_Z', 'DESIDR1_FLUX_W1', 'DESIDR1_FLUX_W2',
    'DESIDR1_SHAPE_R', 'DESIDR1_SHAPE_E1', 'DESIDR1_SHAPE_E2', 'DESIDR1_SERSIC',
    'mapped_smooth_combined_score', 'mapped_barred_spirals_combined_score',
    'mapped_unbarred_spirals_combined_score', 'mapped_edge_on_with_bulge_combined_score',
    'mapped_mergers_combined_score', 'mapped_irregular_combined_score', 'DESIDR1_Z',
    'MassEMLine_MASS_CG', 'MassEMLine_SFR_CG', 'MassEMLine_AGE_CG',
    'MassEMLine_HALPHA_FLUX', 'MassEMLine_OIII5007_FLUX', 'MassEMLine_EBV',
]
# =====================================================================

class GalaxyRegressor(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        return self.output_act(self.backbone(x))

class ScientificEvaluator:
    def __init__(self):
        print("[*] 正在初始化天体物理科学验证引擎 (Regressor-based)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载物理裁判
        self.model = GalaxyRegressor(22).to(self.device)
        self.model.load_state_dict(torch.load(REGRESSOR_CKPT, map_location=self.device))
        self.model.eval()
        
        # 2. 加载归一化字典
        self.stats_dict = {}
        df_stats = pd.read_csv(STATS_FILE)
        for _, row in df_stats.iterrows():
            self.stats_dict[row['feature']] = {
                'method': row['method'], 'min': float(row['transform_min']), 'max': float(row['transform_max'])
            }
            
        # 3. 加载目标星表
        self.cat = pd.read_csv(CAT_FILE)
        self.cat['index_str'] = self.cat['index'].astype(str)
        self.cat.set_index('index_str', inplace=True)

    def extract_normalized_features(self, row):
        feature_vector = []
        for col in FEATURE_COLUMNS:
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

    def run(self):
        img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
        if not img_files:
            print(f"[!] 在 {IMG_DIR} 未找到图像！")
            return
            
        print(f"[*] 找到 {len(img_files)} 张生成图像，正在进行神经常微分物理预测...")
        
        preds_list = []
        targets_list = []
        
        with torch.no_grad():
            for fname in tqdm(img_files):
                idx = str(os.path.splitext(fname)[0])
                if idx not in self.cat.index: continue
                
                # 提取目标 Ground Truth
                row = self.cat.loc[idx]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                target_norm = self.extract_normalized_features(row)
                
                # 读取图像并进行预测
                img = Image.open(os.path.join(IMG_DIR, fname)).convert('RGB')
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                # 适配 ResNet 的输入标准化 (ImageNet 标准) - 如果你训练 regressor 时用了，请保留；否则注释掉
                # img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
                
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                pred_norm = self.model(img_tensor).cpu().numpy()[0]
                
                preds_list.append(pred_norm)
                targets_list.append(target_norm)
                
        preds_np = np.array(preds_list)
        targets_np = np.array(targets_list)
        
        print("\n==================================================")
        print(" 🌌 GalaxySD + DAPS 物理一致性终极报告")
        print("==================================================")
        
        for i, col_name in enumerate(FEATURE_COLUMNS):
            p_val = preds_np[:, i]
            t_val = targets_np[:, i]
            # 过滤掉 0.5 的缺失值填充默认项
            valid_mask = (t_val != 0.5)
            
            if np.sum(valid_mask) > 10:
                corr, _ = spearmanr(p_val[valid_mask], t_val[valid_mask])
                print(f" -> {col_name:40s} | Spearman Corr: {corr:+.4f}")
            else:
                print(f" -> {col_name:40s} | Spearman Corr: N/A (数据不足)")
        print("==================================================\n")

if __name__ == "__main__":
    ScientificEvaluator().run()