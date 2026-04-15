#!/usr/bin/env python3
"""
GalaxySD 物理回归器 (Physics Regressor) 训练脚本
特性：
1. 严格复刻 V3 版复合物理归一化
2. 动态标签修正：支持空间几何增强的同时，保护 E2 的物理一致性
3. 修复 I/O 投毒问题与冗余推理导致的算力浪费
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. 🔧 超参数配置区
# =====================================================================
IMG_DIR = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/filtered_data"
CATALOG_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/labeled_catalog_cleaned_v3.csv"
STATS_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/normalization_stats_v3.csv"
OUTPUT_DIR = "regressor_output"

BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 22维特征严格对齐 (注意: index 7 是 DESIDR1_SHAPE_E2)
FEATURE_COLUMNS = [
    'DESIDR1_FLUX_G', 'DESIDR1_FLUX_R', 'DESIDR1_FLUX_Z',
    'DESIDR1_FLUX_W1', 'DESIDR1_FLUX_W2',
    'DESIDR1_SHAPE_R', 'DESIDR1_SHAPE_E1', 'DESIDR1_SHAPE_E2', 'DESIDR1_SERSIC',
    'mapped_smooth_combined_score',
    'mapped_barred_spirals_combined_score',
    'mapped_unbarred_spirals_combined_score',
    'mapped_edge_on_with_bulge_combined_score',
    'mapped_mergers_combined_score',
    'mapped_irregular_combined_score',
    'DESIDR1_Z',
    'MassEMLine_MASS_CG', 'MassEMLine_SFR_CG', 'MassEMLine_AGE_CG',
    'MassEMLine_HALPHA_FLUX', 'MassEMLine_OIII5007_FLUX',
    'MassEMLine_EBV'
]

# =====================================================================
# 2. 📊 数据集与 V3 复合归一化处理
# =====================================================================
class GalaxyRegressionDataset(Dataset):
    def __init__(self, df, img_dir, stats_file, transform=None, is_train=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train  # 标识是否为训练集，决定是否触发动态增强
        
        # 加载归一化统计参数
        self.stats_dict = self._load_normalization_stats(stats_file)
        
        # 预先处理并缓存所有目标的归一化特征
        print(f"正在提取并执行 V3 复合归一化处理 (is_train={is_train})...")
        norm_features = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            norm_vec = self.extract_normalized_features(row)
            norm_features.append(norm_vec)
            
        # 目标值现在被严密地限制在了 [0.0, 1.0] 区间内
        self.norm_features = np.array(norm_features, dtype=np.float32)

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

    def extract_normalized_features(self, row):
        feature_vector = []
        for col in FEATURE_COLUMNS:
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.loc[idx, 'index']) + ".jpg" 
        img_path = os.path.join(self.img_dir, img_name)
        
        # 防投毒拦截
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"数据集缺失错误: 未找到图像文件 {img_path}")
            
        image = Image.open(img_path).convert('RGB')
        
        # ⚠️ 必须 copy！防止在缓存上做 in-place 修改导致永久污染
        target = self.norm_features[idx].copy()

        # ==========================================================
        # 🌟 方案二核心：动态数据增强与标签同步修正
        # 物理数学：E2在水平和垂直翻转下均会反转符号。
        # 归一化空间中，符号反转 E2 -> -E2 等价于 1.0 - target_E2
        # ==========================================================
        if self.is_train:
            # 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                target[7] = 1.0 - target[7]  # 修正 E2
                
            # 随机垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                target[7] = 1.0 - target[7]  # 修正 E2
        
        # 执行统一的非几何变换 (Resize, ToTensor)
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(target, dtype=torch.float32)

# =====================================================================
# 3. 🧠 物理教练网络架构 (ResNet-18)
# =====================================================================
class GalaxyRegressor(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        # 加 Sigmoid 约束输出域至 [0, 1]
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        return self.output_act(self.backbone(x))

# =====================================================================
# 4. 🏃‍♂️ 训练主函数
# =====================================================================
def main():
    print("🚀 启动 GalaxySD 物理教练 (V3 归一化 + 动态标签修正版) 训练管线...")
    
    # 这里的 transform 仅保留尺寸调整与张量转换，几何增强已移至 Dataset 内部
    transform_base = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(), 
    ])

    print("[*] 读取星表数据...")
    df_full = pd.read_csv(CATALOG_FILE)
    df_train, df_val = train_test_split(df_full, test_size=0.1, random_state=42)
    
    print("[*] 构建训练集...")
    # 开启 is_train=True 触发动态翻转和标签修正
    train_dataset = GalaxyRegressionDataset(df_train, IMG_DIR, STATS_FILE, transform=transform_base, is_train=True)
    print("[*] 构建验证集...")
    val_dataset = GalaxyRegressionDataset(df_val, IMG_DIR, STATS_FILE, transform=transform_base, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = GalaxyRegressor(num_classes=len(FEATURE_COLUMNS)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    def calculate_val_spearman(model, val_loader):
        """计算验证集每维参数的 Spearman 相关性"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        spearman_corrs = []
        for i in range(all_preds.shape[1]):
            valid_mask = (all_targets[:, i] != 0.5)
            if np.sum(valid_mask) > 10:
                corr, _ = spearmanr(all_preds[valid_mask, i], all_targets[valid_mask, i])
                spearman_corrs.append(corr)
            else:
                spearman_corrs.append(np.nan)
        
        return spearman_corrs, np.nanmean(spearman_corrs)
    
    best_val_loss = float('inf')
    best_val_spearman = -1.0

    print(f"\n[*] 开始训练，共计 {EPOCHS} 轮...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]")
        for images, targets in pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_dataset)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val  ]")
            for images, targets in pbar_val:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                pbar_val.set_postfix({'Loss': f"{loss.item():.4f}"})
                
        val_loss /= len(val_dataset)
        
        # 消除冗余推理：每轮仅做一次 Spearman 计算并复用
        current_spearman_corrs, current_mean_spearman = calculate_val_spearman(model, val_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n📊 Epoch {epoch+1} Spearman 诊断:")
            for i, col in enumerate(FEATURE_COLUMNS):
                corr = current_spearman_corrs[i]
                if np.isnan(corr):
                    status = "N/A"
                elif corr > 0.8:
                    status = f"ρ={corr:+.4f} ✅"
                elif corr > 0.6:
                    status = f"ρ={corr:+.4f} ⚠️"
                else:
                    status = f"ρ={corr:+.4f} ❌"
                print(f"  {col:35s} → {status}")
            print(f"  平均 Spearman: {current_mean_spearman:.4f}\n")
        
        print(f"👉 Epoch {epoch+1:02d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | Val Spearman: {current_mean_spearman:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if current_mean_spearman > best_val_spearman:
            best_val_spearman = current_mean_spearman
            save_path = os.path.join(OUTPUT_DIR, "best_regressor_v4.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⭐ 验证集 Spearman 创新高 ({current_mean_spearman:.4f})，模型已保存至 {save_path}")

if __name__ == "__main__":
    main()