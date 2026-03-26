import torch
import pandas as pd
import numpy as np
from hcpdiff.models.textencoder_catalog import CatalogTextEncoder

# 加载模型
te = CatalogTextEncoder(
    feature_dim=22, hidden_dim=1024, output_dim=768,
    norm_stats_path="/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/normalization_stats_v3.csv"
)

# 加载训练好的权重
import safetensors.torch
sd = safetensors.torch.load_file("exps/2026-03-23-13-27-39/ckpts/text_encoder-90000.safetensors")

# 处理权重字典，移除 base: 前缀
new_sd = {}
for k, v in sd.items():
    if k.startswith('base:'):
        new_k = k[5:]  # 移除 'base:' 前缀
        new_sd[new_k] = v
    else:
        new_sd[k] = v

# 加载权重，忽略缺失的 buffer 参数
te.load_state_dict(new_sd, strict=False)
te.eval()

# 取两条极端不同的星表记录
df = pd.read_csv("/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/labeled_catalog_cleaned_v3.csv")
df = df[(df['DESIDR1_SHAPE_R'] > 6.4) & (df['DESIDR1_SHAPE_R'] < 9.6)]

feature_cols = [
    'DESIDR1_FLUX_G', 'DESIDR1_FLUX_R', 'DESIDR1_FLUX_Z',
    'DESIDR1_FLUX_W1', 'DESIDR1_FLUX_W2',
    'DESIDR1_SHAPE_R', 'DESIDR1_SHAPE_E1', 'DESIDR1_SHAPE_E2', 'DESIDR1_SERSIC',
    'mapped_smooth_combined_score', 'mapped_barred_spirals_combined_score',
    'mapped_unbarred_spirals_combined_score', 'mapped_edge_on_with_bulge_combined_score',
    'mapped_mergers_combined_score', 'mapped_irregular_combined_score',
    'DESIDR1_Z', 'MassEMLine_MASS_CG', 'MassEMLine_SFR_CG', 'MassEMLine_AGE_CG',
    'MassEMLine_HALPHA_FLUX', 'MassEMLine_OIII5007_FLUX', 'MassEMLine_EBV',
]

# 椭圆星系
row_e = df[df['mapped_smooth_combined_score'] > 5].iloc[0]
# 旋涡星系
row_s = df[df['mapped_unbarred_spirals_combined_score'] > 5].iloc[0]

feat_e = torch.tensor(row_e[feature_cols].values.astype(float), dtype=torch.float32).unsqueeze(0)
feat_s = torch.tensor(row_s[feature_cols].values.astype(float), dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    emb_e, _ = te(feat_e)
    emb_s, _ = te(feat_s)

# 计算两个 embedding 的余弦相似度
cos_sim = torch.nn.functional.cosine_similarity(
    emb_e.flatten(), emb_s.flatten(), dim=0
).item()

print(f"椭圆星系 embedding 均值: {emb_e.mean().item():.4f}, 标准差: {emb_e.std().item():.4f}")
print(f"旋涡星系 embedding 均值: {emb_s.mean().item():.4f}, 标准差: {emb_s.std().item():.4f}")
print(f"两者余弦相似度: {cos_sim:.4f}")
print("（如果相似度 > 0.99，说明 embedding 已经坍缩，条件信号失效）")