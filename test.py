# verify_adagn.py
import torch
import sys
sys.path.insert(0, '/nfsdata/share/ljc/GalaxySD')

from hcpdiff.models.textencoder_catalog import CatalogTextEncoder
from safetensors.torch import load_file

# ── 1. 初始化TE结构 ──────────────────────────────────────────────────
te = CatalogTextEncoder(
    feature_dim=22,
    hidden_dim=1024,
    output_dim=768,
    num_tokens=22,
    norm_stats_path="/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/normalization_stats_v3.csv"
)

# ── 2. 加载当前checkpoint ─────────────────────────────────────────────
# 换成你当前83k步或最新的TE checkpoint路径
ckpt_path = "exps/2026-04-12-22-46-03/ckpts/text_encoder-80000.safetensors"
state_dict = load_file(ckpt_path)

# hcpdiff保存时key带"base:"前缀，需要去掉
state_dict = {k.replace("base:", ""): v for k, v in state_dict.items()}
missing, unexpected = te.load_state_dict(state_dict, strict=False)
print(f"missing keys:    {missing}")
print(f"unexpected keys: {unexpected}")

te.eval()

# ── 3. 检查adagn_projector权重是否离开零 ─────────────────────────────
print("\n=== adagn_projector权重状态 ===")
for name, param in te.adagn_projector.named_parameters():
    print(f"  {name}: norm={param.norm():.6f}, std={param.std():.6f}")

# ── 4. 跑一次forward，看adagn_emb幅值 ───────────────────────────────
print("\n=== forward输出验证 ===")
# 用两组不同的物理参数，看adagn_emb是否有区分度
dummy_input_1 = torch.zeros(1, 22)   # 全零（无条件）
dummy_input_2 = torch.ones(1, 22)    # 全1
dummy_input_3 = torch.randn(1, 22).abs()  # 随机正值模拟真实参数

with torch.no_grad():
    for i, inp in enumerate([dummy_input_1, dummy_input_2, dummy_input_3]):
        _ = te(inp)
        emb = te.last_adagn_emb
        print(f"  input_{i+1}: adagn_emb norm={emb.norm():.6f}, "
              f"std={emb.std():.6f}, "
              f"mean={emb.mean():.6f}")

# ── 5. 验证两组不同参数的adagn_emb是否有区分度 ──────────────────────
print("\n=== 不同参数间的adagn_emb区分度 ===")
with torch.no_grad():
    _ = te(dummy_input_2)
    emb2 = te.last_adagn_emb.clone()
    _ = te(dummy_input_3)
    emb3 = te.last_adagn_emb.clone()
    diff = (emb2 - emb3).norm()
    print(f"  两组不同参数的adagn_emb差值norm: {diff:.6f}")
    print(f"  如果接近0，说明adagn_projector没有被有效训练")
    print(f"  如果>1，说明有区分度，AdaGN通道在工作")