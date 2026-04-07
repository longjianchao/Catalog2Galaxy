from typing import Tuple, Optional, List

import torch
from torch import nn
import pandas as pd

class CatalogTextEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, norm_stats_path, num_tokens=22):
        """
        初始化纯物理驱动的星表特征编码器 (去 Padding + 重建自监督架构)

        Args:
            feature_dim: 星表特征向量的维度（22维）
            hidden_dim: MLP隐藏层的维度
            output_dim: 输出向量的维度 (SD1.5 默认为 768)
            norm_stats_path: 归一化参数CSV文件路径
            num_tokens: 映射成的物理 Token 数量 (直接等于 feature_dim，即 22)
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.num_embeddings = 49408
        self.embedding_dim = output_dim

        # ── 1. 保留核心：物理归一化参数加载 ─────────────────────────
        if norm_stats_path is None:
            raise ValueError("norm_stats_path 必须提供")

        norm_df = pd.read_csv(norm_stats_path)
        assert len(norm_df) == feature_dim, (
            f"norm_stats 有 {len(norm_df)} 个特征，但 feature_dim={feature_dim}"
        )

        self.norm_methods = norm_df['method'].tolist()
        self.register_buffer('norm_mins', torch.tensor(norm_df['transform_min'].values, dtype=torch.float32))
        self.register_buffer('norm_maxs', torch.tensor(norm_df['transform_max'].values, dtype=torch.float32))

        # ── 2. 基座 MLP ──────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_tokens * output_dim)
        )

        # ── 3. 物理位置编码 ──────────────────────────────────────
        # 给每一个物理 Token 一个专属的身份标识
        self.position_embedding = nn.Parameter(torch.randn(1, num_tokens, output_dim) * 0.02)

        # ── 4. 🛡️ 植入防御机制：物理信息重建解码器 ─────────────────
        # 信息瓶颈：强制高维 Token 包含足以还原原始 22 维参数的信息
        self.physics_decoder = nn.Sequential(
            nn.Linear(num_tokens * output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        [完全保留] 极其安全的非 In-place 归一化函数
        """
        device = x.device
        norm_mins = self.norm_mins.to(device)
        norm_maxs = self.norm_maxs.to(device)
        
        normalized_cols = []
        for i, method in enumerate(self.norm_methods): # ✅ 修复了 self.self.norm_methods 的笔误
            val = x[:, i].float()
            lo = norm_mins[i]
            hi = norm_maxs[i]

            if 'log10' in method:
                val = torch.log10(val.clamp(min=1e-8))
            elif 'asinh' in method:
                val = torch.asinh(val)

            val = (val - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
            val = val.clamp(-1.5, 1.5)
            normalized_cols.append(val)

        return torch.stack(normalized_cols, dim=1)

    def forward(self, feature_vectors, attention_mask=None, position_ids=None, output_hidden_states=False):
        batch_size = feature_vectors.shape[0]
        expected_dim = self.mlp[0].in_features  # 22
        
        if feature_vectors.shape[1] != expected_dim:
            feature_vectors = feature_vectors[:, :expected_dim]

        mlp_dtype = self.mlp[0].weight.dtype
        feature_vectors = feature_vectors.to(device=self.mlp[0].weight.device, dtype=torch.float32)

        # 1. 物理归一化 (保留真实的物理尺度)
        normed_vectors = self.normalize(feature_vectors)
        x = normed_vectors.to(dtype=mlp_dtype)

        # 2. 升维映射
        flat_out = self.mlp(x)
        
        # 3. 动态序列重塑：[B, 22, 768]，没有多余的 61 个空位！
        sequence_embeddings = flat_out.view(batch_size, self.num_tokens, self.output_dim)
        
        # 4. 注入位置信息
        feature_embeddings = sequence_embeddings + self.position_embedding.to(dtype=mlp_dtype)

        # ── 🚀 特洛伊木马：训练模式下自动计算并挂载重建误差 ──
        if self.training:
            # 使用 flat_out 尝试还原最开始的归一化特征
            recon_x = self.physics_decoder(flat_out)
            # 还原目标是归一化后的数据，保证数值稳定，detach 防止梯度回传破坏归一化
            target_normed = normed_vectors.to(dtype=mlp_dtype).detach()
            # 计算 MSE Loss 并强行挂载到 self 上，等待 train_ac_single.py 拦截提取
            self.recon_loss = torch.nn.functional.mse_loss(recon_x, target_normed)
        else:
            self.recon_loss = None

        # ── 正常推理分支 ──
        pooled_output = feature_embeddings.mean(dim=1)

        class DictTuple(tuple):
            def __getattr__(self, key):
                if key == 'hidden_states':
                    return (feature_embeddings, feature_embeddings, feature_embeddings)
                if key == 'last_hidden_state':
                    return feature_embeddings
                if key == 'pooler_output':
                    return pooled_output
                raise AttributeError(f"'DictTuple' object has no attribute '{key}'")

            def __getitem__(self, key):
                if isinstance(key, str):
                    return getattr(self, key)
                return super().__getitem__(key)

        return DictTuple((feature_embeddings, pooled_output))

    @property
    def dtype(self):
        return self.mlp[0].weight.dtype

    @property
    def device(self):
        return self.mlp[0].weight.device

    @property
    def text_model(self):
        class TextModel:
            @property
            def final_layer_norm(self):
                return nn.Identity()
        return TextModel()

    def get_input_embeddings(self):
        return nn.Embedding(self.num_embeddings, self.embedding_dim)

    def set_input_embeddings(self, value):
        pass