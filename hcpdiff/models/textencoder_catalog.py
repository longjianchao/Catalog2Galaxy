from typing import Tuple, Optional, List

import torch
from torch import nn
import pandas as pd

class CatalogTextEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, norm_stats_path, num_tokens=22):
        """
        初始化纯物理驱动的星表特征编码器
        双通道条件注入架构：
          - 通道A：cross-attention（原有MLP → [B, 22, 768] token序列）
          - 通道B：AdaGN（新增projector → [B, 1280] 全局向量，注入UNet time_embedding）

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

        # ── 1. 物理归一化参数加载 ──────────────────────────────────
        if norm_stats_path is None:
            raise ValueError("norm_stats_path 必须提供")

        norm_df = pd.read_csv(norm_stats_path)
        assert len(norm_df) == feature_dim, (
            f"norm_stats 有 {len(norm_df)} 个特征，但 feature_dim={feature_dim}"
        )

        self.norm_methods = norm_df['method'].tolist()
        self.register_buffer('norm_mins', torch.tensor(norm_df['transform_min'].values, dtype=torch.float32))
        self.register_buffer('norm_maxs', torch.tensor(norm_df['transform_max'].values, dtype=torch.float32))

        # ── 2. 通道A：cross-attention MLP（原有，保持不变）──────────
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_tokens * output_dim)
        )

        # ── 3. 物理位置编码（原有，保持不变）────────────────────────
        self.position_embedding = nn.Parameter(torch.randn(1, num_tokens, output_dim) * 0.02)

        # ── 4. 物理信息重建解码器（原有，保持不变）──────────────────
        self.physics_decoder = nn.Sequential(
            nn.Linear(num_tokens * output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        # ── 5. 通道B：AdaGN projector（新增）────────────────────────
        # 输出1280维，对齐SD1.5 UNet的time_embedding维度
        # 最后一层初始化为零：训练初期不扰乱UNet已有的生成能力
        # 随着训练推进，物理信号会逐渐从零开始增强
        self.adagn_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1280)
        )
        nn.init.zeros_(self.adagn_projector[-1].weight)
        nn.init.zeros_(self.adagn_projector[-1].bias)

        # 用于在forward之外传递AdaGN向量给TEUnetWrapper
        self.last_adagn_emb = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        极其安全的非 In-place 归一化函数
        """
        device = x.device
        norm_mins = self.norm_mins.to(device)
        norm_maxs = self.norm_maxs.to(device)

        normalized_cols = []
        for i, method in enumerate(self.norm_methods):
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

        # 在输入空间检测是否为全零（无条件信号）
        is_uncond = (feature_vectors.abs().sum(dim=1) == 0)  # [B]

        # 1. 物理归一化
        normed_vectors = self.normalize(feature_vectors)
        x = normed_vectors.to(dtype=mlp_dtype)

        # ── 通道A：cross-attention路径（原有逻辑不变）────────────────
        flat_out = self.mlp(x)
        sequence_embeddings = flat_out.view(batch_size, self.num_tokens, self.output_dim)
        feature_embeddings = sequence_embeddings + self.position_embedding.to(dtype=mlp_dtype)

        # ── 通道B：AdaGN路径（新增）──────────────────────────────────
        adagn_emb = self.adagn_projector(x) * 20.0  # [B, 1280]

        # 无条件样本：两个通道都置零
        if is_uncond.any():
            uncond_mask_3d = is_uncond.view(batch_size, 1, 1).to(dtype=mlp_dtype)
            uncond_mask_2d = is_uncond.view(batch_size, 1).to(dtype=mlp_dtype)
            feature_embeddings = feature_embeddings * (1.0 - uncond_mask_3d)
            adagn_emb = adagn_emb * (1.0 - uncond_mask_2d)

        # 把AdaGN向量挂载到self上，供TEUnetWrapper在UNet forward前取用
        self.last_adagn_emb = adagn_emb

        # ── 重建loss（原有逻辑不变）──────────────────────────────────
        if torch.is_grad_enabled():
            if (~is_uncond).any():
                valid_flat = flat_out[~is_uncond]
                valid_normed = normed_vectors[~is_uncond].to(dtype=mlp_dtype).detach()
                recon_x = self.physics_decoder(valid_flat)
                self.recon_loss = torch.nn.functional.mse_loss(recon_x, valid_normed)
            else:
                self.recon_loss = None
        else:
            self.recon_loss = None

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