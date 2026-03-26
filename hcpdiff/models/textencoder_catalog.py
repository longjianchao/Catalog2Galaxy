from typing import Tuple, Optional, List

import torch
from torch import nn
import pandas as pd


class CatalogTextEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, norm_stats_path):
        """
        初始化星表特征编码器

        Args:
            feature_dim: 星表特征向量的维度（现在是22维）
            hidden_dim: MLP隐藏层的维度
            output_dim: 输出向量的维度，应与原始文本编码器的输出维度相同
            norm_stats_path: 归一化参数CSV文件路径（由清洗脚本生成）
        """
        super().__init__()

        # ── 加载归一化参数 ─────────────────────────────────────────
        if norm_stats_path is None:
            raise ValueError("norm_stats_path 必须提供")

        norm_df = pd.read_csv(norm_stats_path)

        # 检查特征数量是否匹配
        assert len(norm_df) == feature_dim, (
            f"norm_stats 有 {len(norm_df)} 个特征，但 feature_dim={feature_dim}，请检查是否一致"
        )

        # 字符串列表不能存 buffer，单独保存
        self.norm_methods = norm_df['method'].tolist()

        # min/max 存为 buffer：不参与梯度，但随模型保存和加载
        self.register_buffer(
            'norm_mins',
            torch.tensor(norm_df['transform_min'].values, dtype=torch.float32)
        )
        self.register_buffer(
            'norm_maxs',
            torch.tensor(norm_df['transform_max'].values, dtype=torch.float32)
        )

        # ── MLP 结构（ReLU → SiLU，更适合连续特征回归）────────────
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # ── 位置编码（保持与原版一致）────────────────────────────
        self.position_embedding = nn.Parameter(torch.randn(1, 77, output_dim) * 0.02)

        self.output_dim = output_dim
        self.num_embeddings = 49408
        self.embedding_dim = output_dim

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        将原始星表值归一化到 [-1, 1]

        Args:
            x: (batch, feature_dim)，原始星表值
        Returns:
            (batch, feature_dim)，归一化后的值
        """
        x_norm = x.clone().float()

        for i, method in enumerate(self.norm_methods):
            val = x_norm[:, i]
            lo = self.norm_mins[i]
            hi = self.norm_maxs[i]

            if method == 'log10_minmax':
                # W1/W2 有合理负值，clamp 到极小正数再取 log
                val = torch.log10(val.clamp(min=1e-8))
            elif method == 'asinh_minmax':
                # asinh 可以正确处理负值（Hα、[OIII] 发射线）
                val = torch.asinh(val)
            # linear_minmax 不需要变换，直接进行 MinMax

            # MinMax 归一化到 [-1, 1]
            val = (val - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

            # 允许推理时输入略超训练范围，但防止极端外推破坏 embedding
            val = val.clamp(-1.5, 1.5)

            x_norm[:, i] = val

        return x_norm

    def forward(self, feature_vectors, attention_mask=None, position_ids=None, output_hidden_states=False):
        batch_size = feature_vectors.shape[0]

        # ── 框架会把 prompt_ids padding 到长度77，这里截取前 feature_dim 个值 ──
        expected_dim = self.mlp[0].in_features  # 22
        if feature_vectors.shape[1] != expected_dim:
            feature_vectors = feature_vectors[:, :expected_dim]

        # 确保数据类型和设备一致
        mlp_dtype = self.mlp[0].weight.dtype
        feature_vectors = feature_vectors.to(
            device=self.mlp[0].weight.device,
            dtype=torch.float32
        )

        # 归一化（内部处理）
        feature_vectors = self.normalize(feature_vectors)

        # 转为 MLP 所需 dtype
        feature_vectors = feature_vectors.to(dtype=mlp_dtype)

        # MLP
        feature_embeddings = self.mlp(feature_vectors)        # (batch, 768)
        feature_embeddings = feature_embeddings.unsqueeze(1)  # (batch, 1, 768)
        feature_embeddings = feature_embeddings.repeat(1, 77, 1)  # (batch, 77, 768)

        # 位置编码
        position_embedding = self.position_embedding.to(dtype=feature_embeddings.dtype)
        feature_embeddings = feature_embeddings + position_embedding

        # 池化
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