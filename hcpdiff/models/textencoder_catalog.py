from typing import Tuple, Optional, List

import torch
from torch import nn
import pandas as pd


class CatalogTextEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, norm_stats_path, num_tokens=16):
        """
        初始化多 Token 星表特征编码器 (v4 架构)

        Args:
            feature_dim: 星表特征向量的维度（现在是22维）
            hidden_dim: MLP隐藏层的维度
            output_dim: 输出向量的维度，应与原始文本编码器的输出维度相同 (768)
            norm_stats_path: 归一化参数CSV文件路径
            num_tokens: 映射成的 Token 数量。将 22 维物理参数非线性混合后，
                        解耦为 16 个独立的特征 Token，打破注意力对称性。默认值为16。
        """

        super().__init__()
        self.num_tokens = num_tokens

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

        # ── MLP 结构升级（输出扩展为 num_tokens * output_dim）────────────
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            # 输出维度放大，为后续拆分成多个 Token 做准备
            nn.Linear(hidden_dim, num_tokens * output_dim)
        )

        # ── 位置编码与序列填充 ──────────────────────────────────────
        # 1. 只有前 num_tokens 个物理特征 Token 拥有专属的位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, num_tokens, output_dim) * 0.02)
        
        # 2. 剩下的空位（77 - num_tokens），使用一个统一的可学习 Padding Token 补齐
        self.padding_token = nn.Parameter(torch.zeros(1, 77 - num_tokens, output_dim))

        self.output_dim = output_dim
        self.num_embeddings = 49408
        self.embedding_dim = output_dim

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        安全的归一化函数：彻底废除了 In-place 原地修改，防止 DDP 和 AMP 梯度报错。
        将原始星表值归一化到 [-1, 1]
        """
        device = x.device
        
        # [✅ 真正的 BUG 1 修复]：在循环外，把极值 Tensor 整体推到目标显存上
        # 直接使用 Tensor 索引操作，避免 GPU 同步开销并保持精度
        norm_mins = self.norm_mins.to(device)
        norm_maxs = self.norm_maxs.to(device)
        
        normalized_cols = []
        
        for i, method in enumerate(self.norm_methods):
            val = x[:, i].float()
            
            # [✅ 真正的 BUG 1 修复]：直接使用 Tensor 索引，避免 GPU 同步锁定并保持精度
            lo = norm_mins[i]
            hi = norm_maxs[i]

            if 'log10' in method:
                val = torch.log10(val.clamp(min=1e-8))
            elif 'asinh' in method:
                val = torch.asinh(val)

            # MinMax 归一化到 [-1, 1]
            val = (val - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

            # 防止极端外推破坏 embedding
            val = val.clamp(-1.5, 1.5)
            normalized_cols.append(val)

        # 沿着特征维度重新拼接，避免 inplace 修改带来的计算图断裂
        return torch.stack(normalized_cols, dim=1)

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

        # 归一化（内部安全处理）
        feature_vectors = self.normalize(feature_vectors)

        # 转为 MLP 所需 dtype
        x = feature_vectors.to(dtype=mlp_dtype)

        # ── 核心张量变换：单向量 -> 多 Token 序列 ────────────────────────
        # 1. 过 MLP，得到扁平输出 (batch_size, 16 * 768)
        flat_out = self.mlp(x)
        
        # 2. 重塑为 Token 序列格式 (batch_size, 16, 768)
        sequence_embeddings = flat_out.view(batch_size, self.num_tokens, self.output_dim)
        
        # 3. 注入位置编码 (让 UNet 区分这 16 个 Token 的不同职能)
        sequence_embeddings = sequence_embeddings + self.position_embedding.to(dtype=mlp_dtype)
        
        # 4. 扩展 Padding Token 补齐到 77 长度，骗过 UNet 的尺寸检查
        # [✅ 修复 Bug 3]：消除 61 硬编码，使用 77 - self.num_tokens 动态描述形状
        # padding shape: (batch_size, 77 - self.num_tokens, 768)
        padding = self.padding_token.expand(batch_size, -1, -1).to(dtype=mlp_dtype)
        
        # 5. 拼接成最终的输入序列 (batch_size, 77, 768)
        feature_embeddings = torch.cat([sequence_embeddings, padding], dim=1)

        # ── 池化 (仅对包含实际物理意义的前 num_tokens 取平均) ─────────────
        pooled_output = feature_embeddings[:, :self.num_tokens, :].mean(dim=1)

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