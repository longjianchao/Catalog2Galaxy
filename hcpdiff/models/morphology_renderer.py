"""
形态学先验图渲染器 - 最终物理一致性版
包含高精度 Sérsic b_n 修正与混合精度 (AMP) 兼容防御
"""

import torch
import torch.nn as nn

def render_morphology_prior_torch(
    geom_params: torch.Tensor,
    latent_size: int = 24,
    eps: float = 1e-8
) -> torch.Tensor:
    device = geom_params.device
    batch_size = geom_params.shape[0]
    
    # 1. 提取物理参数
    R_norm = geom_params[:, 0:1]      # [B, 1] - 半径
    E1 = geom_params[:, 1:2]          # [B, 1] - 椭率分量 1
    E2 = geom_params[:, 2:3]          # [B, 1] - 椭率分量 2
    n_norm = geom_params[:, 3:4]      # [B, 1] - Sérsic 指数
    
    # =========================================================
    # ⚠️ 架构师提醒：此处必须执行反归一化 (Denormalization)！
    # 将网络输入的归一化值，映射为物理世界真实的尺度。
    # 请根据你预处理时用的 MinMaxScaler/StandardScaler 填写下方的公式：
    # =========================================================
    # R = R_norm * (R_max - R_min) + R_min  
    # n = n_norm * (n_max - n_min) + n_min  
    
    # 临时占位（假设已经反归一化或在合理区间）：
    R = R_norm.clamp(min=0.05, max=1.0) # 假设 R 代表占整个 24x24 视场的比例
    n = n_norm.clamp(min=0.5, max=6.0)  
    
    # 2. E1, E2 -> θ, q 转换
    e_sq = E1 * E1 + E2 * E2
    e = torch.sqrt(e_sq + eps).clamp(max=0.95) # 限制极端扁平
    
    theta = 0.5 * torch.atan2(E2, E1 + eps)
    q = torch.sqrt(torch.clamp((1.0 - e) / (1.0 + e + eps), min=eps, max=1.0))
    
    # 3. 生成归一化 2D 坐标网格 [-1, 1]
    y = torch.linspace(-1, 1, latent_size, device=device, dtype=geom_params.dtype)
    x = torch.linspace(-1, 1, latent_size, device=device, dtype=geom_params.dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    xx = xx.unsqueeze(0).expand(batch_size, -1, -1)
    yy = yy.unsqueeze(0).expand(batch_size, -1, -1)
    
    # 4. 坐标旋转
    cos_theta = torch.cos(theta).view(batch_size, 1, 1)
    sin_theta = torch.sin(theta).view(batch_size, 1, 1)
    
    x_rot = xx * cos_theta - yy * sin_theta
    y_rot = xx * sin_theta + yy * cos_theta
    
    # 5. 椭圆距离计算与尺度归一化
    q_inv = (1.0 / (q + eps)).view(batch_size, 1, 1)
    r_ellip_sq = x_rot * x_rot + (y_rot * q_inv) * (y_rot * q_inv)
    r_ellip = torch.sqrt(r_ellip_sq + eps)
    
    # 计算 r/R_e 比例
    R_scaled = R.view(batch_size, 1, 1)
    r_ratio = r_ellip / (R_scaled + eps)
    
    # 6. 高精度 Sérsic 物理轮廓 (引入 b_n)
    # 使用 MacArthur 等人 (2003) 简化近似 b_n ≈ 1.9992n - 0.3271
    b_n = (1.9992 * n - 0.3271).clamp(min=eps)
    
    exponent = (1.0 / (n + eps)).view(batch_size, 1, 1)
    b_n_scaled = b_n.view(batch_size, 1, 1)
    
    # I(r) = exp(-b_n * ((r/Re)^(1/n) - 1))
    intensity = torch.exp(-b_n_scaled * (torch.pow(r_ratio + eps, exponent) - 1.0))
    
    # 7. 强度钳制与输出
    prior_map = intensity.clamp(min=0.0, max=1.0).unsqueeze(1)
    
    return prior_map


def expand_unet_input_channel(unet: nn.Module, new_in_channels: int = 5, old_in_channels: int = 4) -> nn.Module:
    """零卷积扩展 UNet 输入通道，保护原有生成流形"""
    old_conv_in = unet.conv_in
    if old_conv_in.in_channels == new_in_channels:
        return unet # 已扩展，跳过
        
    new_conv_in = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        stride=old_conv_in.stride,
        padding=old_conv_in.padding,
        bias=True if old_conv_in.bias is not None else False
    )
    
    with torch.no_grad():
        new_conv_in.weight[:, :old_in_channels, :, :] = old_conv_in.weight
        new_conv_in.weight[:, old_in_channels:, :, :] = 0.0  # 零初始化
        if old_conv_in.bias is not None:
            new_conv_in.bias.copy_(old_conv_in.bias)
            
    unet.conv_in = new_conv_in
    return unet


class MorphologyPriorInjector(nn.Module):
    def __init__(self, latent_size: int = 24):
        super().__init__()
        self.latent_size = latent_size
    
    def forward(self, geom_params: torch.Tensor, noisy_latents: torch.Tensor) -> torch.Tensor:
        # 1. 渲染
        prior_map = render_morphology_prior_torch(geom_params, latent_size=self.latent_size)
        
        # 2. 🛡️ AMP 防御：强制对齐 dtype 和 device，防止 float32 与 float16 冲突炸炉
        prior_map = prior_map.to(dtype=noisy_latents.dtype, device=noisy_latents.device)
        
        # 3. 拼接
        augmented_latents = torch.cat([noisy_latents, prior_map], dim=1)
        return augmented_latents