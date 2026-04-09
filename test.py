import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from batch_infer_2k import CatalogInferencerDPS, CFG_PATH, CATALOG_FILE, INDEX_FILE, REAL_IMG_ROOT, STATS_FILE, REGRESSOR_CKPT

def run_attention_probe(pipe, raw_feature_vector, save_path="attention_probe_result.png"):
    """
    注意力探针测试：无损挂载 Hook 截获 Cross-Attention 权重，验证 77 Token 的物理能量分布
    
    参数:
        pipe: 已经加载并初始化权重的 diffusers Pipeline
        raw_feature_vector: 一条 22 维的原始未归一化星表数据 (numpy array 或 tensor)
        save_path: 柱状图的保存路径
    """
    print("[*] 正在部署交叉注意力探针 (Cross-Attention Probe)...")
    device = pipe.device
    dtype = pipe.dtype

    # 1. 初始化拦截缓存
    activation_cache = {}

    def get_q_hook():
        def hook(module, input, output):
            activation_cache["q"] = output.detach()
        return hook

    def get_k_hook():
        def hook(module, input, output):
            activation_cache["k"] = output.detach()
        return hook

    # 2. 定位目标 Cross-Attention 层
    # 这里选择 down_blocks[1] 是因为中层特征感受野最适合宏观天体物理结构的条件注入
    try:
        target_attn_layer = pipe.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn2
    except AttributeError:
        print("[!] 无法找到目标注意力层 attn2，请根据你的 UNet 层级路径进行修正。")
        return

    # 注册钩子 (Hook)
    handle_q = target_attn_layer.to_q.register_forward_hook(get_q_hook())
    handle_k = target_attn_layer.to_k.register_forward_hook(get_k_hook())

    print("[*] 探针挂载完毕，正在构造物理流形测试张量...")

    # 3. 构造并格式化测试输入
    if isinstance(raw_feature_vector, np.ndarray):
        raw_feature_tensor = torch.tensor(raw_feature_vector, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        raw_feature_tensor = raw_feature_vector.clone().detach().to(device=device, dtype=torch.float32)
        if raw_feature_tensor.dim() == 1:
            raw_feature_tensor = raw_feature_tensor.unsqueeze(0)

    with torch.no_grad():
        # TextEncoder 前向传播，输出 [Batch, 77, 768] 的混合 Embedding
        emb, _ = pipe.text_encoder(raw_feature_tensor)
        
        # 构造适配 192x192 图像的假潜变量 (降采样 8 倍后尺寸为 24x24)
        latent_size = 192 // 8
        in_channels = pipe.unet.config.in_channels
        dummy_latents = torch.randn(1, in_channels, latent_size, latent_size, device=device, dtype=dtype)
        
        # 取扩散过程的中期时间步，此时注意力权重最能反映条件信号的宏观引导
        dummy_t = torch.tensor([500], device=device, dtype=torch.long)

        print("[*] 正在穿透 UNet 触发前向传播，截取信号特征...")
        _ = pipe.unet(dummy_latents, dummy_t, encoder_hidden_states=emb)

    # 4. 拆除钩子，释放计算图
    handle_q.remove()
    handle_k.remove()

    if "q" not in activation_cache or "k" not in activation_cache:
        print("[!] 致命错误：未能成功截获 Q 或 K 张量，请检查前向传播是否顺利完成。")
        return

    # 5. 核心物理计算：还原 Scaled Dot-Product Attention
    q_proj = activation_cache["q"]  # 形状: [Batch, Sequence_Image, Inner_Dim]
    k_proj = activation_cache["k"]  # 形状: [Batch, Sequence_Text(77), Inner_Dim]

    batch_size = q_proj.shape[0]
    heads = target_attn_layer.heads
    head_dim = q_proj.shape[-1] // heads

    # 严密的维度重塑机制: [B, L, heads * head_dim] -> [B, L, heads, head_dim] -> [B, heads, L, head_dim]
    q = q_proj.view(batch_size, -1, heads, head_dim).transpose(1, 2)
    k = k_proj.view(batch_size, -1, heads, head_dim).transpose(1, 2)

    # 矩阵乘法计算注意力得分 (Q * K^T / sqrt(d))
    # transpose(-1, -2) 用于翻转 K 的特征维度和序列维度
    attention_scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)

    # 计算 Softmax 概率分布 (沿着文本 Token 的 77 个维度，即 dim=-1)
    attention_probs = torch.softmax(attention_scores, dim=-1)

    # 6. 统计物理能量聚集度
    # attention_probs 形状: [B, heads, seq_image, 77]
    # 我们对批次、多头、所有图像空间像素位置取平均，提取每个 Token 的全局平均注意力权重
    mean_attention = attention_probs.mean(dim=(0, 1, 2)).cpu().to(torch.float32).numpy()

    # 7. 学术级数据可视化
    print("[*] 数据截获完毕，正在渲染注意力分布能谱...")
    plt.figure(figsize=(14, 7))
    
    # 🚀 修改点 1：横坐标变成 22，全部标记为物理参数（红色）
    num_tokens = mean_attention.shape[0]  # 现在这里是 22
    colors = ['#FF4B4B'] * num_tokens 
    
    plt.bar(range(num_tokens), mean_attention, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # 🚀 修改点 2：计算总和
    physics_mean = np.sum(mean_attention)
    
    plt.title(f"Cross-Attention Weight Distribution Probe (22-Token Pure Physics)\n(Total Attention Sum: {physics_mean:.2f})", 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Physical Feature Token Index (0-21)", fontsize=14, labelpad=10)
    plt.ylabel("Mean Attention Probability", fontsize=14, labelpad=10)
    
    # 构建高保真图例
    legend_elements = [
        Patch(facecolor='#FF4B4B', edgecolor='black', label='Pure Physics Parameter Tokens (0-21)')
    ]
    plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"[√] 注意力探针测试完成！直观结果图表已保存至: {os.path.abspath(save_path)}")
    print(f"    -> 物理条件实际注入率: {physics_mean:.2%}")



if __name__ == "__main__":
    print("[*] 正在初始化 Inferencer...")
    # 1. 现场实例化
    inferencer = CatalogInferencerDPS(CFG_PATH, CATALOG_FILE, INDEX_FILE, REAL_IMG_ROOT, STATS_FILE, REGRESSOR_CKPT)
    
    # 🚀 [核心修复] 将处于 CPU 内存中的 Pipeline 强制推入 GPU 显存！
    print("[*] 正在将模型加载至 GPU...")
    inferencer.visualizer.pipe.to("cuda")
    
    # 2. 提取一条用于测试的原始特征
    sample_idx = list(inferencer.catalog_dict.keys())[0]
    sample_raw_feature = inferencer.extract_raw_features(inferencer.catalog_dict[sample_idx])
    
    # 3. 传入管线并启动探针
    run_attention_probe(inferencer.visualizer.pipe, sample_raw_feature)