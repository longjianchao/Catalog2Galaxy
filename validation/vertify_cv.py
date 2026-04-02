import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

# ================= 配置区 =================
# 填写你刚才生成的纯净假图目录和对应的真实图目录
FAKE_DIR = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/validation/val_results_150k/fake_only"
REAL_DIR = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/filtered_data"
# ==========================================

def load_image_as_tensor(img_path):
    """将图像读取并转换为 0-255 的 uint8 Tensor (B, C, H, W)，这是 torchmetrics 的硬性要求"""
    img = Image.open(img_path).convert("RGB").resize((192, 192))
    # ToTensor 转换后是 0-1 的 float，乘以 255 转为 uint8
    tensor = (ToTensor()(img) * 255).to(torch.uint8)
    return tensor.unsqueeze(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 正在使用设备: {device}")

    # 初始化评估器 (feature=2048 是标准 Inception-v3 设定)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=50, normalize=False).to(device)

    fake_images = [f for f in os.listdir(FAKE_DIR) if f.endswith(".png") or f.endswith(".jpg")]
    
    print(f"[*] 找到 {len(fake_images)} 张生成图像，开始提取特征...")

    # 提取真实图像和生成图像的特征
    for fake_name in tqdm(fake_images, desc="Processing Images"):
        fake_path = os.path.join(FAKE_DIR, fake_name)
        
        # 还原真实图像的文件名 (假设格式为 {index}_fake.png -> {index}.jpg)
        real_name = fake_name.replace("_fake.png", ".jpg")
        real_path = os.path.join(REAL_DIR, real_name)
        
        if not os.path.exists(real_path):
            real_name = fake_name.replace("_fake.png", ".png")
            real_path = os.path.join(REAL_DIR, real_name)

        if os.path.exists(real_path):
            real_tensor = load_image_as_tensor(real_path).to(device)
            fake_tensor = load_image_as_tensor(fake_path).to(device)

            # 更新特征统计 (real=True 代表真实分布)
            fid.update(real_tensor, real=True)
            fid.update(fake_tensor, real=False)
            
            kid.update(real_tensor, real=True)
            kid.update(fake_tensor, real=False)

    print("\n[*] 特征提取完毕，正在计算最终得分（这可能需要几十秒）...")
    
    fid_score = fid.compute()
    kid_mean, kid_std = kid.compute()

    print(f"\n{'='*40}")
    print(f"🎯 FID Score: {fid_score.item():.4f}")
    print(f"🎯 KID Score: {kid_mean.item():.5f} ± {kid_std.item():.5f}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()