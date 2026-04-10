#!/usr/bin/env python3
"""
从训练日志中提取 Loss 数据并绘制曲线图
"""

import re
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

def parse_log_file(log_file):
    """解析训练日志文件，提取 Loss 数据"""
    steps = []
    losses = []
    loss_recons = []
    lr_models = []
    
    # 匹配日志格式，例如：
    # Step = [100/10000], Epoch = [0/90]<100/1683>, LR_model = 1.00e-04, Loss_recon = 0.01234, Loss = 0.56789
    pattern = re.compile(
        r'Step\s*=\s*\[(\d+)/.*?\].*?'
        r'LR_model\s*=\s*([\d.eE+-]+).*?'
        r'Loss_recon\s*=\s*([\d.]+).*?'
        r'Loss\s*=\s*([\d.]+)'
    )
    
    print(f"[*] 正在解析日志文件：{log_file}")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                lr_models.append(float(match.group(2)))
                loss_recons.append(float(match.group(3)))
                losses.append(float(match.group(4)))
    
    print(f"[√] 成功提取 {len(steps)} 个数据点")
    
    return {
        'steps': steps,
        'losses': losses,
        'loss_recons': loss_recons,
        'lr_models': lr_models
    }


def plot_loss_curves(data, output_dir, exp_name):
    """绘制 Loss 曲线图"""
    if not data['steps']:
        print("[!] 没有数据可绘制")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    
    # 子图 1: 总 Loss
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(data['steps'], data['losses'], 'b-', linewidth=2, label='Total Loss')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 子图 2: 物理重建 Loss
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(data['steps'], data['loss_recons'], 'r-', linewidth=2, label='Reconstruction Loss')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Physics Reconstruction Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 子图 3: 学习率
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(data['steps'], data['lr_models'], 'g-', linewidth=2, label='Learning Rate')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('LR', fontsize=12)
    plt.title('Model Learning Rate', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 子图 4: Loss 对比（双 Y 轴）
    ax4 = plt.subplot(2, 2, 4)
    line1 = ax4.plot(data['steps'], data['losses'], 'b-', linewidth=2, label='Total Loss')
    ax4.set_xlabel('Step', fontsize=12)
    ax4.set_ylabel('Total Loss', color='b', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='b')
    ax4.grid(True, alpha=0.3)
    
    ax5 = ax4.twinx()
    line2 = ax5.plot(data['steps'], data['loss_recons'], 'r-', linewidth=2, label='Recon Loss')
    ax5.set_ylabel('Reconstruction Loss', color='r', fontsize=12)
    ax5.tick_params(axis='y', labelcolor='r')
    ax5.grid(False)
    
    plt.title('Loss Comparison', fontsize=14, fontweight='bold')
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{exp_name}_loss_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[√] Loss 曲线图已保存至：{output_path}")
    
    # 保存数据到 CSV
    csv_path = os.path.join(output_dir, f'{exp_name}_loss_data.csv')
    df = pd.DataFrame({
        'step': data['steps'],
        'loss': data['losses'],
        'loss_recon': data['loss_recons'],
        'lr_model': data['lr_models']
    })
    df.to_csv(csv_path, index=False)
    print(f"[√] Loss 数据已保存至：{csv_path}")
    
    plt.close()


def find_latest_experiment(exps_dir):
    """找到最新的实验目录"""
    exp_dirs = [d for d in os.listdir(exps_dir) 
                if os.path.isdir(os.path.join(exps_dir, d)) and d.startswith('2026-')]
    
    if not exp_dirs:
        return None
    
    exp_dirs.sort(reverse=True)
    return exp_dirs[0]


def main():
    # 配置路径
    exps_dir = "/nfsdata/share/ljc/GalaxySD/exps"
    output_dir = "/nfsdata/share/ljc/GalaxySD/loss_plots"
    
    # 找到最新的实验
    latest_exp = find_latest_experiment(exps_dir)
    
    if latest_exp:
        print(f"[*] 找到最新实验：{latest_exp}")
        log_file = os.path.join(exps_dir, latest_exp, "train.log")
        
        if os.path.exists(log_file):
            data = parse_log_file(log_file)
            plot_loss_curves(data, output_dir, latest_exp)
        else:
            print(f"[!] 日志文件不存在：{log_file}")
    else:
        print("[!] 未找到实验目录")
        
        # 尝试解析所有实验
        for exp_name in os.listdir(exps_dir):
            exp_path = os.path.join(exps_dir, exp_name)
            if os.path.isdir(exp_path):
                log_file = os.path.join(exp_path, "train.log")
                if os.path.exists(log_file):
                    print(f"\n[*] 处理实验：{exp_name}")
                    data = parse_log_file(log_file)
                    plot_loss_curves(data, output_dir, exp_name)


if __name__ == "__main__":
    main()
