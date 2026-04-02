#!/bin/bash
# ... (前面的 export 保持不变) ...

export CUDA_VISIBLE_DEVICES=0,2
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 🌟 新加这一行，解决 ModuleNotFoundError
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "🚀 启动 22 维物理特征全量星系双卡大炼丹 (v4)..."

accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --mixed_precision="bf16" \
    --main_process_port 29501 \
    hcpdiff/train_ac_single.py \
    --cf cfgs/train/examples/fine-tuning_catalog_new.yaml