#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com

name="2026-04-02"  # 替换为你的真实实验名称


# ⚠️ 必须传入你【训练时使用的完整大星表】，以保证算出来的均值/方差和训练时绝对一致！
catalog_file="/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/labeled_catalog_cleaned_v3.csv"

# 配置文件 (我们需要确保这个 yaml 里加载了你训练好的 MLP 权重)
cfg_file="cfgs/infer/text2img_galaxy_catalog.yaml"

stats_file="/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/normalization_stats_v3.csv"

output_dir="output/${name}_catalog_infer_dps"

indexs="284731 288533 32549"
# indexs="256830 155321 273391"
# indexs="4 11 19 22"
# indexs="59771 329218 50184 106016"
# indexs="105998 105999 284900 105986 105987 329340 105967 106312 105936 105993"

regressor_ckpt="regressor_output/best_regressor_v3.pth"

lambda_guide=10.0

skip_ratio=0.2

command="python3 infer_catalog.py \
    --cfg ${cfg_file} \
    --catalog ${catalog_file} \
    --out_dir ${output_dir} \
    --indexs ${indexs} \
    --regressor_ckpt ${regressor_ckpt} \
    --stats_file ${stats_file} \
    --lambda_guide ${lambda_guide} \
    --skip_ratio ${skip_ratio}"

echo "Executing: $command"
eval "$command"