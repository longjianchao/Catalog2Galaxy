# 🌌 Catalog2Galaxy : Physically Conditioned Galaxy Generation

[!\[PyTorch\](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg null)](https://pytorch.org/)
[!\[Stable Diffusion\](https://img.shields.io/badge/Stable%20Diffusion-v1.5-blue.svg null)](https://github.com/CompVis/stable-diffusion)
[!\[License: MIT\](https://img.shields.io/badge/License-MIT-yellow.svg null)](https://opensource.org/licenses/MIT)

## 📖 Overview | 项目简介

This repository introduces a novel cross-modal generation framework based on Stable Diffusion. Instead of using discrete natural language prompts (CLIP), we design a custom Continuous Feature Encoder to map 22-dimensional galaxy physical and morphological parameters directly into the latent space for conditional generation.

本项目提出了一个基于 Stable Diffusion 的新型跨模态生成框架。我们抛弃了传统的离散自然语言（CLIP）输入，专门设计了一个连续特征编码器（Continuous Feature Encoder），将 22 维星系物理与形态参数直接映射到潜空间，实现精准的条件控制生成。

## ✨ Core Innovations | 核心创新点

### 🔭 Astrophysics Perspective (天体物理视角)

- **Data-Driven Morphology**: Bridging the gap between photometric/spectroscopic catalogs and visual morphology.
- **22D Parameter Conditioning**: Incorporates DESI photometric fluxes, geometric parameters (Sersic index, ellipticity), semantic morphological scores, and physical properties (Redshift $z$, Stellar Mass, SFR, Age, Emission Lines).

### 💻 Computer Science Perspective (计算机科学视角)

- **Multi-Token MLP Architecture**: Replaces the standard CLIP Text Encoder. Maps the input vector $x \in \mathbb{R}^{22}$ into a highly decoupled multi-token embedding space $E \in \mathbb{R}^{16 \times 768}$, breaking the attention symmetry and allowing the UNet's Cross-Attention mechanism to capture high-frequency details (e.g., spiral arms).
- **Robust Normalization**: Implements an operation-safe normalization strategy (Log10, Asinh, Linear MinMax) compatible with PyTorch DDP and AMP.
- **Condition Dropout**: Introduces a 10% probability of zeroing out conditions during Dataloading to enable Classifier-Free Guidance (CFG) during inference.

## 🛠️ Model Architecture | 模型架构

- **Base Model**: RunwayML Stable Diffusion v1.5 (UNet + VAE)
- **Condition Encoder**: Custom 4-layer SiLU MLP + Positional Embeddings
- **Training Strategy**: Fine-tuning with `hcpdiff` framework using Bfloat16 mixed precision and gradient checkpointing.

## 📊 Feature Columns | 物理特征列表

The model is strictly conditioned on the following 22 continuous variables:

1. **Photometry**: `FLUX_G`, `FLUX_R`, `FLUX_Z`, `FLUX_W1`, `FLUX_W2`
2. **Geometry**: `SHAPE_R`, `SHAPE_E1`, `SHAPE_E2`, `SERSIC`
3. **Morphology Scores**: Smooth, Barred/Unbarred Spirals, Edge-on, Mergers, Irregular
4. **Physical Evolution**: `Redshift (Z)`, `Mass`, `SFR`, `Age`, `H-alpha Flux`, `OIII Flux`, `EBV`

## 🚀 Quick Start | 快速开始

*(这里留白，以后补充你的* *`git clone`、环境配置和* *`infer.sh`* *运行命令)*

## 🖼️ Gallery | 生成结果展示

*(这里留白，等你 10000 步的 Checkpoint 出来后，放几张极具代表性的对比图：输入高旋臂评分参数 -> 真的画出了旋臂的图)*

## 📝 TODO List | 未来计划

- Implement multi-token MLP encoder ($16 \times 768$)
- Add condition dropout for CFG support
- Train first 150k steps baseline
- Ablation study on token sequence length

