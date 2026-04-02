#!/usr/bin/env python3
"""
推理脚本：从星表数据生成星系图像
归一化在 CatalogTextEncoder 内部完成，推理时直接传入原始星表值。
"""

import os
import csv
import numpy as np
import torch
from PIL import Image
from hcpdiff.visualizer import Visualizer
from omegaconf import OmegaConf


class CustomVisualizer(Visualizer):
    """自定义可视化器：拦截并修复 HCP-Diffusion 底层的 text_encoder 传参 Bug"""

    def load_model(self, pretrained_model):
        custom_te = None

        if 'new_components' in self.cfgs and 'text_encoder' in self.cfgs.new_components:
            custom_te = self.cfgs.new_components.text_encoder
            OmegaConf.set_struct(self.cfgs, False)
            self.cfgs.new_components.pop('text_encoder')

        pipe = super().load_model(pretrained_model)

        if custom_te is not None:
            pipe.text_encoder = custom_te
            self.cfgs.new_components.text_encoder = custom_te

        return pipe


class CatalogInferencer:
    def __init__(self, cfg_path, catalog_file, target_indexs):
        """
        初始化星表推理器

        Args:
            cfg_path: 配置文件路径
            catalog_file: 星表CSV文件路径
            target_indexs: 需要生成的 index 列表，只加载这些条目
        """
        self.cfg = OmegaConf.load(cfg_path)
        self._fill_default_cfg()

        # 特征列定义（必须与训练时、normalization_stats_v2.csv 行顺序完全一致）
        self.feature_columns = [
            'DESIDR1_FLUX_G', 'DESIDR1_FLUX_R', 'DESIDR1_FLUX_Z',
            'DESIDR1_FLUX_W1', 'DESIDR1_FLUX_W2',
            'DESIDR1_SHAPE_R', 'DESIDR1_SHAPE_E1', 'DESIDR1_SHAPE_E2', 'DESIDR1_SERSIC',
            'mapped_smooth_combined_score',
            'mapped_barred_spirals_combined_score',
            'mapped_unbarred_spirals_combined_score',
            'mapped_edge_on_with_bulge_combined_score',
            'mapped_mergers_combined_score',
            'mapped_irregular_combined_score',
            'DESIDR1_Z',
            'MassEMLine_MASS_CG', 'MassEMLine_SFR_CG', 'MassEMLine_AGE_CG',
            'MassEMLine_HALPHA_FLUX', 'MassEMLine_OIII5007_FLUX',
            'MassEMLine_EBV',
        ]

        # 只加载目标 index 的数据，找齐后立即停止
        target_set = {str(i) for i in target_indexs}
        print(f"从星表中提取 {len(target_set)} 条目标记录...")
        self.catalog_dict = self._load_catalog_by_indexs(catalog_file, target_set)
        print(f"成功提取 {len(self.catalog_dict)} 条记录")

        # 检查是否有找不到的 index
        missing = target_set - set(self.catalog_dict.keys())
        if missing:
            print(f"警告：以下 index 在星表中不存在: {missing}")

        # 初始化可视化器
        self.visualizer = CustomVisualizer(self.cfg)

    def _load_catalog_by_indexs(self, catalog_file, target_set):
        """
        只读取目标 index 的行，找完所有目标后立即停止，不加载整个星表。
        """
        catalog_dict = {}
        with open(catalog_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row.get('index', '')
                if idx in target_set:
                    catalog_dict[idx] = row
                    if len(catalog_dict) == len(target_set):
                        break
        return catalog_dict

    def _fill_default_cfg(self):
        """补充缺省配置项"""
        defaults = {
            'dtype': 'fp32',
            'amp': True,
            'condition': None,
            'emb_dir': 'embs/',
            'clip_skip': 0,
            'clip_final_norm': True,
            'encoder_attention_mask': True,
            'seed': None,
            'offload': None,
            'vae_optimize': {'tiling': False, 'slicing': False},
            'save': {'out_dir': 'output/', 'save_cfg': True,
                     'image_type': 'png', 'quality': 95},
            'ex_input': {},
        }
        for key, val in defaults.items():
            if key not in self.cfg:
                self.cfg[key] = val

    def extract_raw_features(self, row):
        """
        从一行星表记录中提取原始特征值。
        不做任何归一化或变换，归一化由 CatalogTextEncoder.normalize() 内部处理。
        """
        feature_vector = []
        missing_cols = []

        for col in self.feature_columns:
            val_str = row.get(col, '')
            if val_str in ('', 'NA_Value', '-99.0', 'NaN', 'nan', 'None'):
                val = 0.0
                missing_cols.append(col)
            else:
                try:
                    val = float(val_str)
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                        missing_cols.append(col)
                except (ValueError, TypeError):
                    val = 0.0
                    missing_cols.append(col)
            feature_vector.append(val)

        if missing_cols:
            print(f"  警告：以下列缺失，已用0填充: {missing_cols}")

        return np.array(feature_vector, dtype=np.float32)

    def infer_from_indexs(self, indexs, output_dir):
        """按 index 列表生成图像，自动居中处理"""
        os.makedirs(output_dir, exist_ok=True)

        from hcpdiff.models.textencoder_catalog import CatalogTextEncoder

        if not isinstance(self.visualizer.pipe.text_encoder, CatalogTextEncoder):
            raise RuntimeError(
                "text_encoder 不是 CatalogTextEncoder，请检查推理 yaml 配置。"
            )

        device = self.visualizer.pipe.device
        infer_args = OmegaConf.to_container(self.cfg.infer_args, resolve=True)

        for index in indexs:
            index_str = str(index)
            print(f"\n{'='*50}")
            print(f"正在生成 index: {index_str}")

            if index_str not in self.catalog_dict:
                print(f"  跳过（星表中不存在）")
                continue

            row = self.catalog_dict[index_str]
            raw_features = self.extract_raw_features(row)

            try:
                # 转为 tensor，shape: (1, 22)
                feature_tensor = torch.tensor(
                    raw_features, dtype=torch.float32
                ).unsqueeze(0).to(device)

                # 前向传播（内部自动归一化）
                with torch.no_grad():
                    emb, _ = self.visualizer.pipe.text_encoder(feature_tensor)

                # 无条件 embedding（全零）
                negative_emb = torch.zeros_like(emb)

                # 生成图像
                images = self.visualizer.pipe(
                    prompt_embeds=emb,
                    negative_prompt_embeds=negative_emb,
                    **infer_args
                ).images

                save_path = os.path.join(output_dir, f"{index_str}.png")
                images[0].save(save_path)
                print(f"  已保存至: {save_path}")

            except Exception as e:
                print(f"  生成失败: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Galaxy Catalog Inference")
    parser.add_argument("--cfg",     type=str, required=True,
                        help="推理配置 yaml 路径")
    parser.add_argument("--catalog", type=str, required=True,
                        help="星表 CSV 路径")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="图像输出目录")
    parser.add_argument("--indexs",  type=str, nargs='+', required=True,
                        help="要生成的 index 列表")
    args = parser.parse_args()

    inferencer = CatalogInferencer(args.cfg, args.catalog, args.indexs)
    inferencer.infer_from_indexs(args.indexs, args.out_dir)
    print("\n全部推理任务完成！")
