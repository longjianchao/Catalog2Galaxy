from .text2img import Text2ImageAttMapSource, default_image_transforms
import os
import csv
from typing import Dict, Any, List, Tuple
import numpy as np
import torch


class Text2ImageCatalogSource(Text2ImageAttMapSource):

    def load_template(self, template_file):
        """拦截父类的模板加载机制，直接返回空模板"""
        return ['']

    def __init__(self, img_root, catalog_file, text_transforms, image_transforms=default_image_transforms,
                 bg_color=(255, 255, 255), repeat=1, **kwargs):
        super().__init__(img_root, None, '', text_transforms=text_transforms,
                         image_transforms=image_transforms,
                         bg_color=bg_color,
                         repeat=repeat)
        self.prompt_template = ['']
        self.caption_dict = {}
        self.img_root = img_root
        self.repeat = repeat
        self.image_transforms = image_transforms
        self.text_transforms = text_transforms
        self.bg_color = tuple(bg_color)
        self.att_mask = {}

        # 特征列定义（22维，顺序必须与 normalization_stats_v2.csv 完全一致）
        self.feature_columns = [
            # 1. 测光流量（5维）
            'DESIDR1_FLUX_G', 'DESIDR1_FLUX_R', 'DESIDR1_FLUX_Z',
            'DESIDR1_FLUX_W1', 'DESIDR1_FLUX_W2',
            # 2. 几何结构（4维）
            'DESIDR1_SHAPE_R', 'DESIDR1_SHAPE_E1', 'DESIDR1_SHAPE_E2', 'DESIDR1_SERSIC',
            # 3. 形态语义评分，mapped_ 版本（6维）
            'mapped_smooth_combined_score',
            'mapped_barred_spirals_combined_score',
            'mapped_unbarred_spirals_combined_score',
            'mapped_edge_on_with_bulge_combined_score',
            'mapped_mergers_combined_score',
            'mapped_irregular_combined_score',
            # 4. 物理演化与光谱（6维）
            'DESIDR1_Z',
            'MassEMLine_MASS_CG', 'MassEMLine_SFR_CG', 'MassEMLine_AGE_CG',
            'MassEMLine_HALPHA_FLUX', 'MassEMLine_OIII5007_FLUX',
            # 5. 银河系消光（1维）
            'MassEMLine_EBV',
        ]

        # 加载星表（只建 index→row 字典，不做统计计算）
        self.catalog_data = self.load_catalog(catalog_file)
        print(f"星表加载完成，共 {len(self.catalog_data)} 条记录，特征维度: {len(self.feature_columns)}")

    def load_catalog(self, catalog_file):
        """加载星表，返回 {index: row_dict}"""
        catalog_data = {}
        with open(catalog_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row['index']
                catalog_data[idx] = row
        return catalog_data

    def extract_raw_features(self, data):
        """
        提取原始特征值，不做任何归一化或变换。
        归一化完全由 CatalogTextEncoder.normalize() 在内部处理。
        缺失值用 0.0 填充。
        """
        feature_vector = []
        for col in self.feature_columns:
            val_str = data.get(col, '')
            if val_str in ('', 'NA_Value', '-99.0', 'NaN', 'nan', 'None'):
                val = 0.0
            else:
                try:
                    val = float(val_str)
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                except (ValueError, TypeError):
                    val = 0.0
            feature_vector.append(val)

        return np.array(feature_vector, dtype=np.float32)

    def get_index_from_img_name(self, img_name):
        """从文件名（如 12345.png）中提取 index 字符串"""
        return os.path.splitext(img_name)[0]

    def load_caption(self, img_name) -> str:
        """读取图像对应的原始星表特征，序列化为逗号分隔字符串"""
        idx = self.get_index_from_img_name(img_name)

        if idx not in self.catalog_data:
            # 找不到对应条目时返回全零（训练时应尽量避免这种情况）
            feature_vector = np.zeros(len(self.feature_columns), dtype=np.float32)
            return ','.join([str(f) for f in feature_vector])

        # 直接返回原始值，不做归一化
        data = self.catalog_data[idx]
        raw_vector = self.extract_raw_features(data)
        return ','.join([str(f) for f in raw_vector])

    def process_text(self, text_dict):
        feature_str = text_dict['caption']
        feature_vector = np.array([float(f) for f in feature_str.split(',')], dtype=np.float32)

        assert len(feature_vector) == len(self.feature_columns), (
            f"特征维度不匹配：期望 {len(self.feature_columns)}，实际 {len(feature_vector)}"
        )

        # ── 🔴 核心修改：10% 概率的 Condition Dropout ──
        # 这一步极其关键！以 10% 的概率将输入特征全置为 0，让模型学习“无条件背景”。
        # 没有这一步，推理时 CFG (guidance_scale > 1.0) 必定会引发特征崩溃。
        if torch.rand(1).item() < 0.1:
            feature_vector = np.zeros_like(feature_vector)
        # ───────────────────────────────────────

        # 框架会把这个 tensor 当 token ids 处理并 padding 到77
        # CatalogTextEncoder.forward() 里会截取前 feature_dim 个值
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32)
        return {'prompt_ids': feature_tensor}

    def load_image(self, path) -> Dict[str, Any]:
        """加载图像并返回图像字典"""
        from PIL import Image
        image = Image.open(path)
        if image.mode == 'RGBA':
            x, y = image.size
            canvas = Image.new('RGBA', image.size, self.bg_color)
            canvas.paste(image, (0, 0, x, y), image)
            image = canvas
        return {'image': image.convert("RGB")}

    def get_image_list(self) -> List[Tuple[str, 'Text2ImageCatalogSource']]:
        """获取图像列表"""
        from hcpdiff.utils.utils import get_file_ext
        from hcpdiff.utils.img_size_tool import types_support
        imgs = [(os.path.join(self.img_root, x), self)
                for x in os.listdir(self.img_root)
                if get_file_ext(x) in types_support]
        return imgs * self.repeat