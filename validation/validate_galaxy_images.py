#!/usr/bin/env python3
"""
星系图像验证脚本

验证生成的星系图像是否符合星表数据
"""

import os
import csv
from PIL import Image
import numpy as np
import torch
from scipy import ndimage
from skimage import measure
from skimage.metrics import structural_similarity as ssim
from astropy.nddata import NDData
from photutils import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry
from photutils.background import Background2D, MedianBackground

class GalaxyValidator:
    def __init__(self, catalog_file):
        """
        初始化验证器
        
        Args:
            catalog_file: 星表CSV文件路径
        """
        self.catalog_data = self.load_catalog(catalog_file)
        self.catalog_dict = {row['index']: row for row in self.catalog_data if 'index' in row}
        
    def load_catalog(self, catalog_file):
        """加载星表数据"""
        catalog_data = []
        with open(catalog_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                catalog_data.append(row)
        return catalog_data
    
    def extract_image_features(self, image_path, shape_r=None):
        """
        从图像中提取特征（天体物理标准）
        
        Args:
            image_path: 图像路径
            shape_r: 星表中的 DESIDR1_SHAPE_R (arcsec)，用于孔径大小计算
            
        Returns:
            dict: 提取的图像特征
        """
        # 常量定义
        PIXEL_SCALE = 0.104  # 0.104 arcsec/pix
        
        # 1. 加载图像
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # 2. 分离 RGB 通道（对应 G, R, Z 波段）
        if len(img_array.shape) == 3:
            # RGB 对应 G, R, Z 波段
            g_channel = img_array[:, :, 0].astype(float)
            r_channel = img_array[:, :, 1].astype(float)
            z_channel = img_array[:, :, 2].astype(float)
        else:
            # 灰度图
            g_channel = img_array.astype(float)
            r_channel = img_array.astype(float)
            z_channel = img_array.astype(float)
        
        features = {}
        
        # 3. 背景扣除（使用边缘区域的中位数）
        def estimate_background(channel):
            # 取图像边缘 10% 的区域
            edge_width = int(min(channel.shape) * 0.1)
            # 分别计算各个边缘的中位数，然后取平均
            top = channel[:edge_width, :]
            bottom = channel[-edge_width:, :]
            left = channel[:, :edge_width]
            right = channel[:, -edge_width:]
            
            # 计算每个边缘的中位数
            medians = []
            if top.size > 0:
                medians.append(np.median(top))
            if bottom.size > 0:
                medians.append(np.median(bottom))
            if left.size > 0:
                medians.append(np.median(left))
            if right.size > 0:
                medians.append(np.median(right))
            
            return np.mean(medians) if medians else 0
        
        bg_g = estimate_background(g_channel)
        bg_r = estimate_background(r_channel)
        bg_z = estimate_background(z_channel)
        
        # 扣除背景
        g_channel -= bg_g
        r_channel -= bg_r
        z_channel -= bg_z
        
        # 确保非负
        g_channel = np.maximum(g_channel, 0)
        r_channel = np.maximum(r_channel, 0)
        z_channel = np.maximum(z_channel, 0)
        
        # 4. 星系检测和质心计算（使用 Z 波段）
        threshold = np.percentile(z_channel, 75)
        binary = z_channel > threshold
        labeled, num_features = ndimage.label(binary)
        
        if num_features > 0:
            # 最大连通区域
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            
            # 质心位置
            centroid_y, centroid_x = ndimage.center_of_mass(binary, labeled, largest_label)
            features['centroid_y'] = centroid_y
            features['centroid_x'] = centroid_x
            
            # 5. 孔径测光
            positions = [(centroid_x, centroid_y)]
            
            # 计算孔径大小
            if shape_r is not None:
                # 从 arcsec 转换为像素
                aperture_radius = shape_r / PIXEL_SCALE
            else:
                # 默认孔径大小
                aperture_radius = 10
            
            # 创建光圈
            aperture = CircularAperture(positions, r=aperture_radius)
            
            # 对每个通道进行测光
            def photometry(channel):
                data = NDData(channel)
                phot_table = aperture_photometry(data, aperture)
                return phot_table['aperture_sum'][0]
            
            features['flux_g'] = photometry(g_channel)
            features['flux_r'] = photometry(r_channel)
            features['flux_z'] = photometry(z_channel)
            
            # 6. 集中度指数 (Concentration Index)
            def calculate_concentration(channel, centroid):
                # 计算增长曲线
                radii = np.linspace(1, min(centroid[0], centroid[1], 
                                         channel.shape[0] - centroid[0], 
                                         channel.shape[1] - centroid[1]), 50)
                
                fluxes = []
                for r in radii:
                    aper = CircularAperture([(centroid[1], centroid[0])], r=r)
                    data = NDData(channel)
                    phot_table = aperture_photometry(data, aper)
                    fluxes.append(phot_table['aperture_sum'][0])
                
                total_flux = fluxes[-1]
                if total_flux == 0:
                    return 0
                
                # 找到 R20 和 R80
                flux_20 = total_flux * 0.2
                flux_80 = total_flux * 0.8
                
                # 线性插值找到对应的半径
                r20 = np.interp(flux_20, fluxes, radii)
                r80 = np.interp(flux_80, fluxes, radii)
                
                if r20 == 0:
                    return 0
                
                # 计算集中度指数
                C = 5 * np.log10(r80 / r20)
                return C
            
            features['concentration_index'] = calculate_concentration(z_channel, (centroid_y, centroid_x))
            
            # 7. 形态特征
            mask = (labeled == largest_label).astype(np.uint8)
            region = measure.regionprops(mask)[0]
            features['eccentricity'] = region.eccentricity
            features['solidity'] = region.solidity
            features['extent'] = region.extent
            
            # 8. 颜色特征
            if features['flux_g'] > 0 and features['flux_r'] > 0:
                features['color_gr'] = -2.5 * np.log10(features['flux_g'] / features['flux_r'])
            else:
                features['color_gr'] = 0
            
            if features['flux_r'] > 0 and features['flux_z'] > 0:
                features['color_rz'] = -2.5 * np.log10(features['flux_r'] / features['flux_z'])
            else:
                features['color_rz'] = 0
        else:
            # 没有检测到星系
            features['flux_g'] = 0
            features['flux_r'] = 0
            features['flux_z'] = 0
            features['concentration_index'] = 0
            features['eccentricity'] = 0
            features['color_gr'] = 0
            features['color_rz'] = 0
        
        # 9. 基本统计特征
        features['bg_g'] = bg_g
        features['bg_r'] = bg_r
        features['bg_z'] = bg_z
        features['mean_g'] = g_channel.mean()
        features['mean_r'] = r_channel.mean()
        features['mean_z'] = z_channel.mean()
        features['std_g'] = g_channel.std()
        features['std_r'] = r_channel.std()
        features['std_z'] = z_channel.std()
        
        return features
    
    def get_catalog_features(self, index):
        """
        从星表中获取对应特征
        
        Args:
            index: 星系索引
            
        Returns:
            dict: 星表特征
        """
        index_str = str(index)
        if index_str not in self.catalog_dict:
            return None
        
        row = self.catalog_dict[index_str]
        features = {}
        
        # 提取关键特征
        flux_columns = ['DESIDR1_FLUX_G', 'DESIDR1_FLUX_R', 'DESIDR1_FLUX_Z', 
                       'DESIDR1_FLUX_W1', 'DESIDR1_FLUX_W2']
        
        for col in flux_columns:
            val_str = row.get(col, '')
            if val_str and val_str not in ('', 'NA_Value', '-99.0', 'NaN', 'nan', 'None'):
                try:
                    features[col] = float(val_str)
                except:
                    features[col] = 0.0
            else:
                features[col] = 0.0
        
        # 形态特征
        features['DESIDR1_SHAPE_R'] = float(row.get('DESIDR1_SHAPE_R', 0))
        features['DESIDR1_SHAPE_E1'] = float(row.get('DESIDR1_SHAPE_E1', 0))
        features['DESIDR1_SHAPE_E2'] = float(row.get('DESIDR1_SHAPE_E2', 0))
        features['DESIDR1_SERSIC'] = float(row.get('DESIDR1_SERSIC', 0))
        
        # 形态评分
        score_columns = ['mapped_smooth_combined_score', 
                        'mapped_barred_spirals_combined_score', 
                        'mapped_unbarred_spirals_combined_score', 
                        'mapped_edge_on_with_bulge_combined_score', 
                        'mapped_mergers_combined_score', 
                        'mapped_irregular_combined_score']
        
        for col in score_columns:
            val_str = row.get(col, '')
            if val_str and val_str not in ('', 'NA_Value', '-99.0', 'NaN', 'nan', 'None'):
                try:
                    features[col] = float(val_str)
                except:
                    features[col] = 0.0
            else:
                features[col] = 0.0
        
        # 物理特征
        features['DESIDR1_Z'] = float(row.get('DESIDR1_Z', 0))
        features['MassEMLine_MASS_CG'] = float(row.get('MassEMLine_MASS_CG', 0))
        features['MassEMLine_SFR_CG'] = float(row.get('MassEMLine_SFR_CG', 0))
        features['MassEMLine_AGE_CG'] = float(row.get('MassEMLine_AGE_CG', 0))
        features['MassEMLine_HALPHA_FLUX'] = float(row.get('MassEMLine_HALPHA_FLUX', 0))
        features['MassEMLine_OIII5007_FLUX'] = float(row.get('MassEMLine_OIII5007_FLUX', 0))
        features['MassEMLine_EBV'] = float(row.get('MassEMLine_EBV', 0))
        
        return features
    
    def validate_image(self, image_path, index):
        """
        验证单个图像（天体物理标准）
        
        Args:
            image_path: 图像路径
            index: 星系索引
            
        Returns:
            dict: 验证结果
        """
        # 常量定义
        PIXEL_SCALE = 0.104  # 0.104 arcsec/pix
        EBV_R_COEFF = 2.27  # R 波段的 EBV 红化系数
        EBV_G_COEFF = 3.30  # G 波段的 EBV 红化系数
        
        # 获取星表特征
        cat_features = self.get_catalog_features(index)
        
        if cat_features is None:
            return {
                'index': index,
                'error': f'Index {index} not found in catalog'
            }
        
        # 提取图像特征（传入星表的 SHAPE_R 用于孔径计算）
        img_features = self.extract_image_features(image_path, cat_features.get('DESIDR1_SHAPE_R'))
        
        # 计算验证指标
        validation = {
            'index': index,
            'image_features': img_features,
            'catalog_features': cat_features,
            'metrics': {}
        }
        
        # 1. 通量验证（按波段）
        flux_g_catalog = cat_features.get('DESIDR1_FLUX_G', 0)
        flux_r_catalog = cat_features.get('DESIDR1_FLUX_R', 0)
        flux_z_catalog = cat_features.get('DESIDR1_FLUX_Z', 0)
        
        flux_g_image = img_features.get('flux_g', 0)
        flux_r_image = img_features.get('flux_r', 0)
        flux_z_image = img_features.get('flux_z', 0)
        
        if flux_g_catalog > 0:
            validation['metrics']['flux_g_ratio'] = flux_g_image / flux_g_catalog
        else:
            validation['metrics']['flux_g_ratio'] = 0
            
        if flux_r_catalog > 0:
            validation['metrics']['flux_r_ratio'] = flux_r_image / flux_r_catalog
        else:
            validation['metrics']['flux_r_ratio'] = 0
            
        if flux_z_catalog > 0:
            validation['metrics']['flux_z_ratio'] = flux_z_image / flux_z_catalog
        else:
            validation['metrics']['flux_z_ratio'] = 0
        
        # 2. 颜色验证（考虑 EBV 红化修正）
        ebv = cat_features.get('MassEMLine_EBV', 0)
        
        # 计算理论颜色（考虑红化修正）
        if flux_g_catalog > 0 and flux_r_catalog > 0:
            # 计算理论颜色（未修正红化）
            color_gr_theory = -2.5 * np.log10(flux_g_catalog / flux_r_catalog)
            # 应用红化修正
            color_gr_theory_corrected = color_gr_theory + (EBV_G_COEFF - EBV_R_COEFF) * ebv
            validation['metrics']['color_gr_theory'] = color_gr_theory
            validation['metrics']['color_gr_theory_corrected'] = color_gr_theory_corrected
            
            # 与测量颜色对比
            color_gr_measured = img_features.get('color_gr', 0)
            validation['metrics']['color_gr_measured'] = color_gr_measured
            validation['metrics']['color_gr_diff'] = abs(color_gr_measured - color_gr_theory_corrected)
        
        # 3. 形态验证
        expected_eccentricity = np.sqrt(cat_features['DESIDR1_SHAPE_E1']**2 + cat_features['DESIDR1_SHAPE_E2']**2)
        actual_eccentricity = img_features.get('eccentricity', 0)
        validation['metrics']['eccentricity_diff'] = abs(actual_eccentricity - expected_eccentricity)
        
        # 4. 大小验证（单位对齐）
        expected_size = cat_features['DESIDR1_SHAPE_R']  # arcsec
        # 从图像测量的大小（通过孔径半径转换）
        if 'centroid_x' in img_features and 'centroid_y' in img_features:
            # 计算等效半径（从面积转换）
            if 'flux_z' in img_features and img_features['flux_z'] > 0:
                # 使用集中度指数估算大小
                # 这里简化处理，实际应该从增长曲线中获取
                pass
        validation['metrics']['size_arcsec_expected'] = expected_size
        
        # 5. 集中度指数验证（验证 SERSIC）
        concentration_index = img_features.get('concentration_index', 0)
        sersic_index = cat_features.get('DESIDR1_SERSIC', 0)
        validation['metrics']['concentration_index'] = concentration_index
        validation['metrics']['sersic_index'] = sersic_index
        
        # 6. 整体质量评估
        flux_ratios = [v for k, v in validation['metrics'].items() if k.endswith('_ratio') and v > 0]
        if flux_ratios:
            avg_flux_ratio = np.mean(flux_ratios)
            validation['metrics']['avg_flux_ratio'] = avg_flux_ratio
        
        # 7. 形态类型验证
        max_score = max(cat_features[col] for col in [
            'mapped_smooth_combined_score',
            'mapped_barred_spirals_combined_score',
            'mapped_unbarred_spirals_combined_score',
            'mapped_edge_on_with_bulge_combined_score',
            'mapped_mergers_combined_score',
            'mapped_irregular_combined_score'
        ])
        validation['metrics']['max_morph_score'] = max_score
        
        return validation
    
    def validate_directory(self, output_dir, catalog_file):
        """
        验证目录中的所有图像
        
        Args:
            output_dir: 图像输出目录
            catalog_file: 星表文件路径
            
        Returns:
            dict: 验证结果汇总
        """
        results = []
        
        for img_name in os.listdir(output_dir):
            if img_name.endswith('.png'):
                index_str = img_name.split('.')[0]
                try:
                    index = int(index_str)
                    img_path = os.path.join(output_dir, img_name)
                    result = self.validate_image(img_path, index)
                    results.append(result)
                    print(f"✓ 验证完成: {img_name}")
                except Exception as e:
                    print(f"✗ 验证失败 {img_name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 汇总结果
        summary = {
            'total_images': len(results),
            'valid_images': len([r for r in results if 'error' not in r]),
            'metrics': {}
        }
        
        if results:
            # 计算平均指标
            valid_results = [r for r in results if 'error' not in r]
            
            # 通量比率
            flux_g_ratios = [r['metrics'].get('flux_g_ratio', 0) for r in valid_results if r['metrics'].get('flux_g_ratio', 0) > 0]
            flux_r_ratios = [r['metrics'].get('flux_r_ratio', 0) for r in valid_results if r['metrics'].get('flux_r_ratio', 0) > 0]
            flux_z_ratios = [r['metrics'].get('flux_z_ratio', 0) for r in valid_results if r['metrics'].get('flux_z_ratio', 0) > 0]
            avg_flux_ratios = [r['metrics'].get('avg_flux_ratio', 0) for r in valid_results if r['metrics'].get('avg_flux_ratio', 0) > 0]
            
            # 颜色差异
            color_gr_diffs = [r['metrics'].get('color_gr_diff', 0) for r in valid_results if 'color_gr_diff' in r['metrics']]
            
            # 形态指标
            eccentricity_diffs = [r['metrics'].get('eccentricity_diff', 0) for r in valid_results if 'eccentricity_diff' in r['metrics']]
            
            # 集中度指数
            concentration_indices = [r['metrics'].get('concentration_index', 0) for r in valid_results if 'concentration_index' in r['metrics']]
            sersic_indices = [r['metrics'].get('sersic_index', 0) for r in valid_results if 'sersic_index' in r['metrics']]
            
            # 形态评分
            max_scores = [r['metrics'].get('max_morph_score', 0) for r in valid_results if 'max_morph_score' in r['metrics']]
            
            # 填充汇总指标
            if flux_g_ratios:
                summary['metrics']['avg_flux_g_ratio'] = np.mean(flux_g_ratios)
            if flux_r_ratios:
                summary['metrics']['avg_flux_r_ratio'] = np.mean(flux_r_ratios)
            if flux_z_ratios:
                summary['metrics']['avg_flux_z_ratio'] = np.mean(flux_z_ratios)
            if avg_flux_ratios:
                summary['metrics']['avg_flux_ratio'] = np.mean(avg_flux_ratios)
            if color_gr_diffs:
                summary['metrics']['avg_color_gr_diff'] = np.mean(color_gr_diffs)
            if eccentricity_diffs:
                summary['metrics']['avg_eccentricity_diff'] = np.mean(eccentricity_diffs)
            if concentration_indices:
                summary['metrics']['avg_concentration_index'] = np.mean(concentration_indices)
            if sersic_indices:
                summary['metrics']['avg_sersic_index'] = np.mean(sersic_indices)
            if max_scores:
                summary['metrics']['avg_max_morph_score'] = np.mean(max_scores)
        
        return summary, results

def main():
    # 硬编码路径
    OUTPUT_DIR = "output/2026-03-23_catalog_infer"  # 图像输出目录
    CATALOG_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/labeled_catalog_cleaned_v2.csv"  # 星表文件路径
    VALIDATION_DIR = "validation"  # 验证结果输出目录
    
    # 确保验证目录存在
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    
    validator = GalaxyValidator(CATALOG_FILE)
    summary, results = validator.validate_directory(OUTPUT_DIR, CATALOG_FILE)
    
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    print(f"总图像数: {summary['total_images']}")
    print(f"有效图像: {summary['valid_images']}")
    
    if summary['metrics']:
        print("\n平均指标:")
        for key, value in summary['metrics'].items():
            print(f"  {key}: {value:.4f}")
    
    # 保存详细结果
    import json
    output_file = os.path.join(VALIDATION_DIR, 'validation_results.json')
    with open(output_file, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)
    
    print(f"\n详细结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
