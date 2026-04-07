import os
import pandas as pd
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, ApertureStats
from photutils.centroids import centroid_com
from skimage.measure import regionprops, label
from skimage.transform import rotate

# ==========================================
# ⚙️ 科学配置 (Scientific Config)
# ==========================================
class Config:
    IMG_DIR = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/validation/val_results_150k_dps/fake_only"
    CAT_FILE = "/nfsdata/share/ljc/DESI_data/data_catalog_weighted-mass_emline-z-by_readeID/validation/validation_catalog_2500.csv"
    SAVE_DIR = "validation"
    
    PIXEL_SCALE = 0.262  # arcsec/pixel
    EBV_COEFFS = {'G': 3.214, 'R': 2.165, 'Z': 1.211}
    CHANNELS = {'G': 2, 'R': 1, 'Z': 0} # 对应图像 RGB 索引

# ==========================================
# 🛠️ 核心验证套件
# ==========================================
class GalaxyScientificSuiteV3:
    def __init__(self):
        print(f"[*] 正在初始化验证引擎...")
        try:
            self.cat = pd.read_csv(Config.CAT_FILE)
            self.cat['index_str'] = self.cat['index'].astype(str)
            self.cat.set_index('index_str', inplace=True)
            os.makedirs(Config.SAVE_DIR, exist_ok=True)
        except Exception as e:
            print(f"[!] 初始化失败，请检查路径: {e}")

    def measure_galaxy(self, img_path, cat_row):
        """执行全方位物理与形态测量"""
        try:
            # 1. 图像加载与归一化
            raw_img = np.array(Image.open(img_path)).astype(float) / 255.0
            
            # 2. 逐波段背景扣除 (Sigma-Clipped)
            data_dict = {}
            for band, idx in Config.CHANNELS.items():
                band_data = raw_img[:, :, idx]
                _, median, _ = sigma_clipped_stats(band_data, sigma=3.0)
                data_dict[band] = np.maximum(band_data - median, 0)

            # 3. 质心检测与孔径设置
            if np.sum(data_dict['R']) > 0:
                y_c, x_c = centroid_com(data_dict['R'])
            else:
                y_c, x_c = 96, 96  # 默认中心位置
            
            re_pix = cat_row['DESIDR1_SHAPE_R'] / Config.PIXEL_SCALE
            re_pix = max(1.0, re_pix)  # 确保最小有效半径
            ap_r = max(5.0, re_pix * 2.5)
            aperture = CircularAperture((x_c, y_c), r=ap_r)

            res = {'id': cat_row.name}

            # 4. 增加通量分析 (Per-band Flux Analysis)
            for band in ['G', 'R', 'Z']:
                meas_flux = ApertureStats(data_dict[band], aperture).sum
                cat_flux = float(cat_row[f'DESIDR1_FLUX_{band}'])
                res[f'flux_{band.lower()}_meas'] = meas_flux
                res[f'flux_{band.lower()}_ratio'] = meas_flux / cat_flux if cat_flux > 0 else np.nan

            # 5. 集中度分析 (Concentration)
            z_data = data_dict['Z']
            radii = np.linspace(1, ap_r * 2, 50)
            flux_prof = [ApertureStats(z_data, CircularAperture((x_c, y_c), r=r)).sum for r in radii]
            if flux_prof[-1] > 0:
                # 确保找到有效的 r20 和 r80
                if flux_prof[0] < 0.2 * flux_prof[-1] and flux_prof[0] < 0.8 * flux_prof[-1]:
                    r20 = np.interp(0.2 * flux_prof[-1], flux_prof, radii)
                    r80 = np.interp(0.8 * flux_prof[-1], flux_prof, radii)
                    if r20 > 0 and r80 > r20:
                        res['C_meas'] = 5 * np.log10(r80 / r20)
                    else:
                        res['C_meas'] = np.nan
                else:
                    res['C_meas'] = np.nan
            else:
                res['C_meas'] = np.nan

            # 6. 增加形态分析 (Eccentricity & Asymmetry)
            # 使用 Z 波段二值化进行形状测量
            thresh = np.percentile(z_data, 90)
            binary = z_data > thresh
            labeled = label(binary)
            props = regionprops(labeled, intensity_image=z_data)
            
            if props:
                # 寻找最靠近中心的区域
                main_prop = sorted(props, key=lambda p: (p.centroid[0]-96)**2 + (p.centroid[1]-96)**2)[0]
                res['eccentricity_meas'] = main_prop.eccentricity
            else:
                res['eccentricity_meas'] = np.nan

            # 不对称度
            img_rot = rotate(data_dict['R'], 180, center=(x_c, y_c), preserve_range=True)
            res['A_meas'] = np.sum(np.abs(data_dict['R'] - img_rot)) / (np.sum(data_dict['R']) + 1e-6)

            return res

        except Exception as e:
            print(f"[!] 测量失败: {e}")
            return None

    def run(self):
        img_files = [f for f in os.listdir(Config.IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        records = []

        print(f"[*] 正在分析图像...")
        for fname in tqdm(img_files):
            idx = str(os.path.splitext(fname)[0])  # 更可靠的扩展名处理
            if idx not in self.cat.index:
                print(f"[!] 索引不存在: {idx}")
                continue
            
            row = self.cat.loc[idx]
            m = self.measure_galaxy(os.path.join(Config.IMG_DIR, fname), row)
            
            if m is None:
                print(f"[!] 测量失败: {fname}")
                continue
            
            # 物理颜色理论值计算
            ebv = float(row['MassEMLine_EBV'])
            try:
                if row['DESIDR1_FLUX_G'] > 0 and row['DESIDR1_FLUX_R'] > 0:
                    c_int = -2.5 * np.log10(row['DESIDR1_FLUX_G'] / row['DESIDR1_FLUX_R'])
                    m['color_theory'] = c_int + (Config.EBV_COEFFS['G'] - Config.EBV_COEFFS['R']) * ebv
                else:
                    m['color_theory'] = np.nan
            except Exception as e:
                m['color_theory'] = np.nan
            
            try:
                if m['flux_g_meas'] > 0 and m['flux_r_meas'] > 0:
                    m['color_meas'] = -2.5 * np.log10(m['flux_g_meas'] / m['flux_r_meas'])
                else:
                    m['color_meas'] = np.nan
            except Exception as e:
                m['color_meas'] = np.nan
            
            m['sersic_cat'] = row['DESIDR1_SERSIC']
            m['ecc_cat'] = np.sqrt(row['DESIDR1_SHAPE_E1']**2 + row['DESIDR1_SHAPE_E2']**2)
            records.append(m)

        # 统计分析与保存
        df = pd.DataFrame(records).dropna()
        df.to_csv(os.path.join(Config.SAVE_DIR, "v3_metrics.csv"), index=False)
        self.plot_results(df)

    def plot_results(self, df):
        """生成四象限科研报告图"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        sns.set_context("talk")

        # 1. 颜色 reddening (核心物理)
        sns.regplot(ax=axes[0,0], data=df, x='color_theory', y='color_meas', scatter_kws={'alpha':0.3, 's':10})
        axes[0,0].set_title(f"Color Consistency\nCorr: {df['color_theory'].corr(df['color_meas']):.3f}")

        # 2. 通量比例分布 (通量稳定性)
        flux_ratios = df[['flux_g_ratio', 'flux_r_ratio', 'flux_z_ratio']]
        sns.boxplot(ax=axes[0,1], data=pd.melt(flux_ratios), x='variable', y='value')
        axes[0,1].set_title("Flux Ratio Stability (Meas/Cat)")
        axes[0,1].set_yscale('log')

        # 3. 结构集中度
        sns.regplot(ax=axes[1,0], data=df, x='sersic_cat', y='C_meas', scatter_kws={'alpha':0.3, 's':10})
        axes[1,0].set_title(f"Structure: C vs Sersic n\nCorr: {df['sersic_cat'].corr(df['C_meas']):.3f}")

        # 4. 形态偏心率 (新增)
        sns.regplot(ax=axes[1,1], data=df, x='ecc_cat', y='eccentricity_meas', scatter_kws={'alpha':0.3, 's':10}, color='purple')
        axes[1,1].set_title(f"Morphology: Eccentricity Consistency\nCorr: {df['ecc_cat'].corr(df['eccentricity_meas']):.3f}")

        plt.tight_layout()
        plt.savefig(os.path.join(Config.SAVE_DIR, "v3_report.png"), dpi=300)
        print(f"[√] 报告已生成: {Config.SAVE_DIR}")

if __name__ == "__main__":
    GalaxyScientificSuiteV3().run()