import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd

# ==========================================
# 1. 獲取 HYCOM 實時流場數據 (與導航系統連動)
# ==========================================
def get_hycom_speed_data():
    tds_url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        # 採用避開時間解碼錯誤的設定
        ds = xr.open_dataset(tds_url, decode_times=False)
        subset = ds.isel(time=-1).sel(
            depth=0,
            lon=slice(118, 124),
            lat=slice(20, 27)
        ).load()
        
        # 計算流速強度 (Current Speed) = sqrt(u^2 + v^2)
        u = np.nan_to_num(subset.water_u.values)
        v = np.nan_to_num(subset.water_v.values)
        speed = np.sqrt(u**2 + v**2)
        
        # 取得數據時間
        try:
            decoded_ds = xr.decode_cf(subset)
            data_ts = pd.to_datetime(decoded_ds.time.values).strftime('%Y-%m-%d %H:%M')
        except:
            data_ts = "Latest Forecast"
            
        return subset.lat.values, subset.lon.values, speed, data_ts
    except Exception as e:
        print(f"Error fetching HYCOM: {e}")
        return None, None, None, None

lat, lon, speed_data, data_time = get_hycom_speed_data()

# ==========================================
# 2. 繪製綠色色階流速對比圖 (全英文設定)
# ==========================================
if lat is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), 
                                   subplot_kw={'projection': ccrs.PlateCarree()})

    def draw_speed_map(ax, title, config_info):
        # 設定範圍完整顯示台灣不卡到
        ax.set_extent([117.5, 124.0, 21.0, 27.0], crs=ccrs.PlateCarree())
        
        # 使用綠色色階 (YlGn: 黃到綠) 繪製流速背景
        # vmin/vmax 根據海流特性設定，通常 0.0 到 1.5 m/s 較明顯
        im = ax.pcolormesh(lon, lat, speed_data, cmap='YlGn', 
                           shading='auto', alpha=0.9, vmin=0, vmax=1.2)
        
        # 深灰色陸地
        ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2)
        ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.2, zorder=3)
        
        # 英文標題與說明文字
        ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
        ax.text(0.5, -0.12, config_info, transform=ax.transAxes, 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        return im

    # 左圖：36顆星模擬場景
    im1 = draw_speed_map(ax1, "Simulation: 36 LEO Satellite Constellation", 
                         f"Background: HYCOM Current Speed ({data_time})\n"
                         "Settings: 3 Planes x 12 Sat | Inclination: 97.5°\n"
                         "Comm. Performance: High Reliability | Latency < 50ms")

    # 右圖：玉山衛星場景
    im2 = draw_speed_map(ax2, "Comparison: TASA Yushan Satellite (YUSAT)", 
                         f"Background: HYCOM Current Speed ({data_time})\n"
                         "Settings: Single LEO Sat | Band: UHF / SOTDMA\n"
                         "Comm. Performance: Variable Reliability | SSO 97.5°")

    # 統一色條設定 (流速強度)
    cbar_ax = fig.add_axes([0.94, 0.25, 0.015, 0.5])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Current Speed (m/s)', fontsize=14, fontweight='bold')

    plt.subplots_adjust(wspace=0.18)
    plt.show()
else:
    print("Failed to load map data.")
