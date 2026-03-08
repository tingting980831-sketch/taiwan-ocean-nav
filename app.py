import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ===============================
# 1. 系統核心設定 (400km, 3面4顆)
# ===============================
SAT_CONFIG = {
    "total_sats": 12,
    "planes": 3,
    "altitude_km": 400,
    "status": "Active (12/12)"
}

st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧導航系統")
st.markdown(f"**衛星系統：** {SAT_CONFIG['total_sats']} 顆 LEO | **軌道高度：** {SAT_CONFIG['altitude_km']}km | **狀態：** {SAT_CONFIG['status']}")

# ===============================
# 2. 讀取資料 (使用你確認可跑的邏輯)
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_custom():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        
        # 時間處理
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        time_values = time_origin + pd.to_timedelta(ds['time'].values, unit='h')
        
        # 範圍選取
        lat_slice = slice(20, 26)
        lon_slice = slice(119, 123)
        
        # 抓取最新一筆 (isel time=-1)
        data_u = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        data_v = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        
        return (data_u['lon'].values, data_u['lat'].values, 
                data_u.values, data_v.values, time_values[-1])
    except Exception as e:
        st.error(f"資料讀取錯誤: {e}")
        return None, None, None, None, None

lons, lats, u, v, latest_time = load_hycom_custom()

if lons is not None:
    st.sidebar.header("🚢 導航參數設定")
    base_speed = st.sidebar.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
    
    # 計算流速強度
    speed = np.sqrt(u**2 + v**2)

    # ===============================
    # 3. 視覺化繪圖 (Matplotlib + Cartopy)
    # ===============================
    # 建立畫布與地理投影
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 設定顯示範圍 (與你的 slice 一致)
    ax.set_extent([119, 123, 20, 26], crs=ccrs.PlateCarree())

    # --- 陸地顯示為灰色 (解決你的需求) ---
    ax.add_feature(cfeature.LAND, facecolor='gray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=3)
    ax.add_feature(cfeature.OCEAN, facecolor='#001525', zorder=0) # 深藍海域底色
    
    # 繪製流速熱圖 (背景)
    im = ax.pcolormesh(lons, lats, speed, cmap='viridis', shading='auto', alpha=0.6, zorder=1)
    plt.colorbar(im, label='流速 (m/s)', shrink=0.7)

    # 繪製海流箭頭 (Quiver)
    # 透過 [::2] 抽樣讓箭頭不要太擠
    skip = (slice(None, None, 2), slice(None, None, 2))
    ax.quiver(lons[::2], lats[::2], u[::2, ::2], v[::2, ::2], 
              color='white', scale=10, width=0.003, zorder=4, alpha=0.8)

    plt.title(f'台灣海域即時海流圖 (衛星同步時間: {latest_time})', color='black')
    
    # --- 重要：使用 Streamlit 顯示 ---
    st.pyplot(fig)
    
    # 狀態面板
    c1, c2 = st.columns(2)
    c1.info(f"📅 資料觀測時間: {latest_time}")
    c2.success(f"📡 衛星鏈路正常 (LEO Alt: {SAT_CONFIG['altitude_km']}km)")
