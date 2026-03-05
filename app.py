import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd
from datetime import datetime, timezone

# ===============================
# 1. 數據源嫁接：切換至 ESPC-D-V02 (最新分析場)
# ===============================
@st.cache_data(ttl=300)
def get_espc_latest_data():
    # 這是目前 HYCOM 唯一的 ACTIVE 實時數據源網址
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/uv3z"
    
    try:
        # 關閉解碼以處理 2026 年的時間戳
        ds = xr.open_dataset(url, decode_times=False)
        
        # 獲取現在的 UTC 時間
        now_utc = datetime.now(timezone.utc)
        
        # 解析時間單位 (例如: hours since 2024-08-10)
        units = ds.time.units
        base_date = pd.to_datetime(units.split("since ")[1].split(".")[0]).replace(tzinfo=timezone.utc)
        
        # 計算現在的小時偏移量
        target_hours = (now_utc - base_date).total_seconds() / 3600
        
        # 尋找與現在最接近的時間索引
        time_vals = ds.time.values
        idx = np.abs(time_vals - target_hours).argmin()
        
        # 抓取台灣海域
        subset = ds.isel(time=idx).sel(
            depth=0, lon=slice(117.2, 124.8), lat=slice(20.2, 26.8)
        ).load()
        
        u = np.nan_to_num(subset.water_u.values)
        v = np.nan_to_num(subset.water_v.values)
        
        # 顯示同步時間
        sync_time = (base_date + pd.Timedelta(hours=float(time_vals[idx]))).strftime('%Y-%m-%d %H:%M UTC')
        
        return subset.lat.values, subset.lon.values, u, v, sync_time
    except Exception as e:
        st.error(f"數據源 ESPC 嫁接失敗: {e}")
        return None, None, None, None, "Link Fail"

# ===============================
# 2. 系統介面與導航邏輯 (固定動力)
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V15 ESPC-Sync")
st.title("🛰️ HELIOS 智慧導航 (ESPC 實時同源版)")

lat, lon, u_raw, v_raw, live_time = get_espc_latest_data()

if lat is not None:
    st.success(f"✅ 成功對接 ESPC-D-V02 數據：{live_time}")
    
    # 建立 2D 遮罩 (修正括號問題)
    LON, LAT = np.meshgrid(lon, lat)
    mask_tw = (((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1
    mask_ph = (((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1
    forbidden = mask_tw | mask_ph

    with st.sidebar:
        st.header("📍 任務參數")
        s_lat = st.number_input("起點緯度", value=22.35)
        s_lon = st.number_input("起點經度", value=120.10)
        e_lat = st.number_input("終點緯度", value=25.20)
        e_lon = st.number_input("終點經度", value=122.00)
        base_speed = st.slider("航速 (kn)", 10, 25, 15)
        run_btn = st.button("🚀 執行路徑計算")

    # (此處執行 A* 演算法，代碼同前)
    # ...

    # ===============================
    # 3. 地圖視覺化 (比照 NODASS 风格)
    # ===============================
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    speed = np.sqrt(u_raw**2 + v_raw**2)
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', alpha=0.9)
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)')
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.5, zorder=6)
    
    # 標註台灣主要港口做參考
    ax.text(120.1, 22.6, 'Kaohsiung', color='white', fontsize=8, transform=ccrs.PlateCarree(), zorder=7)
    
    st.pyplot(fig)
