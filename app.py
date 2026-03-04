import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq
import time
import xarray as xr
import pandas as pd

# ===============================
# 1. 核心數學工具 (計算方位與距離)
# ===============================
def calc_bearing(p1, p2):
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半徑 (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 獲取 HYCOM 實時資料
# ===============================
@st.cache_data(ttl=3600)
def get_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.isel(time=-1).sel(depth=0, lon=slice(118, 124), lat=slice(20, 27)).load()
        u, v = np.nan_to_num(subset.water_u.values), np.nan_to_num(subset.water_v.values)
        speed_grid = np.sqrt(u**2 + v**2)
        try:
            dt = pd.to_datetime(xr.decode_cf(subset).time.values).strftime('%m-%d %H:%M')
        except:
            dt = "Real-time"
        return subset.lat.values, subset.lon.values, u, v, speed_grid, dt
    except:
        return None, None, None, None, None, None

lat, lon, u, v, speed_grid, ocean_time = get_hycom_data()

# ===============================
# 3. 頁面配置與儀表板邏輯
# ===============================
st.set_page_config(layout="wide")
st.title("🛰️ HELIOS 智慧航行系統")

if lat is not None:
    # --- 側邊欄輸入 ---
    with st.sidebar:
        st.header("Navigation Parameters")
        s_lat = st.number_input("Start Lat", value=22.30)
        s_lon = st.number_input("Start Lon", value=120.00)
        e_lat = st.number_input("End Lat", value=25.20)
        e_lon = st.number_input("End Lon", value=122.00)
        ship_speed = 15.0 # 假設航速 15 節
        run_btn = st.button("🚀 Calculate Optimized Route", use_container_width=True)

    # --- A* 演算法 (簡化邏輯) ---
    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    start_idx, goal_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)
    
    # 預設儀表板變數
    dist_km, fuel_bonus, eta, brg = 0.0, 0.0, 0.0, "---"
    stream_status = "穩定"
    satellite_now = 36 # 你的模擬星座

    # --- 計算路徑 ---
    path = None
    if run_btn:
        # 這裡套用你原本的 astar 函數 (略)
        # 假設 path 已經產出...
        # [此處應插入 A* 演算法代碼]
        pass 

    # --- 儀表板顯示 (兩行) ---
    r1 = st.columns(4)
    r1[0].metric("🚀 航速", f"{ship_speed:.1f} kn")
    r1[1].metric("⛽ 省油效益", f"{fuel_bonus:.1f}%") # 可根據順流程度計算
    r1[2].metric("📡 衛星", f"{satellite_now} Pcs")
    r1[3].metric("🌊 流場狀態", stream_status)

    r2 = st.columns(4)
    r2[0].metric("🧭 建議航向", brg)
    r2[1].metric("📏 預計距離", f"{dist_km:.1f} km")
    r2[2].metric("🕒 預計抵達", f"{eta:.1f} hr")
    r2[3].metric("🕒 流場時間", ocean_time)
    st.markdown("---")

    # ===============================
    # 4. 繪圖 (綠色色階底圖)
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.5, 123.5, 21.0, 26.5])
    
    # 背景流速色塊
    mesh = ax.pcolormesh(lon, lat, speed_grid, cmap='YlGn', alpha=0.7, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#222222', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', zorder=3)
    
    # 海流向量
    ax.quiver(lon[::6], lat[::6], u[::6, ::6], v[::6, ::6], color='cyan', alpha=0.3, zorder=4)
    
    # 起終點
    ax.scatter([s_lon, e_lon], [s_lat, e_lat], color=['lime', 'yellow'], s=100, zorder=5)
    
    st.pyplot(fig)
