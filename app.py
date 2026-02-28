import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

# 初始化 Session State (儲存航行數據)
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'total_dist' not in st.session_state: st.session_state.total_dist = 0.0
if 'total_time' not in st.session_state: st.session_state.total_time = 0.0
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'pred_p' not in st.session_state: st.session_state.pred_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# --- 2. 側邊欄控制台 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
loc_mode = st.sidebar.radio("起始點選擇", ["立即定位 (GPS 模擬)", "自行輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    s_lat, s_lon = 23.184, 121.739
    st.sidebar.info(f"📍 GPS 座標: {s_lat}, {s_lon}")
else:
    s_lat = st.sidebar.number_input("起始緯度", value=23.184, format="%.3f")
    s_lon = st.sidebar.number_input("起始經度", value=121.739, format="%.3f")

d_lat = st.sidebar.number_input("終點緯度", value=25.500, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=121.800, format="%.3f")

# --- 3. 【效能優化版】數據讀取函數 ---
@st.cache_data(ttl=3600)
def get_fast_ocean_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        # 優化點：1. 縮小地理範圍 2. 跳點抽樣 (isel step=3) 減少 90% 資料量
        subset = ds.sel(
            lat=slice(21.0, 26.5), 
            lon=slice(118.5, 124.0), 
            depth=0
        ).isel(
            time=-1, 
            lat=slice(None, None, 3), 
            lon=slice(None, None, 3)
        ).load()
        return subset
    except Exception as e:
        return None

# --- 4. 執行路徑分析 (含避障邏輯) ---
def plan_paths(slat, slon, dlat, dlon):
    steps = 25
    lats = np.linspace(slat, dlat, steps)
    lons = np.linspace(slon, dlon, steps)
    r_path, p_path = [], []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        # 台灣陸地避障
        if 120.0 < lo < 122.2 and 21.9 < la < 25.4:
            lo = 122.6
        r_path.append((la, lo))
        p_path.append((la, lo - 0.12 if i > 5 else lo)) # 模擬預測誤差
    return r_path, p_path

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    with st.spinner('📡 正在介接衛星流場數據...'):
        st.session_state.real_p, st.session_state.pred_p = plan_paths(s_lat, s_lon, d_lat, d_lon)
        st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
        st.session_state.step_idx, st.session_state.total_dist, st.session_state.total_time = 0, 0.0, 0.0

# --- 5. 數據計算與儀表板 ---
subset = get_fast_ocean_data()

if subset is not None:
    # 取得當前位置流速
    curr_pt = subset.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u, v = float(curr_pt.water_u), float(curr_pt.water_v)
    
    # 計算各項指標
    sog = 15.0 + (u * 1.94)
    fuel = 25.4 if u > 0.4 else 12.5
    rem_dist = np.sqrt((d_lat - st.session_state.ship_lat)**2 + (d_lon - st.session_state.ship_lon)**2) * 60
    head = np.degrees(np.arctan2(v, u)) % 360
    
    # 儀表板顯示
    st.subheader("📊 HELIOS 衛星導航即時儀表板")
    r1 = st.columns(4)
    r1[0].metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
    r1[1].metric("⛽ 能源紅利", f"{fuel}%")
    r1[2].metric("🎯 剩餘距離", f"{rem_dist:.1f} nmi")
    r1[3].metric("🧭 建議航向", f"{head:.0f}°")
    
    r2 = st.columns(3)
    r2[0].metric("📡 衛星接收", "穩定 (98.2%)", "LEO-Link")
    r2[1].metric("📏 航行總距離", f"{st.session_state.total_dist:.1f} nmi")
    r2[2].metric("🕒 航行總時間", f"{st.session_state.total_time:.2f} hrs")

    # --- 6. 地圖繪圖區 ---
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    # 海流格子底圖 (關鍵：使用 speed_grid)
    speed_grid = np.sqrt(subset.water_u**2 + subset.water_v**2)
    mesh = ax.pcolormesh(subset.lon, subset.lat, speed_grid, cmap='YlGnBu', alpha=0.5, shading='auto')
    plt.colorbar(mesh, ax=ax, label='Current Speed (m/s)', fraction=0.03, pad=0.04)

    # 陸地
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#2c2c2c')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white')

    # 繪製路徑：正確路徑(紅實線) vs 預測路徑(白虛線)
    if st.session_state.real_p:
        px, py = [p[1] for p in st.session_state.pred_p], [p[0] for p in st.session_state.pred_p]
        ax.plot(px, py, color='white', linestyle='--', linewidth=1, label='Forecast (Predicted)')
        
        rx, ry = [p[1] for p in st.session_state.real_p], [p[0] for p in st.session_state.real_p]
        ax.plot(rx, ry, color='red', linestyle='-', linewidth=2.5, label='HELIOS Optimized (Actual)')

    # 船隻標記
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, edgecolors='white', zorder=5)
    ax.set_extent([118.5, 124.5, 21.0, 26.5])
    ax.legend(loc='lower right')
    st.pyplot(fig)

# --- 7. 移動模擬 ---
if st.button("🚢 執行下一步移動"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.session_state.total_dist += sog * 0.5
        st.session_state.total_time += 0.5
        st.rerun()
