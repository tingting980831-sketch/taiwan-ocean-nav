import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time

# --- 1. 系統初始化與狀態儲存 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

# 初始化所有需要累計的數據
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'total_dist' not in st.session_state: st.session_state.total_dist = 0.0 # 航行總距離
if 'total_time' not in st.session_state: st.session_state.total_time = 0.0 # 航行總時間 (小時)
if 'real_p' not in st.session_state: st.session_state.real_p = [] # 紅色實線 (正確資料路徑)
if 'pred_p' not in st.session_state: st.session_state.pred_p = [] # 虛線 (推測資料路徑)
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

# --- 3. 路徑演算法 (含避障) ---
def calculate_paths(slat, slon, dlat, dlon):
    steps = 25
    lats = np.linspace(slat, dlat, steps)
    lons = np.linspace(slon, dlon, steps)
    
    r_path, p_path = [], []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        # 陸地避障：台灣島區域
        if 120.0 < lo < 122.2 and 21.8 < la < 25.4:
            lo = 122.6
        
        # 模擬正確海流路徑 (紅色實線 - 假設精準切入流軸)
        r_path.append((la, lo))
        # 模擬推測海流路徑 (虛線 - 帶有預測誤差的偏角)
        p_path.append((la, lo - 0.12 if i > 5 else lo))
        
    return r_path, p_path

if st.sidebar.button("🚀 執行路徑分析"):
    st.session_state.real_p, st.session_state.pred_p = calculate_paths(s_lat, s_lon, d_lat, d_lon)
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.step_idx = 0
    st.session_state.total_dist = 0.0
    st.session_state.total_time = 0.0

# --- 4. 數據讀取與計算 ---
@st.cache_data(ttl=3600)
def get_ocean_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    return xr.open_dataset(url, decode_times=False)

try:
    ds = get_ocean_data()
    subset = ds.sel(lat=slice(20, 27), lon=slice(118, 126), depth=0).isel(time=-1).load()
    speed_grid = np.sqrt(subset.water_u**2 + subset.water_v**2)
    
    # 取得船隻當前位置的精確流速
    curr_data = subset.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u_act = float(curr_data.water_u)
    v_act = float(curr_data.water_v)
    
    # --- 儀表板數值計算 ---
    engine_speed = 15.0 # 船隻推力 15節
    sog = engine_speed + (u_act * 1.94) # 節 (1 m/s = 1.94 knots)
    fuel_save = 25.4 if u_act > 0.4 else 12.5
    
    # 剩餘距離 (海里)
    rem_dist = np.sqrt((d_lat - st.session_state.ship_lat)**2 + (d_lon - st.session_state.ship_lon)**2) * 60
    
    # 建議航向
    suggested_head = np.degrees(np.arctan2(v_act, u_act)) % 360
    
except:
    sog, fuel_save, rem_dist, suggested_head = 15.0, 0.0, 100.0, 0.0
    speed_grid = None

# --- 5. 儀表板呈現 (Metrics) ---
st.subheader("📊 HELIOS 衛星導航即時儀表板")
row1 = st.columns(4)
row1[0].metric("🚀 當前航速 (SOG)", f"{sog:.1f} kn")
row1[1].metric("⛽ 能源紅利增益", f"{fuel_save}%")
row1[2].metric("🎯 剩餘距離", f"{rem_dist:.1f} nmi")
row1[3].metric("🧭 建議航向", f"{suggested_head:.0f}°")

row2 = st.columns(3)
row2[0].metric("📡 衛星接收強度", "穩定 (98.2%)", "LEO-Link")
row2[1].metric("📏 航行總距離", f"{st.session_state.total_dist:.1f} nmi")
row2[2].metric("🕒 航行總時間", f"{st.session_state.total_time:.2f} hrs")

# --- 6. 繪圖區 (地圖) ---
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#2c2c2c', zorder=2)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', zorder=3)

# 繪製海流底圖
if speed_grid is not None:
    mesh = ax.pcolormesh(subset.lon, subset.lat, speed_grid, cmap='YlGnBu', alpha=0.5, shading='auto')
    plt.colorbar(mesh, ax=ax, label='Current Speed (m/s)', fraction=0.03, pad=0.04)

# 繪製路徑
if st.session_state.real_p:
    # 1. 推測海流路徑 (白色虛線)
    px = [p[1] for p in st.session_state.pred_p]
    py = [p[0] for p in st.session_state.pred_p]
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, label='Forecast Route (Predicted)')
    
    # 2. 正確海流路徑 (紅色實線)
    rx = [p[1] for p in st.session_state.real_p]
    ry = [p[0] for p in st.session_state.real_p]
    ax.plot(rx, ry, color='red', linestyle='-', linewidth=2.5, label='HELIOS Optimized (Actual Data)')

# 標記船隻位置
ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, edgecolors='white', zorder=5)
ax.set_extent([118, 126, 20, 27])
ax.legend(loc='lower right')
st.pyplot(fig)

# --- 7. 模擬移動控制 ---
if st.button("🚢 執行下一步移動 (數據更新)"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        # 計算位移量 (簡單假設每步 0.5 小時)
        time_step = 0.5
        dist_step = sog * time_step
        
        # 更新狀態
        st.session_state.step_idx += 1
        new_loc = st.session_state.real_p[st.session_state.step_idx]
        st.session_state.ship_lat, st.session_state.ship_lon = new_loc
        
        st.session_state.total_dist += dist_step
        st.session_state.total_time += time_step
        st.rerun()
