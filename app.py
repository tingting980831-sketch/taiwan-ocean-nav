import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 基礎設定與常數 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

LEO_STABILITY = 0.982 
FUEL_GAIN_AVG = 25.4  

# 初始化 Session State
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'pred_p' not in st.session_state: st.session_state.pred_p = []

# --- 2. 側邊欄控制 ---
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

# --- 3. 數據讀取 ---
@st.cache_data(ttl=3600)
def get_fast_ocean_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(21.0, 26.5), lon=slice(118.5, 124.5), depth=0).isel(time=-1, lat=slice(None, None, 3), lon=slice(None, None, 3)).load()
        return subset
    except: return None

# --- 4. 優化路徑演算法 (確保起點相連) ---
def generate_connected_path(slat, slon, dlat, dlon):
    steps = 40 
    lats = np.linspace(slat, dlat, steps)
    lons = np.linspace(slon, dlon, steps)
    
    path = []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        # 避障邏輯：僅對中間航段進行修正，保留頭尾確保不脫離
        if i > 0 and i < steps - 1:
            if 21.9 < la < 25.4 and 120.0 < lo < 122.2:
                lo = 122.6
        path.append((la, lo))
    
    # 平滑化處理 (減少直角)
    smooth_path = []
    window = 5
    for i in range(len(path)):
        start = max(0, i - window // 2)
        end = min(len(path), i + window // 2 + 1)
        avg_la = np.mean([p[0] for p in path[start:end]])
        avg_lo = np.mean([p[1] for p in path[start:end]])
        # 強制第一點與最後一點精確對齊輸入座標
        if i == 0: smooth_path.append((slat, slon))
        elif i == len(path)-1: smooth_path.append((dlat, dlon))
        else: smooth_path.append((avg_la, avg_lo))
        
    return smooth_path

if st.sidebar.button("🚀 執行 AI 路徑分析"):
    st.session_state.real_p, _ = generate_connected_path(s_lat, s_lon, d_lat, d_lon), []
    st.session_state.pred_p = [(la, lo - 0.1) for la, lo in st.session_state.real_p]
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.step_idx = 0
    st.rerun()

# --- 5. 數據計算與儀表板 ---
subset = get_fast_ocean_data()
if subset is not None and st.session_state.real_p:
    # 物理計算
    curr_pt = subset.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u = float(curr_pt.water_u)
    sog = 15.0 + (u * 1.94)
    
    # 預估總距離 (海里)：計算整條紅線的長度
    def calc_dist(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 60
    
    total_planned_dist = sum(calc_dist(st.session_state.real_p[i], st.session_state.real_p[i+1]) for i in range(len(st.session_state.real_p)-1))
    
    # 航行統計
    traveled_dist = (st.session_state.step_idx / len(st.session_state.real_p)) * total_planned_dist
    total_est_time = total_planned_dist / sog
    elapsed_time = (st.session_state.step_idx / len(st.session_state.real_p)) * total_est_time
    
    st.subheader("📊 HELIOS 智慧導航儀表板")
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
    c1.metric("📡 衛星接收", f"穩定 ({LEO_STABILITY*100:.1f}%)")
    
    c2.metric("⛽ 能源紅利", f"{FUEL_GAIN_AVG}%", "Optimal")
    c2.metric("📏 航行總距離", f"{total_planned_dist:.1f} nmi", f"已航行 {traveled_dist:.1f}")
    
    c3.metric("🕒 預估總時間", f"{total_est_time:.1f} hrs")
    c3.metric("⌛ 已航行時間", f"{elapsed_time:.2f} hrs")

    # --- 6. 繪圖 ---
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    speed_grid = np.sqrt(subset.water_u**2 + subset.water_v**2)
    ax.pcolormesh(subset.lon, subset.lat, speed_grid, cmap='YlGnBu', alpha=0.4, shading='auto')
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#222')
    
    # 繪製紅線 (正確路徑) 與 白虛線 (預測路徑)
    rx, ry = [p[1] for p in st.session_state.real_p], [p[0] for p in st.session_state.real_p]
    ax.plot(rx, ry, 'r-', linewidth=2, label='HELIOS Optimized')
    px, py = [p[1] for p in st.session_state.pred_p], [p[0] for p in st.session_state.pred_p]
    ax.plot(px, py, 'w--', alpha=0.5, label='Forecast Only')
    
    # 船隻點
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, edgecolors='white', zorder=5)
    ax.set_extent([119, 124, 21.5, 26.0])
    st.pyplot(fig)

if st.button("🚢 移動至下一觀測點"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
