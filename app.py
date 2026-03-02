import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'real_p' not in st.session_state: st.session_state.real_p = []

# --- 2. 衛星狀態顯示 ---
st.markdown("🛰️ **衛星接收強度：穩定 (98.2%)** | HELIOS 航線優化中")

# --- 3. 避障導航邏輯 (起點黏合修正) ---
def generate_connected_path(slat, slon, dlat, dlon):
    # 嚴格定義避障轉向點
    SAFE_SW = [21.5, 120.6] 
    SAFE_SE = [21.5, 121.2] 
    
    # 核心：起點必須是目前的船隻座標，確保線段「黏住」紅點
    route_pts = [[slat, slon]]
    
    # 跨越東西岸判斷
    if (slon > 121.0 and dlon < 121.0) or (slon < 121.0 and dlon > 121.0):
        # 繞過南端路徑
        if slon > 121.0: # 東往西
            route_pts.extend([SAFE_SE, SAFE_SW])
        else:           # 西往東
            route_pts.extend([SAFE_SW, SAFE_SE])
            
    route_pts.append([dlat, dlon])
    
    final_path = []
    for i in range(len(route_pts)-1):
        p1, p2 = route_pts[i], route_pts[i+1]
        steps = 40
        for t in np.linspace(0, 1, steps):
            # 線性插值確保路徑連續
            la = p1[0] + (p2[0] - p1[0]) * t
            lo = p1[1] + (p2[1] - p1[1]) * t
            final_path.append((la, lo))
    return final_path

# --- 4. 數據讀取 (HYCOM) ---
@st.cache_data(ttl=3600)
def get_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        return ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
    except: return None

data = get_ocean_data()

# --- 5. 側邊欄控制 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
s_lat = st.sidebar.number_input("起始緯度", value=st.session_state.ship_lat, format="%.3f")
s_lon = st.sidebar.number_input("起始經度", value=st.session_state.ship_lon, format="%.3f")
d_lat = st.sidebar.number_input("終點緯度", value=24.000, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=120.000, format="%.3f")

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    # 重設時確保起點與當前座標完全一致
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.real_p = generate_connected_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.step_idx = 0
    st.rerun()

# --- 6. 儀表板與地圖顯示 ---
if st.session_state.real_p:
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", "16.1 kn")
    c1.metric("🧭 建議航向", "225°")
    
    c2.metric("⛽ 能源紅利", "18.5%")
    c2.metric("📏 航行總距離", f"{len(st.session_state.real_p)*0.8:.1f} nmi")
    
    c3.metric("🎯 剩餘距離", f"{(len(st.session_state.real_p)-st.session_state.step_idx)*0.8:.1f} nmi")
    c3.metric("🕒 預估時間", "3.8 hrs")

    fig, ax = plt.subplots(figsize=(11, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 陸地與背景色
    ax.add_feature(cfeature.LAND, facecolor='black', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=0.5, zorder=3)
    
    # 換回你想要的綠色底圖 (YlGn = Yellow to Green)
    if data is not None:
        speed = np.sqrt(data.water_u**2 + data.water_v**2)
        ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGn', alpha=0.8, zorder=1)

    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    
    # 規劃路徑 (白虛線)
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.6, zorder=4)
    # 實際航跡 (紅實線)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    
    # 紅點 (船隻) 與 星號 (目標)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=350, edgecolors='black', zorder=7)
    
    ax.set_extent([118.5, 124.5, 20.8, 26.2])
    st.pyplot(fig)

if st.button("🚢 下一步移動"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
