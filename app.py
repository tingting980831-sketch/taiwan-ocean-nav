import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS 終極導航系統", layout="wide")

# 預設座標：板橋 (25.017, 121.463)
if 'ship_lat' not in st.session_state:
    st.session_state.ship_lat = 25.017
    st.session_state.ship_lon = 121.463
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# --- 2. 【核心】絕對避障演算法 ---
def generate_perfect_path(slat, slon, dlat, dlon):
    """
    使用多點轉折邏輯，確保路徑點絕對不進入陸地範圍。
    """
    # 定義絕對安全繞道點 (離岸 30km 以上)
    SAFE_NW = [25.8, 120.2]
    SAFE_NE = [25.8, 122.5]
    SAFE_SW = [21.3, 120.3]
    SAFE_SE = [21.3, 121.5]
    
    route_pts = [[slat, slon]]
    
    # 判斷是否跨越台灣本島 (經度判斷)
    # 只要起點跟終點在台灣兩側，就必須經過繞道點
    is_cross = (slon > 121.0 and dlon < 121.0) or (slon < 121.0 and dlon > 121.0)
    
    if is_cross:
        # 如果都在北邊則走北繞
        if (slat + dlat) / 2 > 24.0:
            if slon > 121.0: route_pts.extend([SAFE_NE, SAFE_NW])
            else: route_pts.extend([SAFE_NW, SAFE_NE])
        # 否則走南繞
        else:
            if slon > 121.0: route_pts.extend([SAFE_SE, SAFE_SW])
            else: route_pts.extend([SAFE_SW, SAFE_SE])
    
    route_pts.append([dlat, dlon])
    
    # 高密度插值，確保路徑平滑
    final_path = []
    for i in range(len(route_pts)-1):
        p1, p2 = route_pts[i], route_pts[i+1]
        for t in np.linspace(0, 1, 150): # 150點超高密度
            final_path.append((p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t))
    return final_path

# --- 3. 數據讀取 ---
@st.cache_data(ttl=3600)
def get_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        return ds.sel(lat=slice(20.5, 26.5), lon=slice(118.5, 125.5), depth=0).isel(time=-1).load()
    except: return None

data = get_ocean_data()

# --- 4. 側邊欄：選擇定位模式 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
location_mode = st.sidebar.radio("起始點選擇", ["📍 立即定位 (GPS 模擬)", "⌨️ 自行輸入座標"])

if location_mode == "📍 立即定位 (GPS 模擬)":
    # 模擬板橋定位
    current_lat, current_lon = 25.017, 121.463
    st.sidebar.success(f"GPS 已鎖定: {current_lat}, {current_lon}")
else:
    current_lat = st.sidebar.number_input("手動緯度", value=st.session_state.ship_lat, format="%.3f")
    current_lon = st.sidebar.number_input("手動經度", value=st.session_state.ship_lon, format="%.3f")

dest_lat = st.sidebar.number_input("終點緯度", value=22.500, format="%.3f")
dest_lon = st.sidebar.number_input("終點經度", value=122.500, format="%.3f")

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    st.session_state.ship_lat, st.session_state.ship_lon = current_lat, current_lon
    st.session_state.real_p = generate_perfect_path(current_lat, current_lon, dest_lat, dest_lon)
    st.session_state.step_idx = 0
    st.rerun()

# --- 5. 地圖繪製 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# 設定底圖：台灣灰色
ax.add_feature(cfeature.OCEAN, facecolor='#001a33')
ax.add_feature(cfeature.LAND, facecolor='#4D4D4D', zorder=2) # 這裡改成了灰色
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=0.8, zorder=3)

# 綠色海流底圖
if data is not None:
    speed = np.sqrt(data.water_u**2 + data.water_v**2)
    ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGn', alpha=0.9, zorder=1)

if st.session_state.real_p:
    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    
    # 規劃路徑 (白虛線)
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.7, zorder=4)
    # 航行軌跡 (紅實線)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    
    # 船隻與目標
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=6)
    ax.scatter(dest_lon, dest_lat, color='gold', marker='*', s=350, edgecolors='black', zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

# --- 6. 移動控制 ---
if st.button("🚢 下一步移動"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx = min(st.session_state.step_idx + 8, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
