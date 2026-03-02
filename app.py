import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 初始化設定 ---
st.set_page_config(page_title="HELIOS 專業導航 V6", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.017
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.463
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# --- 2. 陸地檢查器 (Land Checker) ---
def is_on_land(lat, lon):
    # 台灣本島的粗略禁航矩形 (包含緩衝區)
    # 經度 120.0 ~ 122.0, 緯度 21.8 ~ 25.4
    if 21.8 <= lat <= 25.4 and 120.0 <= lon <= 122.0:
        return True
    return False

# --- 3. 固定儀表板 ---
st.title("🛰️ HELIOS 智慧導航 (陸地禁入版)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速", "16.2 kn")
c2.metric("⛽ 能源紅利", "24.8%")
dist_val = f"{len(st.session_state.real_p)*0.1:.1f} nmi" if st.session_state.real_p else "0.0 nmi"
c3.metric("📏 總航程", dist_val)
c4.metric("🕒 預估時間", "4.5 hrs")

# --- 4. 避障核心 (弧形繞道) ---
def generate_v6_path(slat, slon, dlat, dlon):
    SAFE_NW = [26.2, 119.8]
    SAFE_N  = [26.2, 121.5]
    SAFE_NE = [26.2, 123.0]
    SAFE_SW = [21.0, 119.8]
    SAFE_S  = [21.0, 121.0]
    SAFE_SE = [21.0, 122.5]
    
    route_pts = [[slat, slon]]
    is_cross = (slon > 121.1 and dlon < 120.9) or (slon < 120.9 and dlon > 121.1)
    
    if is_cross:
        if (slat + dlat) / 2 > 23.5:
            route_pts.extend([SAFE_NE, SAFE_N, SAFE_NW] if slon > 121.1 else [SAFE_NW, SAFE_N, SAFE_NE])
        else:
            route_pts.extend([SAFE_SE, SAFE_S, SAFE_SW] if slon > 121.1 else [SAFE_SW, SAFE_S, SAFE_SE])
            
    route_pts.append([dlat, dlon])
    
    final_p = []
    for i in range(len(route_pts)-1):
        p1, p2 = route_pts[i], route_pts[i+1]
        for t in np.linspace(0, 1, 100):
            final_p.append((p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t))
    return final_p

# --- 5. 側邊欄與陸地攔截邏輯 ---
st.sidebar.header("🧭 導航控制中心")
mode = st.sidebar.radio("起始點選擇", ["📍 立即定位 (GPS 模擬)", "⌨️ 自行輸入座標"])

if mode == "📍 立即定位 (GPS 模擬)":
    curr_lat, curr_lon = 25.017, 121.463 # 板橋
else:
    curr_lat = st.sidebar.number_input("起始緯度", value=st.session_state.ship_lat, format="%.3f")
    curr_lon = st.sidebar.number_input("起始經度", value=st.session_state.ship_lon, format="%.3f")

d_lat = st.sidebar.number_input("終點緯度", value=22.500, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=122.500, format="%.3f")

# 🚩 關鍵檢查：如果座標在陸地，直接禁用按鈕並顯示警告
start_on_land = is_on_land(curr_lat, curr_lon)
end_on_land = is_on_land(d_lat, d_lon)

if start_on_land:
    st.sidebar.error("❌ 警告：起點位於陸地！無法導航。")
if end_on_land:
    st.sidebar.error("❌ 警告：終點位於陸地！請重新選擇。")

if st.sidebar.button("🚀 執行 AI 路徑分析", disabled=(start_on_land or end_on_land)):
    st.session_state.ship_lat, st.session_state.ship_lon = curr_lat, curr_lon
    st.session_state.real_p = generate_v6_path(curr_lat, curr_lon, d_lat, d_lon)
    st.session_state.step_idx = 0
    st.rerun()

# --- 6. 地圖繪製 (灰色台灣) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.OCEAN, facecolor='#001a33')
ax.add_feature(cfeature.LAND, facecolor='#555555', zorder=2) # 台灣灰色
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

# 經典綠色底圖 (HYCOM 模擬)
lon = np.linspace(118, 126, 50)
lat = np.linspace(20, 27, 50)
speed = np.random.rand(50, 50)
ax.pcolormesh(lon, lat, speed, cmap='YlGn', alpha=0.3, zorder=1)

if st.session_state.real_p:
    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    ax.plot(px, py, color='white', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=6)
    ax.scatter(d_lon, d_lat, color='gold', marker='*', s=350, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

if st.button("🚢 下一步移動", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx = min(st.session_state.step_idx + 10, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
