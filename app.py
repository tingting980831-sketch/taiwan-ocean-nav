import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 定位與初始化 ---
st.set_page_config(page_title="HELIOS 專業導航系統", layout="wide")

# 自動帶入你的目前位置 (抓取自系統定位)
MY_LAT, MY_LON = 25.017, 121.463

if 'ship_lat' not in st.session_state:
    st.session_state.ship_lat = MY_LAT
    st.session_state.ship_lon = MY_LON
if 'path' not in st.session_state: st.session_state.path = []
if 'idx' not in st.session_state: st.session_state.idx = 0

# --- 2. 【核心】絕對避障演算法 ---
def calculate_safe_path(slat, slon, dlat, dlon):
    # 台灣陸地禁航邊界 (加寬緩衝區)
    BUFFER = 0.2
    LAT_RANGE = [21.8 - BUFFER, 25.4 + BUFFER]
    LON_RANGE = [120.0 - BUFFER, 122.1 + BUFFER]
    
    # 安全轉折點 (外海深水區)
    WAYPOINTS = {
        'NORTH': [26.0, 121.0], # 北方外海
        'SOUTH': [21.0, 121.0], # 南方外海
        'EAST': [23.5, 123.0],  # 東部外海
        'WEST': [23.5, 119.5]   # 西部外海
    }

    # 判斷是否跨越陸地 (檢查連線是否穿過禁航矩形)
    # 簡單邏輯：若起終點分別在東西兩側，則必須繞道
    cross_east_west = (slon < 121.0 and dlon > 121.0) or (slon > 121.0 and dlon < 121.0)
    cross_north_south = (slat < 23.5 and dlat > 23.5) or (slat > 23.5 and dlat < 23.5)

    route = [[slat, slon]]
    
    if cross_east_west:
        # 如果都在北邊則走北繞，否則走南繞
        if slat > 23.8 or dlat > 23.8:
            route.append(WAYPOINTS['NORTH'])
        else:
            route.append(WAYPOINTS['SOUTH'])
    
    route.append([dlat, dlon])
    
    # 線性插值生成平滑路徑 (每段 50 點，確保不亂跳)
    final_p = []
    for i in range(len(route)-1):
        p1, p2 = route[i], route[i+1]
        for t in np.linspace(0, 1, 50):
            final_p.append((p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t))
    return final_p

# --- 3. 介面設計 ---
st.title("🛰️ HELIOS 智慧避障導航")
st.sidebar.markdown(f"📍 **目前定位**: `{MY_LAT}, {MY_LON}`")

with st.sidebar:
    dest_lat = st.number_input("目標緯度", value=22.500, format="%.3f")
    dest_lon = st.number_input("目標經度", value=122.500, format="%.3f")
    
    if st.button("🗺️ 計算路徑", use_container_width=True):
        st.session_state.path = calculate_safe_path(st.session_state.ship_lat, st.session_state.ship_lon, dest_lat, dest_lon)
        st.session_state.idx = 0
        st.rerun()

# --- 4. 地圖繪製 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#1a1a1a', zorder=2)
ax.add_feature(cfeature.OCEAN, facecolor='#001219', zorder=1)
ax.add_feature(cfeature.COASTLINE, edgecolor='lime', linewidth=1, zorder=3)

if st.session_state.path:
    path_lats = [p[0] for p in st.session_state.path]
    path_lons = [p[1] for p in st.session_state.path]
    
    # 畫出完整預計航線
    ax.plot(path_lons, path_lats, 'gray', linestyle='--', alpha=0.5, zorder=4)
    # 畫出當前船位與已走航路
    ax.plot(path_lons[:st.session_state.idx+1], path_lats[:st.session_state.idx+1], 'red', linewidth=3, zorder=5)
    
    # 標記起終點
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=6, label="船隻位置")
    ax.scatter(dest_lon, dest_lat, color='gold', marker='*', s=200, zorder=6, label="目的地")

ax.set_extent([118, 125, 20, 27])
st.pyplot(fig)

# --- 5. 移動控制 ---
if st.session_state.path:
    if st.button("🚢 推進航程"):
        if st.session_state.idx < len(st.session_state.path) - 1:
            st.session_state.idx = min(st.session_state.idx + 5, len(st.session_state.path) - 1)
            st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.path[st.session_state.idx]
            st.rerun()
