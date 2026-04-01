import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path

# ===============================
# 1. 頁面配置與基本參數
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Intelligent Navigation")
st.title("🛰️ HELIOS Intelligent Navigation System")

# 禁止航行區 (No-Go Zones)
NO_GO_ZONES = [
    [[22.95, 120.17], [22.93, 120.17], [22.93, 120.15], [22.95, 120.15]],
]

# 離岸風場 (Offshore Wind Farms) - 設有額外權重
OFFSHORE_WIND = [
    [[24.18, 120.12], [24.22, 120.28], [24.05, 120.35], [24.00, 120.15]],
    [[24.00, 120.10], [24.05, 120.32], [23.90, 120.38], [23.85, 120.15]],
]

# ===============================
# 2. 數據讀取 (HYCOM)
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():
    # 使用 2026 年最新數據源
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)
    
    # 擷取台灣周邊範圍
    sub = ds.sel(lat=slice(21, 26), lon=slice(118, 124))
    lons = sub.lon.values
    lats = sub.lat.values
    
    # 建立陸地遮罩 (NaN 代表陸地)
    u_vals = sub['ssu'].isel(time=-1).values
    land_mask = np.isnan(u_vals)
    
    # 計算離岸距離 (避開海岸線用)
    sea_mask = ~land_mask
    dist_to_land = distance_transform_edt(sea_mask)
    
    return sub, lons, lats, land_mask, dist_to_land

ds_sub, lons, lats, land_mask, dist_to_land = load_hycom_data()

# ===============================
# 3. 側邊欄輸入與路徑計算邏輯
# ===============================
with st.sidebar:
    st.header("📍 Route Settings")
    s_lon = st.number_input("Start Longitude", 118.0, 124.0, 120.3)
    s_lat = st.number_input("Start Latitude", 21.0, 26.0, 22.6)
    e_lon = st.number_input("End Longitude", 118.0, 124.0, 122.0)
    e_lat = st.number_input("End Latitude", 21.0, 26.0, 24.5)
    ship_speed = st.number_input("Ship Speed (km/h)", 1.0, 60.0, 20.0)
    
    if st.button("Next Step"):
        if "ship_step_idx" in st.session_state:
            st.session_state.ship_step_idx += 1

# A* 輔助函數
def get_nearest_idx(lon, lat):
    return (np.abs(lats - lat).argmin(), np.abs(lons - lon).argmin())

def astar_algorithm(start_node, goal_node):
    rows, cols = land_mask.shape
    pq = [(0, start_node)]
    came_from = {}
    cost_so_far = {start_node: 0}
    
    dirs = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    while pq:
        _, current = heapq.heappop(pq)
        if current == goal_node: break
        
        for d in dirs:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if land_mask[neighbor[0], neighbor[1]]: continue # 撞到陸地
                
                # 基本距離 + 岸邊懲罰 (避免靠岸太近)
                step_cost = np.hypot(d[0], d[1])
                if dist_to_land[neighbor[0], neighbor[1]] < 3:
                    step_cost += 10 
                
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + np.hypot(neighbor[0]-goal_node[0], neighbor[1]-goal_node[1])
                    heapq.heappush(pq, (priority, neighbor))
                    came_from[neighbor] = current
    
    path = []
    curr = goal_node
    while curr in came_from:
        path.append(curr)
        curr = came_from[curr]
    return path[::-1]

# 初始化路徑
start_idx = get_nearest_idx(s_lon, s_lat)
goal_idx = get_nearest_idx(e_lon, e_lat)
route_id = (s_lon, s_lat, e_lon, e_lat)

if "route_id" not in st.session_state or st.session_state.route_id != route_id:
    st.session_state.full_path = astar_algorithm(start_idx, goal_idx)
    st.session_state.ship_step_idx = 0
    st.session_state.route_id = route_id

# ===============================
# 4. 儀表板資訊
# ===============================
path = st.session_state.full_path
step = min(st.session_state.ship_step_idx, len(path)-1)
curr_node = path[step]

col1, col2 = st.columns([2, 1])
with col2:
    st.subheader("📊 Navigation Status")
    st.metric("Current Latitude", f"{lats[curr_node[0]]:.3f}°N")
    st.metric("Current Longitude", f"{lons[curr_node[1]]:.3f}°E")
    st.info(f"Total Waypoints: {len(path)}")

# ===============================
# 5. 地圖繪製 (核心穩定版)
# ===============================
with col1:
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118.5, 123.5, 21.5, 25.5], crs=ccrs.PlateCarree())

    # A. 繪製海流 (使用 xarray 內建 plot 最穩定)
    u = ds_sub['ssu'].isel(time=-1)
    v = ds_sub['ssv'].isel(time=-1)
    speed = np.sqrt(u**2 + v**2)
    
    # 確保座標轉置正確並繪製
    speed.plot.pcolormesh(
        ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
        cmap="YlGnBu_r", vmin=0, vmax=1.5, zorder=1,
        cbar_kwargs={'label': 'Current Speed (m/s)', 'shrink': 0.8}
    )

    # B. 加入地理特徵
    ax.add_feature(cfeature.LAND, facecolor="#e0e0e0", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=3)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)

    # C. 繪製航線
    p_lons = [lons[p[1]] for p in path]
    p_lats = [lats[p[0]] for p in path]
    ax.plot(p_lons, p_lats, color="#FF69B4", linewidth=2, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4, label="Planned")
    ax.plot(p_lons[:step+1], p_lats[:step+1], color="red", linewidth=3, transform=ccrs.PlateCarree(), zorder=5, label="Traveled")

    # D. 繪製位置點
    ax.scatter(lons[curr_node[1]], lats[curr_node[0]], color="black", marker="^", s=200, transform=ccrs.PlateCarree(), zorder=10)
    ax.scatter(s_lon, s_lat, color="purple", s=100, transform=ccrs.PlateCarree(), zorder=6)
    ax.scatter(e_lon, e_lat, color="gold", marker="*", s=300, transform=ccrs.PlateCarree(), zorder=6)

    st.pyplot(fig)
