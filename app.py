import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

# ==================== 1️⃣ 系統配置 ====================
st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧導航控制台")

# ==================== 2️⃣ 讀取 HYCOM 資料 ====================
@st.cache_data(ttl=3600)
def load_hycom_data():
    # 使用你確認可跑的 URL 與 變數名稱
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        
        # 台灣海域切片
        lat_slice, lon_slice = slice(20, 26), slice(118, 124)
        u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        
        lons = u_data['lon'].values
        lats = u_data['lat'].values
        u_val = np.nan_to_num(u_data.values)
        v_val = np.nan_to_num(v_data.values)
        land_mask = np.isnan(u_data.values)
        
        return lons, lats, u_val, v_val, land_mask, latest_time
    except Exception as e:
        st.error(f"📡 衛星連線異常: {e}")
        return None, None, None, None, None, None

lons, lats, u, v, land_mask, obs_time = load_hycom_data()

# ==================== 3️⃣ 導航邏輯與工具 ====================
def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()
    if not land_mask[lat_idx, lon_idx]:
        return (lat_idx, lon_idx)
    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]] - lat)**2 + (lons[ocean[1]] - lon)**2)
    i = dist.argmin()
    return (ocean[0][i], ocean[1][i])

def compute_safety(land_mask):
    return distance_transform_edt(~land_mask)

def astar_balanced(start, goal, u, v, land_mask, safety, ship_speed_kmh):
    v_ship = ship_speed_kmh * 0.277 # 轉為 m/s
    rows, cols = land_mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq = []
    heapq.heappush(pq, (0, start))
    came_from = {}
    total_cost = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal: break
        
        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni,nj]:
                # 距離計算 (假設每格約 8km)
                dist_m = np.sqrt(d[0]**2 + d[1]**2) * 8000 
                
                # 海流投影分量
                norm_d = np.sqrt(d[0]**2 + d[1]**2)
                flow_v = (u[cur[0], cur[1]] * (d[1]/norm_d) + v[cur[0], cur[1]] * (d[0]/norm_d))
                
                # 省時權重：對地速度計算
                v_ground = v_ship + flow_v
                if v_ground <= 0.5: v_ground = 0.5
                time_cost = dist_m / v_ground
                
                # 省油權重：海流功耗補償 (1:1 平衡係數)
                fuel_cost = -flow_v * (dist_m / v_ship) * 1.5 
                
                # 避障權重
                land_penalty = 800 / (safety[ni,nj] + 1)
                
                step_cost = time_cost + fuel_cost + land_penalty
                new_total = total_cost[cur] + step_cost
                
                if (ni,nj) not in total_cost or new_total < total_cost[(ni,nj)]:
                    total_cost[(ni,nj)] = new_total
                    # 啟發式：預估剩餘時間，並加強拉力防止繞路
                    h_dist = np.sqrt((ni-goal[0])**2 + (nj-goal[1])**2) * 8000
                    priority = new_total + 3.0 * (h_dist / v_ship)
                    heapq.heappush(pq, (priority, (ni,nj)))
                    came_from[(ni,nj)] = cur
    
    path = []
    curr = goal
    while curr in came_from:
        path.append(curr)
        curr = came_from[curr]
    if path: path.append(start)
    return path[::-1]

# ==================== 4️⃣ 側邊欄輸入 ====================
with st.sidebar:
    st.header("📍 導航設定")
    s_lon = st.number_input("起點經度 (Start Lon)", value=120.30, format="%.2f")
    s_lat = st.number_input("起點緯度 (Start Lat)", value=22.60, format="%.2f")
    e_lon = st.number_input("終點經度 (Goal Lon)", value=122.00, format="%.2f")
    e_lat = st.number_input("終點緯度 (Goal Lat)", value=24.50, format="%.2f")
    ship_spd = st.number_input("🚤 船速 (km/h)", value=25.0, step=1.0)
    run_nav = st.button("🚀 重新計算航線", use_container_width=True)

# ==================== 5️⃣ 執行計算與儀表板顯示 ====================
if lons is not None:
    safety = compute_safety(land_mask)
    start_idx = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
    goal_idx = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)
    
    path = astar_balanced(start_idx, goal_idx, u, v, land_mask, safety, ship_spd)

    # --- 儀表板置頂 ---
    if path:
        # 計算實際航行距離
        total_dist_km = 0
        for i in range(len(path)-1):
            p1 = (lats[path[i][0]], lons[path[i][1]])
            p2 = (lats[path[i+1][0]], lons[path[i+1][1]])
            total_dist_km += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 111
        est_hours = total_dist_km / ship_spd
    else:
        total_dist_km, est_hours = 0, 0

    col1, col2, col3 = st.columns(3)
    col1.metric("⏱️ 預估航行時間", f"{est_hours:.1f} Hours")
    col2.metric("📏 總航行距離", f"{total_dist_km:.1f} km")
    col3.metric("📅 衛星數據時間", obs_time.strftime("%m/%d %H:%M"))

    st.divider()

    # ==================== 6️⃣ 地圖視覺化 ====================
    # 自訂海流顏色
    colors = ["#E5F0FF","#99C2FF","#3385FF","#0052CC","#001433"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("flow", colors)

    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118.5, 123.5, 21.0, 26.5], crs=ccrs.PlateCarree())

    # 灰色陸地設定
    ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.8, zorder=3)
    
    # 海流背景
    speed = np.sqrt(u**2 + v**2)
    im = ax.pcolormesh(lons, lats, speed, cmap=custom_cmap, shading='auto', alpha=0.8, zorder=1)
    plt.colorbar(im, label='Current Speed (m/s)', shrink=0.5)

    # 航線繪製 (洋紅色)
    if path:
        p_lons = [lons[p[1]] for p in path]
        p_lats = [lats[p[0]] for p in path]
        ax.plot(p_lons, p_lats, color='#FF00FF', linewidth=3, label='AI Balanced Path', zorder=5)

    # 標記起點終點
    ax.scatter(s_lon, s_lat, color='lime', s=120, edgecolors='black', label='Start', zorder=6)
    ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=250, edgecolors='black', label='Goal', zorder=6)

    ax.legend(loc='upper left')
    ax.gridlines(draw_labels=True, alpha=0.1)
    plt.title(f"HELIOS V6 AI Navigation (Balanced Mode)", fontsize=14)

    st.pyplot(fig)
