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
import requests
from datetime import datetime, timezone
from matplotlib.patches import Polygon as MPolygon
from matplotlib.path import Path

# 設定頁面
st.set_page_config(layout="wide", page_title="HELIOS V7 - Grey Path")
st.title("🛰️ HELIOS V7 智慧海象導航系統 (灰色航線版)")

# ===============================
# 禁航區座標資料庫 (十進位 [Lat, Lon])
# ===============================
NO_GO_ZONES = [
    # 台南安平/四草區域
    [[22.953536, 120.171678], [22.934628, 120.175472], [22.933136, 120.170942], [22.957810, 120.160780]],
    [[22.943956, 120.172358], [22.939717, 120.173944], [22.928353, 120.157372], [22.936636, 120.153547]],
    [[22.88462, 120.17547], [22.91242, 120.17696], [22.91178, 120.17214], [22.933136, 120.170942]],
    # 屏東東港區域 (由度分秒 A1-A4 轉換)
    [[22.360902, 120.381109], [22.354722, 120.382139], [22.353191, 120.384728], [22.357684, 120.387818]],
    [[22.452778, 120.458611], [22.4475, 120.451389], [22.446111, 120.452222], [22.449167, 120.460556]],
    # 澎湖海域
    [[23.788500, 119.598368], [23.784251, 119.598368], [23.784251, 119.602022], [23.788500, 119.602022]],
    [[23.280833, 119.500000], [23.280833, 119.509722], [23.274444, 119.509722], [23.274444, 119.500000]],
    # 野柳/基隆區域
    [[25.231417, 121.648863], [25.226151, 121.651505], [25.233410, 121.642090], [25.242200, 121.634560]],
    # 彰化外海風場集群 (北)
    [[24.18, 120.12], [24.22, 120.28], [24.05, 120.35], [24.00, 120.15]],
    # 彰化外海風場集群 (中)
    [[24.00, 120.10], [24.05, 120.32], [23.90, 120.38], [23.85, 120.15]],
    # 新竹/苗栗區域
    [[24.849621, 120.928948], [24.848034, 120.929797], [24.847862, 120.930568], [24.850101, 120.930086]],
    [[24.831074, 120.914995], [24.822774, 120.909618], [24.818237, 120.907696], [24.822791, 120.909332]],
]

# ===============================
# 資料讀取與處理
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        lat_slice, lon_slice = slice(21,26), slice(118,124)
        u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        lons, lats = u_data['lon'].values, u_data['lat'].values
        return lons, lats, np.nan_to_num(u_data.values), np.nan_to_num(v_data.values), np.isnan(u_data.values), latest_time
    except:
        st.error("無法載入 HYCOM 資料")
        return None, None, None, None, None, None

lons, lats, u, v, land_mask, obs_time = load_hycom_data()

def apply_no_go_mask(lons, lats, base_mask, zones):
    new_mask = base_mask.copy()
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
    for zone in zones:
        if len(zone) < 3: continue
        poly_points = [[p[1], p[0]] for p in zone] # (Lon, Lat)
        path = Path(poly_points)
        grid_mask = path.contains_points(points).reshape(lon_grid.shape)
        new_mask = np.logical_or(new_mask, grid_mask)
    return new_mask

# ===============================
# A* 算法
# ===============================
def astar(start, goal, u, v, mask, safety, ship_spd):
    v_ship = ship_spd * 0.277 # km/h -> m/s
    rows, cols = mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq = [(0, start)]
    cost, came = {start: 0}, {}
    
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal: break
        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if 0<=ni<rows and 0<=nj<cols and not mask[ni,nj]:
                dist_m = np.sqrt(d[0]**2+d[1]**2)*8000 # 預設格點 8km
                
                # 計算流速影響 (順流/逆流)
                norm = np.sqrt(d[0]**2+d[1]**2)
                flow = (u[cur]*(d[1]/norm) + v[cur]*(d[0]/norm))
                
                # 地面速度 (最慢 0.5 m/s 防止倒退)
                v_ground = max(0.5, v_ship + flow)
                
                # 時間成本
                step_cost = (dist_m/v_ground)
                
                # 安全成本 (靠近阻礙物會增加成本)
                dist_danger = safety[ni, nj]
                if dist_danger < 3:
                    step_cost += 15000 / (dist_danger + 0.1)**2
                
                new_cost = cost[cur] + step_cost
                
                if (ni,nj) not in cost or new_cost < cost[(ni,nj)]:
                    cost[(ni,nj)] = new_cost
                    # 啟發式函數 (Heuristic): 距離目標的剩餘時間
                    dist_to_goal = np.sqrt((ni-goal[0])**2 + (nj-goal[1])**2) * 8000
                    priority = new_cost + 1.2 * (dist_to_goal / v_ship)
                    heapq.heappush(pq, (priority, (ni,nj)))
                    came[(ni,nj)] = cur
    path = []
    curr = goal
    while curr in came:
        path.append(curr)
        curr = came[curr]
    return path[::-1] + ([start] if goal in came else [])

# ===============================
# UI 與 繪圖
# ===============================
if lons is not None:
    # 應用禁航區遮罩
    final_mask = apply_no_go_mask(lons, lats, land_mask, NO_GO_ZONES)
    # 計算安全距離矩陣
    safety = distance_transform_edt(~final_mask)

    with st.sidebar:
        st.header("航點設定")
        s_lon = st.number_input("起點經度", 118.0, 124.0, 120.3)
        s_lat = st.number_input("起點緯度", 21.0, 26.0, 22.8)
        e_lon = st.number_input("終點經度", 118.0, 124.0, 122.5)
        e_lat = st.number_input("終點緯度", 21.0, 26.0, 25.0)
        ship_speed = st.number_input("船速 (km/h)", 5.0, 60.0, 25.0)

    # 尋找最近格點
    start_grid = (np.abs(lats-s_lat).argmin(), np.abs(lons-s_lon).argmin())
    goal_grid = (np.abs(lats-e_lat).argmin(), np.abs(lons-e_lon).argmin())
    
    # 計算航線
    path = astar(start_grid, goal_grid, u, v, final_mask, safety, ship_speed)

    # 地圖渲染
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118, 124, 21, 26])
    
    # 配色調整：深灰色陸地
    ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, color='white', linewidth=0.5)

    # 背景：海流強度 (藍色系)
    mag = np.sqrt(u**2 + v**2)
    im = ax.pcolormesh(lons, lats, mag, cmap="Blues_r", alpha=0.5, shading='gouraud')
    
    # 繪製禁航區 (紅色)
    for zone in NO_GO_ZONES:
        poly = MPolygon([[p[1], p[0]] for p in zone], closed=True, 
                        edgecolor='red', facecolor='red', alpha=0.5, linewidth=2)
        ax.add_patch(poly)

    # --- 修正點：航線改為灰色 ---
    if path:
        path_lons = [lons[p[1]] for p in path]
        path_lats = [lats[p[0]] for p in path]
        ax.plot(path_lons, path_lats, color='#A9A9A9', linewidth=4) # DarkGray
    
    # 標記起點與終點
    ax.scatter([s_lon, e_lon], [s_lat, e_lat], color=['lime', 'gold'], s=120, edgecolors='black', zorder=10)
    
    plt.title(f"HELIOS V7: Navigation (Speed: {ship_speed} km/h)", color='white', fontsize=16)
    fig.patch.set_facecolor('#1e1e1e') # Streamlit 暗色背景
    ax.set_facecolor('#1e1e1e')
    
    # --- 修正點：圖例拿掉 (移除 plt.legend()) ---

    st.pyplot(fig)
