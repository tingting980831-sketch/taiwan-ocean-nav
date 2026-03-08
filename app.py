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
import datetime

# ==================== 1️⃣ 系統配置與衛星模擬 ====================
st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧導航控制台")

# 衛星配置參數
PLANE_COUNT = 3
SATS_PER_PLANE = 4
INCLINATION = 15 # 15度傾角

def get_visible_sats():
    # 模擬 12 顆衛星在 15 度頃角下的覆蓋率
    # 台灣緯度約 23-25 度，剛好在 15 度軌道的邊緣，模擬隨機可見 2~4 顆
    return np.random.randint(2, 5)

# ==================== 2️⃣ 讀取 HYCOM 資料 ====================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        
        # 調整切片範圍：緯度只抓到 25.5，避免上方出現空白
        lat_slice, lon_slice = slice(20, 25.5), slice(118.5, 123.5)
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

# ==================== 3️⃣ A* 平衡導航邏輯 (加強避障與防繞路) ====================
def astar_v6(start, goal, u, v, land_mask, safety, ship_spd_kmh):
    v_ship = ship_spd_kmh * 0.277
    rows, cols = land_mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq = []
    heapq.heappush(pq, (0, start))
    came_from, cost = {}, {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal: break
        
        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni,nj]:
                dist_m = np.sqrt(d[0]**2 + d[1]**2) * 8000
                norm_d = np.sqrt(d[0]**2 + d[1]**2)
                flow_v = (u[cur[0], cur[1]]*(d[1]/norm_d) + v[cur[0], cur[1]]*(d[0]/norm_d))
                
                v_ground = max(0.5, v_ship + flow_v)
                time_c = dist_m / v_ground
                fuel_c = -flow_v * (dist_m / v_ship) * 1.5 
                
                # 指數級避障：離岸越近懲罰越高，強迫船隻往外海走
                safety_c = 8000 / (safety[ni,nj] + 0.3)**2 if safety[ni,nj] < 5 else 0
                
                step_c = time_c + fuel_c + safety_c
                new_total = cost[cur] + step_c
                
                if (ni,nj) not in cost or new_total < cost[(ni,nj)]:
                    cost[(ni,nj)] = new_total
                    # 強大目的地拉力 (Heuristic 3.5x)
                    priority = new_total + 3.5 * (np.sqrt((ni-goal[0])**2 + (nj-goal[1])**2) * 8000 / v_ship)
                    heapq.heappush(pq, (priority, (ni,nj)))
                    came_from[(ni,nj)] = cur
    
    path = []
    curr = goal
    while curr in came_from:
        path.append(curr); curr = came_from[curr]
    if path: path.append(start)
    return path[::-1]

# ==================== 4️⃣ 側邊欄與儀表板顯示 ====================
with st.sidebar:
    st.header("📍 導航與衛星設定")
    s_lon = st.number_input("Start Lon", value=120.30, format="%.2f")
    s_lat = st.number_input("Start Lat", value=22.60, format="%.2f")
    e_lon = st.number_input("Goal Lon", value=121.90, format="%.2f")
    e_lat = st.number_input("Goal Lat", value=24.50, format="%.2f")
    ship_spd = st.number_input("🚤 船速 (km/h)", value=25.0)
    st.markdown(f"---")
    st.write(f"🛰️ **衛星軌道面:** {PLANE_COUNT}")
    st.write(f"🛰️ **單面衛星數:** {SATS_PER_PLANE}")
    st.write(f"📐 **軌道頃角:** {INCLINATION}°")

if lons is not None:
    # 計算
    safety = distance_transform_edt(~land_mask)
    start_idx = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask) if 'nearest_ocean_cell' in globals() else (0,0) # 簡化
    # 這裡重複使用之前的 nearest_ocean_cell 函數
    
    # 計算路徑
    path = astar_v6(start_idx, nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask), u, v, land_mask, safety, ship_spd)

    # --- 置頂儀表板 (含衛星資訊) ---
    c1, c2, c3, c4 = st.columns(4)
    if path:
        total_dist = sum(np.sqrt((lats[path[i][0]]-lats[path[i+1][0]])**2 + (lons[path[i][1]]-lons[path[i+1][1]])**2) for i in range(len(path)-1)) * 111
        c1.metric("⏱️ 預估航時", f"{total_dist/ship_spd:.1f} Hours")
        c2.metric("📏 總距離", f"{total_dist:.1f} km")
    
    c3.metric("🛰️ 當前覆蓋衛星", f"{get_visible_sats()} SATS", delta="衛星鏈路正常")
    c4.metric("📅 數據同步時間", obs_time.strftime("%m/%d %H:%M"))

    st.divider()

    # ==================== 5️⃣ 繪圖 (去除上方空白區域) ====================
    fig = plt.figure(figsize=(11, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 修正顯示邊界：將北緯上限設定為 25.5，確保不露白
    ax.set_extent([118.8, 123.2, 21.2, 25.4], crs=ccrs.PlateCarree())

    # 背景
    ax.add_feature(cfeature.LAND, facecolor='#2C2C2C', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.6, zorder=3)
    
    speed = np.sqrt(u**2 + v**2)
    colors = ["#000E24", "#003D99", "#3385FF", "#99C2FF", "#E5F0FF"]
    cmap = mcolors.LinearSegmentedColormap.from_list("ocean", colors)
    
    im = ax.pcolormesh(lons, lats, speed, cmap=cmap, shading='auto', alpha=0.9, zorder=1)
    plt.colorbar(im, label='Speed (m/s)', shrink=0.6)

    # 路徑與標記
    if path:
        ax.plot([lons[p[1]] for p in path], [lats[p[0]] for p in path], color='#FF00FF', linewidth=2.5, label='HELIOS AI Path', zorder=5)
    
    ax.scatter(s_lon, s_lat, color='lime', s=100, edgecolors='black', label='Start', zorder=6)
    ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, edgecolors='black', label='Goal', zorder=6)

    ax.legend(loc='upper left', fontsize='small')
    ax.gridlines(draw_labels=True, alpha=0.1, color='white')
    plt.title(f"HELIOS V6 AI Marine Navigation - LEO Orbit {INCLINATION}° Inclination", fontsize=12)

    st.pyplot(fig)
