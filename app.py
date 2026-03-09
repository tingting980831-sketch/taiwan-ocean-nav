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
import plotly.graph_objects as go
import requests
from datetime import datetime

# 設定頁面
st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("🛰️ HELIOS V7 智慧海象導航系統")

# ===============================
# 模擬可視衛星
# ===============================
def get_visible_sats():
    return np.random.randint(8,12)

# ===============================
# 讀取 HYCOM 海流數據
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():
    # 注意：此 URL 需根據 HYCOM 實際存取路徑調整
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        lat_slice = slice(21,26)
        lon_slice = slice(118,124)
        u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        lons = u_data['lon'].values
        lats = u_data['lat'].values
        u_val = np.nan_to_num(u_data.values)
        v_val = np.nan_to_num(v_data.values)
        land_mask = np.isnan(u_data.values)
        return lons, lats, u_val, v_val, land_mask, latest_time
    except Exception as e:
        st.error(f"無法讀取海流數據: {e}")
        return None, None, None, None, None, None

lons, lats, u, v, land_mask, obs_time = load_hycom_data()

# ===============================
# 核心修正：同步時間的波高與風速 API
# ===============================
@st.cache_data(ttl=1800)
def get_realtime_marine_data(lat, lon):
    current_hour = datetime.now().hour  # 獲取當前小時 (0-23)
    
    # 1. 取得波高 (Marine API)
    marine_url = "https://marine-api.open-meteo.com/v1/marine"
    marine_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "wave_height",
        "timezone": "Asia/Taipei",
        "forecast_days": 1
    }
    wave = None
    try:
        res = requests.get(marine_url, params=marine_params, timeout=5).json()
        waves = res.get("hourly", {}).get("wave_height", [])
        if len(waves) > current_hour and waves[current_hour] is not None:
            wave = float(waves[current_hour])
        else:
            valid_waves = [w for w in waves if w is not None]
            if valid_waves: wave = float(valid_waves[0])
    except:
        wave = None

    # 2. 取得風速 (Weather API)
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["wind_speed_10m","wind_direction_10m"],
        "timezone": "Asia/Taipei",
        "forecast_days": 1
    }
    wind_speed, wind_dir = None, None
    try:
        w_res = requests.get(weather_url, params=weather_params, timeout=5).json()
        speeds = w_res.get("hourly", {}).get("wind_speed_10m", [])
        dirs = w_res.get("hourly", {}).get("wind_direction_10m", [])
        if len(speeds) > current_hour and speeds[current_hour] is not None:
            wind_speed = float(speeds[current_hour])
            wind_dir = float(dirs[current_hour])
        else:
            valid_s = [s for s in speeds if s is not None]
            valid_d = [d for d in dirs if d is not None]
            if valid_s: wind_speed = float(valid_s[0])
            if valid_d: wind_dir = float(valid_d[0])
    except:
        pass

    return wave, wind_speed, wind_dir

# ===============================
# 最近海洋格點尋找
# ===============================
def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons-lon).argmin()
    lat_idx = np.abs(lats-lat).argmin()
    if not land_mask[lat_idx, lon_idx]:
        return lat_idx, lon_idx
    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]]-lat)**2 + (lons[ocean[1]]-lon)**2)
    i = dist.argmin()
    return ocean[0][i], ocean[1][i]

# ===============================
# A* 路徑優化算法
# ===============================
def astar(start, goal, u, v, land_mask, safety, ship_spd, wave_factor=0, wind_factor=0):
    v_ship = ship_spd * 0.277  # km/h -> m/s
    rows, cols = land_mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq = [(0, start)]
    cost = {start: 0}
    came = {}
    
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal: break
        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni, nj]:
                dist_m = np.sqrt(d[0]**2 + d[1]**2) * 8000 # 假設格點間距 8km
                norm = np.sqrt(d[0]**2 + d[1]**2)
                flow = (u[cur]*(d[1]/norm) + v[cur]*(d[0]/norm))
                v_ground = max(0.5, v_ship + flow)
                
                step = (dist_m / v_ground) + (-flow * (dist_m / v_ship) * 1.5)
                # 加入波高與風速對成本的影響
                step += (wave_factor**2) * 60 + wind_factor * 40
                
                if safety[ni, nj] < 4:
                    step += 12000 / (safety[ni, nj] + 0.2)**2
                
                new_cost = cost[cur] + step
                if (ni, nj) not in cost or new_cost < cost[(ni, nj)]:
                    cost[(ni, nj)] = new_cost
                    priority = new_cost + 4 * np.sqrt((ni-goal[0])**2 + (nj-goal[1])**2) * 8000 / v_ship
                    heapq.heappush(pq, (priority, (ni, nj)))
                    came[(ni, nj)] = cur
    path = []
    curr = goal
    while curr in came:
        path.append(curr)
        curr = came[curr]
    return path[::-1] + [start] if path else []

# ===============================
# 側邊欄與導航設定
# ===============================
with st.sidebar:
    st.header("🚢 航點設定")
    s_lon = st.number_input("起點經度", 118.0, 124.0, 120.3)
    s_lat = st.number_input("起點緯度", 21.0, 26.0, 22.6)
    e_lon = st.number_input("終點經度", 118.0, 124.0, 122.0)
    e_lat = st.number_input("終點緯度", 21.0, 26.0, 24.5)
    ship_speed = st.number_input("船速 (km/h)", 1.0, 60.0, 20.0)

# ===============================
# 主計算邏輯
# ===============================
if lons is not None:
    safety = distance_transform_edt(~land_mask)
    start = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
    goal  = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)

    # 抓取海域中心點的即時數據
    wave, wind_speed, wind_dir = get_realtime_marine_data((s_lat+e_lat)/2, (s_lon+e_lon)/2)

    path = astar(start, goal, u, v, land_mask, safety, ship_speed, 
                 wave_factor=wave if wave else 0, 
                 wind_factor=wind_speed if wind_speed else 0)

    # 儀表板
    c1, c2, c3 = st.columns(3)
    if path:
        dist_km = sum(np.sqrt((lats[path[i][0]]-lats[path[i+1][0]])**2 + 
                              (lons[path[i][1]]-lons[path[i+1][1]])**2) 
                      for i in range(len(path)-1)) * 111
        c1.metric("預估航時", f"{dist_km/ship_speed:.1f} hr")
        c2.metric("總航程", f"{dist_km:.1f} km")
    c3.metric("波高 / 風速", f"{wave if wave else '--'}m | {wind_speed if wind_speed else '--'}kph")

    st.caption(f"📅 HYCOM 資料時間: {obs_time} | 🌊 波高狀態: {'已對齊時間' if wave else '自動找尋格點中'}")

    # ===============================
    # 2D 地圖繪製
    # ===============================
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118, 124, 21, 26])
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    
    total_factor = np.sqrt(u**2 + v**2) + (wave if wave else 0)
    im = ax.pcolormesh(lons, lats, total_factor, cmap="YlGnBu_r", alpha=0.8)
    plt.colorbar(im, ax=ax, label="海象影響強度")
    
    if path:
        path_lons = [lons[p[1]] for p in path]
        path_lats = [lats[p[0]] for p in path]
        ax.plot(path_lons, path_lats, color='red', linewidth=2.5, label="最優航線")
    
    ax.scatter(s_lon, s_lat, color='green', s=100, label="Start")
    ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, label="Goal")
    plt.title("HELIOS V7 Real-time Marine Navigation")
    st.pyplot(fig)

    # ===============================
    # 3D 海象模型 (Plotly)
    # ===============================
    st.subheader("🌊 3D 深度動態海象")
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flow_speed = np.sqrt(u**2 + v**2)
    fig3d = go.Figure(data=[go.Surface(x=lon_grid, y=lat_grid, z=flow_speed, colorscale="Blues")])
    fig3d.update_layout(height=600, scene=dict(xaxis_title="經度", yaxis_title="緯度", zaxis_title="流速"))
    st.plotly_chart(fig3d, use_container_width=True)
