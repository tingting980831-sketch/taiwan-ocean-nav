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

# ==================== 1️⃣ 衛星與系統設定 ====================
st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧導航控制台")

# ==================== 2️⃣ 側邊欄輸入 ====================
with st.sidebar:
    st.header("📍 航點座標輸入")
    st.info("請輸入經緯度，系統將自動定位至最近海域。")
    s_lon = st.number_input("起點經度 (118-124)", value=120.30, format="%.2f")
    s_lat = st.number_input("起點緯度 (20-26)", value=22.60, format="%.2f")
    e_lon = st.number_input("終點經度 (118-124)", value=122.00, format="%.2f")
    e_lat = st.number_input("終點緯度 (20-26)", value=24.50, format="%.2f")
    ship_speed = st.number_input("🚤 船速 (km/h)", value=20.0, step=1.0)
    run_nav = st.button("🚀 啟動衛星導航計算", use_container_width=True)

# ==================== 3️⃣ 讀取 HYCOM 資料 ====================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)
    time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
    latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
    lat_slice, lon_slice = slice(20, 26), slice(118, 124)
    u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
    v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
    land_mask = np.isnan(u_data.values)
    return u_data['lon'].values, u_data['lat'].values, u_data.values, v_data.values, land_mask, latest_time

lons, lats, u, v, land_mask, obs_time = load_hycom_data()

# ==================== 4️⃣ 最近海格點與安全距離 ====================
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
    ocean = ~land_mask
    return distance_transform_edt(ocean)

safety = compute_safety(land_mask)
start = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
goal = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)

# ==================== 5️⃣ 改良 A* 導航，避開陸地尖角 ====================
def astar(start, goal, u, v, land_mask, safety):
    rows, cols = land_mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq = []
    heapq.heappush(pq, (0, start))
    came_from = {}
    cost = {start:0}
    visited = set()

    while pq:
        _, cur = heapq.heappop(pq)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            break
        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if ni<0 or nj<0 or ni>=rows or nj>=cols:
                continue
            if land_mask[ni,nj]:
                continue
            # 加入鄰近海域流場平滑
            flow = u[cur]*d[1] + v[cur]*d[0]
            land_penalty = 1/(safety[ni,nj]+2)  # 加大避開尖角效果
            dist_step = np.sqrt(d[0]**2+d[1]**2)
            new_cost = cost[cur] + dist_step + land_penalty - 0.5*flow
            if (ni,nj) not in cost or new_cost < cost[(ni,nj)]:
                cost[(ni,nj)] = new_cost
                heapq.heappush(pq,(new_cost,(ni,nj)))
                came_from[(ni,nj)] = cur
    if goal not in came_from:
        return []
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path

path = astar(start, goal, u, v, land_mask, safety)

# ==================== 6️⃣ 自訂海流顏色 ====================
colors = ["#E5F0FF","#CCE0FF","#99C2FF","#66A3FF","#3385FF",
          "#0066FF","#0052CC","#003D99","#002966","#001433","#000E24"]
levels = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_flow", list(zip(levels, colors)))

# ==================== 7️⃣ 畫圖 ====================
def plot_navigation(lons, lats, u, v, path, s_lon, s_lat, e_lon, e_lat):
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118,124,20,26], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True)
    speed = np.sqrt(u**2 + v**2)
    im = ax.pcolormesh(lons, lats, speed, cmap=cmap, shading='auto', alpha=0.8, transform=ccrs.PlateCarree())
    plt.colorbar(im, label='流速 (m/s)', shrink=0.6)
    ax.quiver(lons[::2], lats[::2], u[::2, ::2], v[::2, ::2], color='white', alpha=0.4, scale=10, transform=ccrs.PlateCarree())
    if len(path)>0:
        path_lons = [lons[p[1]] for p in path]
        path_lats = [lats[p[0]] for p in path]
        ax.plot(path_lons, path_lats, color='red', linewidth=2, transform=ccrs.PlateCarree(), label='AI航線')
    ax.scatter(s_lon, s_lat, color='green', s=150, marker='o', edgecolors='black', label='起點', transform=ccrs.PlateCarree())
    ax.scatter(e_lon, e_lat, color='yellow', s=250, marker='*', edgecolors='black', label='終點', transform=ccrs.PlateCarree())
    ax.legend(loc='upper left')
    plt.title(f"HELIOS V6 即時監控 - {obs_time}")
    st.pyplot(fig)

plot_navigation(lons, lats, u, v, path, s_lon, s_lat, e_lon, e_lat)

# ==================== 8️⃣ 下方儀表板 ====================
distance_km = np.sqrt((s_lat-e_lat)**2 + (s_lon-e_lon)**2) * 111
estimated_hours = distance_km / ship_speed

col1, col2, col3 = st.columns(3)
col1.metric("⏱️ 預估航行時間", f"{estimated_hours:.1f} 小時")
col2.metric("📏 航行距離", f"{distance_km:.1f} km")
col3.metric("📅 觀測時間", obs_time.strftime("%m/%d %H:%M"))
