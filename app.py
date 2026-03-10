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

st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("🛰️ HELIOS V7 智慧海象導航系統 (2D)")

# ===============================
# 禁航區
# ===============================
NO_GO_ZONES = [
    [[22.953536, 120.171678], [22.934628, 120.175472], [22.933136, 120.170942], [22.957810, 120.160780]],
    [[22.943956, 120.172358], [22.939717, 120.173944], [22.928353, 120.157372], [22.936636, 120.153547]],
    # 其他禁航區省略，保留你的原始資料
]

# ===============================
# 離岸風場
# ===============================
OFFSHORE_WIND = [
    [[24.18, 120.12], [24.22, 120.28], [24.05, 120.35], [24.00, 120.15]],
    # 其他離岸風場省略
]

# ===============================
# 讀取 HYCOM 海流
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)

    # 安全抓取時間
    try:
        time_origin_str = ds['time'].attrs.get('time_origin', None)
        if time_origin_str:
            time_origin = pd.to_datetime(time_origin_str)
            latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        else:
            latest_time = pd.Timestamp.now()
    except Exception:
        latest_time = pd.Timestamp.now()

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

lons, lats, u, v, land_mask, obs_time = load_hycom_data()

# ===============================
# 側邊欄設定
# ===============================
with st.sidebar:
    st.header("航點設定")
    s_lon = st.number_input("起點經度", 118.0,124.0,120.3)
    s_lat = st.number_input("起點緯度", 21.0,26.0,22.6)
    e_lon = st.number_input("終點經度", 118.0,124.0,122.0)
    e_lat = st.number_input("終點緯度", 21.0,26.0,24.5)
    ship_speed = st.number_input("船速 km/h",1.0,60.0,20.0)

# ===============================
# 最近海洋格點
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
# A* 航線
# ===============================
def astar(start, goal, u, v, land_mask, safety, ship_spd):
    v_ship = ship_spd*0.277  # km/h -> m/s
    rows, cols = land_mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq = [(0, start)]
    cost = {start:0}
    came = {}
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            break
        for d in dirs:
            ni = cur[0]+d[0]
            nj = cur[1]+d[1]
            if 0<=ni<rows and 0<=nj<cols and not land_mask[ni,nj]:
                dist_m = np.sqrt(d[0]**2+d[1]**2)*8000
                norm = np.sqrt(d[0]**2+d[1]**2)
                flow = (u[cur]*(d[1]/norm) + v[cur]*(d[0]/norm))
                v_ground = max(0.5, v_ship+flow)
                step = (dist_m/v_ground)+(-flow*(dist_m/v_ship)*1.5)
                if safety[ni,nj]<4:
                    step += 12000/(safety[ni,nj]+0.2)**2
                new = cost[cur]+step
                if (ni,nj) not in cost or new<cost[(ni,nj)]:
                    cost[(ni,nj)] = new
                    priority = new + 4*np.sqrt((ni-goal[0])**2+(nj-goal[1])**2)*8000/v_ship
                    heapq.heappush(pq,(priority,(ni,nj)))
                    came[(ni,nj)] = cur
    path=[]
    curr = goal
    while curr in came:
        path.append(curr)
        curr = came[curr]
    if path:
        path.append(start)
    return path[::-1]

# ===============================
# 計算航線並畫圖
# ===============================
if lons is not None:
    safety = distance_transform_edt(~land_mask)
    start = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
    goal  = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)
    path = astar(start, goal, u, v, land_mask, safety, ship_speed)

    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118,124,21,26])
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)

    # 海流圖 (只畫流速)
    total_factor = np.sqrt(u**2+v**2)
    colors=[ "#E5F0FF","#CCE0FF","#99C2FF","#66A3FF",
             "#3385FF","#0066FF","#0052CC","#003D99",
             "#002966","#001433","#000E24"]
    cmap = mcolors.LinearSegmentedColormap.from_list("flow", colors)
    ax.pcolormesh(lons, lats, total_factor, cmap=cmap, shading='auto', alpha=0.8)

    # 禁航區 (紅色)
    for zone in NO_GO_ZONES:
        poly = np.array(zone)
        ax.fill(poly[:,1], poly[:,0], color='red', alpha=0.4)

    # 離岸風場 (黃色)
    for zone in OFFSHORE_WIND:
        poly = np.array(zone)
        ax.fill(poly[:,1], poly[:,0], color='yellow', alpha=0.4)

    # 航線
    if path:
        path_lons = [lons[p[1]] for p in path]
        path_lats = [lats[p[0]] for p in path]
        ax.plot(path_lons, path_lats, color='green', linewidth=2)

    # 起終點
    ax.scatter(s_lon, s_lat, color='green', s=120, edgecolors='black', zorder=5)
    ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, edgecolors='black', zorder=5)

    plt.title(f"HELIOS V7 Navigation (2D) - {obs_time}")
    st.pyplot(fig)
    plt.close(fig)
