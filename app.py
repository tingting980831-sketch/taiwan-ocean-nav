import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import heapq
from scipy.ndimage import distance_transform_edt

# ==================== 1️⃣ 設定 ====================
st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧導航控制台")

# ----------------- 側邊欄輸入 -----------------
with st.sidebar:
    st.header("📍 航點座標輸入")
    s_lon = st.number_input("起點經度", min_value=118.0, max_value=124.0, value=120.3, step=0.01)
    s_lat = st.number_input("起點緯度", min_value=21.0, max_value=26.0, value=22.6, step=0.01)
    e_lon = st.number_input("終點經度", min_value=118.0, max_value=124.0, value=122.0, step=0.01)
    e_lat = st.number_input("終點緯度", min_value=21.0, max_value=26.0, value=24.5, step=0.01)
    ship_speed = st.number_input("🚤 船速 (km/h)", min_value=1.0, max_value=50.0, value=20.0, step=1.0)
    map_option = st.selectbox("選擇底圖", ["流向", "風向", "波高"])
    run_nav = st.button("🚀 計算導航")

# ==================== 2️⃣ 讀取 HYCOM 與風浪資料 ====================
@st.cache_data(ttl=3600)
def load_hycom():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        
        lat_slice, lon_slice = slice(21, 26), slice(118, 124)
        u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        
        lons = u_data['lon'].values
        lats = u_data['lat'].values
        u_val = np.nan_to_num(u_data.values)
        v_val = np.nan_to_num(v_data.values)
        land_mask = np.isnan(u_data.values)

        return lons, lats, u_val, v_val, land_mask, latest_time
    except Exception as e:
        st.error(f"讀取 HYCOM 資料失敗: {e}")
        return None, None, None, None, None, None

lons, lats, u, v, land_mask, obs_time = load_hycom()

# ==================== 3️⃣ A* 導航算法 ====================
def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()
    if not land_mask[lat_idx, lon_idx]:
        return (lat_idx, lon_idx)
    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]] - lat)**2 + (lons[ocean[1]] - lon)**2)
    i = dist.argmin()
    return (ocean[0][i], ocean[1][i])

def astar(start, goal, u, v, land_mask, safety, ship_spd_kmh):
    v_ship = ship_spd_kmh * 0.277
    rows, cols = land_mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq, cost, came_from = [(0, start)], {start: 0}, {}

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
                step_c = (dist_m / v_ground) + (-flow_v * (dist_m / v_ship) * 1.5)
                if safety[ni,nj] < 4:
                    step_c += 12000 / (safety[ni,nj] + 0.2)**2
                new_total = cost[cur] + step_c
                if (ni,nj) not in cost or new_total < cost[(ni,nj)]:
                    cost[(ni,nj)] = new_total
                    priority = new_total + 4.0 * (np.sqrt((ni-goal[0])**2 + (nj-goal[1])**2) * 8000 / v_ship)
                    heapq.heappush(pq, (priority, (ni,nj)))
                    came_from[(ni,nj)] = cur
    path = []
    curr = goal
    while curr in came_from:
        path.append(curr); curr = came_from[curr]
    if path: path.append(start)
    return path[::-1]

# ==================== 4️⃣ 繪圖 ====================
if lons is not None and run_nav:
    safety = distance_transform_edt(~land_mask)
    start_idx = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
    goal_idx = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)
    path = astar(start_idx, goal_idx, u, v, land_mask, safety, ship_speed)

    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118, 124, 21, 26], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, zorder=3)

    gl = ax.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True

    # ----------------- 依底圖切換 -----------------
    if map_option == "流向":
        data_plot = np.sqrt(u**2 + v**2)
        cmap = plt.cm.Blues
        label = '流速 (m/s)'
    elif map_option == "風向":
        # 假設有風速 u_wind, v_wind
        u_wind, v_wind = u.copy(), v.copy()  # 用 HYCOM 替代示意
        data_plot = np.sqrt(u_wind**2 + v_wind**2)
        cmap = plt.cm.YlOrBr
        label = '風速 (m/s)'
    else:  # 波高
        swh = np.sqrt(u**2 + v**2)  # 示意 SWH, 可替換真實資料
        data_plot = swh
        cmap = plt.cm.Greens
        label = '波高 (m)'

    im = ax.pcolormesh(lons, lats, data_plot, cmap=cmap, shading='auto', alpha=0.8, transform=ccrs.PlateCarree(), zorder=1)
    cbar = ax.figure.colorbar(im, ax=ax, label=label, shrink=0.6, pad=0.05)

    # ----------------- 禁航區 & 離岸風場 -----------------
    try:
        no_go_area = gpd.read_file("no_go_area.geojson")  # 替換成你的禁航區檔案
        wind_area = gpd.read_file("wind_area.geojson")    # 離岸風電區
        no_go_area.plot(ax=ax, facecolor='red', alpha=0.3, edgecolor='red', zorder=3)
        wind_area.plot(ax=ax, facecolor='yellow', alpha=0.2, edgecolor='orange', zorder=3)
    except Exception as e:
        st.warning(f"禁航區或離岸風場資料缺失: {e}")

    # ----------------- 海流箭頭 -----------------
    ax.quiver(lons[::2], lats[::2], u[::2, ::2], v[::2, ::2], color='white', alpha=0.4, scale=10, transform=ccrs.PlateCarree(), zorder=4)

    # ----------------- 航行路徑 -----------------
    if path:
        path_lons = [lons[p[1]] for p in path]
        path_lats = [lats[p[0]] for p in path]
        ax.plot(path_lons, path_lats, color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)

    plt.title(f"HELIOS V6 智慧導航監控")
    st.pyplot(fig)
