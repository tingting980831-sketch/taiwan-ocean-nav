import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from scipy.ndimage import distance_transform_edt
import heapq

# ==================== 設定 ====================
st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧導航控制台")

PLANE_COUNT = 3
SATS_PER_PLANE = 4
INCLINATION = 15

def get_visible_sats():
    return np.random.randint(2, 5)

# ==================== 讀取 HYCOM 資料 ====================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        
        lat_slice, lon_slice = slice(21, 26), slice(118, 124)
        u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        swh_data = ds['surf_el'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)  # 波高
        ws_data = np.sqrt(u_data**2 + v_data**2)  # 風速等效示意
        
        lons = u_data['lon'].values
        lats = u_data['lat'].values
        u_val = np.nan_to_num(u_data.values)
        v_val = np.nan_to_num(v_data.values)
        swh_val = np.nan_to_num(swh_data.values)
        ws_val = np.nan_to_num(ws_data.values)
        land_mask = np.isnan(u_data.values)
        
        return lons, lats, u_val, v_val, ws_val, swh_val, land_mask, latest_time
    except Exception as e:
        st.warning(f"HYCOM資料讀取失敗: {e}")
        return None, None, None, None, None, None, None, None

lons, lats, u, v, ws, swh, land_mask, obs_time = load_hycom_data()

# ==================== 禁航區與離岸風電 ====================
try:
    no_go_area = gpd.read_file("no_go_area.geojson")
except:
    st.warning("禁航區資料缺失，暫時不顯示")
    no_go_area = None

try:
    offshore_wind = gpd.read_file("offshore_wind.geojson")
except:
    st.warning("離岸風場資料缺失，暫時不顯示")
    offshore_wind = None

# ==================== 側邊欄 ====================
with st.sidebar:
    st.header("🌊 底圖選擇")
    map_option = st.radio("選擇底圖", ["流向", "風速", "波高"])

    st.header("📍 航點座標輸入")
    s_lon = st.number_input("起點經度", 118.0, 124.0, 120.3, format="%.2f")
    s_lat = st.number_input("起點緯度", 21.0, 26.0, 22.6, format="%.2f")
    e_lon = st.number_input("終點經度", 118.0, 124.0, 122.0, format="%.2f")
    e_lat = st.number_input("終點緯度", 21.0, 26.0, 24.5, format="%.2f")
    ship_speed = st.number_input("🚤 船速 (km/h)", value=20.0, step=1.0)
    run_nav = st.button("🚀 啟動衛星導航計算")

# ==================== 導航邏輯 ====================
def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()
    if not land_mask[lat_idx, lon_idx]:
        return (lat_idx, lon_idx)
    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]] - lat)**2 + (lons[ocean[1]] - lon)**2)
    i = dist.argmin()
    return (ocean[0][i], ocean[1][i])

def astar_v6(start, goal, u, v, land_mask, safety, ship_spd_kmh):
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

# ==================== 繪圖 ====================
if lons is not None and run_nav:
    safety = distance_transform_edt(~land_mask)
    start_idx = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
    goal_idx = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)
    path = astar_v6(start_idx, goal_idx, u, v, land_mask, safety, ship_speed)

    c1, c2, c3 = st.columns(3)
    if path:
        dist_km = sum(np.sqrt((lats[path[i][0]]-lats[path[i+1][0]])**2 + (lons[path[i][1]]-lons[path[i+1][1]])**2) for i in range(len(path)-1)) * 111
        c1.metric("⏱️ 預估航行時間", f"{dist_km/ship_speed:.1f} 小時")
        c2.metric("📏 航行距離", f"{dist_km:.1f} km")
    c3.metric("🛰️ 覆蓋衛星數", f"{get_visible_sats()} SATS (Incl: {INCLINATION}°)")
    st.caption(f"📅 數據時間: {obs_time.strftime('%Y-%m-%d %H:%M')}")
    st.divider()

    # 顏色設定
    colors_list = ["#E5F0FF","#CCE0FF","#99C2FF","#66A3FF","#3385FF",
                   "#0066FF","#0052CC","#003D99","#002966","#001433","#000E24"]
    levels = np.linspace(0, 1.2, len(colors_list))
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_flow", list(zip(levels/1.2, colors_list)))

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

    # 選擇底圖
    if map_option == "流向":
        data_plot = np.sqrt(u**2 + v**2)
        label = '流速 (m/s)'
    elif map_option == "風速":
        data_plot = ws
        label = '風速 (m/s)'
    elif map_option == "波高":
        data_plot = swh
        label = '波高 (m)'

    im = ax.pcolormesh(lons, lats, data_plot, cmap=cmap_custom, shading='auto', alpha=0.8, transform=ccrs.PlateCarree(), zorder=1)
    cbar = ax.figure.colorbar(im, ax=ax, label=label, shrink=0.6, pad=0.05)

    # 禁航區與離岸風場
    if no_go_area is not None:
        no_go_area.plot(ax=ax, facecolor='red', alpha=0.3, edgecolor='none', zorder=4, transform=ccrs.PlateCarree())
    if offshore_wind is not None:
        offshore_wind.plot(ax=ax, facecolor='yellow', alpha=0.3, edgecolor='none', zorder=5, transform=ccrs.PlateCarree())

    # 航跡
    if path:
        path_lons = [lons[p[1]] for p in path]
        path_lats = [lats[p[0]] for p in path]
        ax.plot(path_lons, path_lats, color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=6)

    plt.title(f"HELIOS V6 智慧導航監控")
    st.pyplot(fig)
