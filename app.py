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
import os

# ==================== 1️⃣ HELIOS V6 設定 ====================
st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧海洋導航系統")

def get_visible_sats():
    return np.random.randint(2, 6)

# ==================== 2️⃣ 讀取公開 ERA5 風場 + 波高 ====================
@st.cache_data(ttl=3600)
def load_era5_ocean():
    # 公開 Zarr Dataset
    url = "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-ocean-v0.zarr"
    ds = xr.open_dataset(url, engine="zarr",
                         storage_options={"client_kwargs": {"trust_env": True}})
    # 選台灣附近區域
    ds = ds.sel(latitude=slice(20, 26), longitude=slice(118, 124))
    # 取最新時間
    ds_latest = ds.isel(time=-1)
    # 風場
    u10 = ds_latest["u10"].values
    v10 = ds_latest["v10"].values
    wind_speed = np.sqrt(u10**2 + v10**2)
    # 波高
    swh = ds_latest["swh"].values
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    return lon, lat, u10, v10, wind_speed, swh

era5_lons, era5_lats, era5_u10, era5_v10, era5_ws, era5_swh = load_era5_ocean()

# ==================== 3️⃣ 讀取 HYCOM 海流資料 ====================
@st.cache_data(ttl=3600)
def load_hycom():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        # 選擇台海域
        u_data = ds['ssu'].sel(lat=slice(21, 26), lon=slice(118, 124)).isel(time=-1)
        v_data = ds['ssv'].sel(lat=slice(21, 26), lon=slice(118, 124)).isel(time=-1)
        lons = u_data['lon'].values
        lats = u_data['lat'].values
        u_val = np.nan_to_num(u_data.values)
        v_val = np.nan_to_num(v_data.values)
        land_mask = np.isnan(u_data.values)
        return lons, lats, u_val, v_val, land_mask, latest_time
    except Exception as e:
        st.error("HYCOM 讀取失敗：" + str(e))
        return None, None, None, None, None, None

lons, lats, u, v, land_mask, obs_time = load_hycom()

# ==================== 4️⃣ A* 導航算法 ====================
def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()
    if not land_mask[lat_idx, lon_idx]:
        return (lat_idx, lon_idx)
    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]] - lat)**2 + (lons[ocean[1]] - lon)**2)
    i = dist.argmin()
    return (ocean[0][i], ocean[1][i])

def astar(start, goal, u, v, land_mask, safety, ship_spd):
    v_ship = ship_spd * 0.277
    rows, cols = land_mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq, cost, came = [(0, start)], {start: 0}, {}
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal: break
        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if 0<=ni<rows and 0<=nj<cols and not land_mask[ni,nj]:
                dist_m = np.sqrt(d[0]**2 + d[1]**2)*8000
                norm = np.sqrt(d[0]**2 + d[1]**2)
                flow_v = (u[cur]*d[1]/norm + v[cur]*d[0]/norm)
                v_ground = max(0.5, v_ship + flow_v)
                cost_step = dist_m/v_ground
                new_cost = cost[cur] + cost_step
                if (ni,nj) not in cost or new_cost < cost[(ni,nj)]:
                    cost[(ni,nj)] = new_cost
                    priority = new_cost + np.sqrt((ni-goal[0])**2 + (nj-goal[1])**2)*8000/v_ship
                    heapq.heappush(pq, (priority,(ni,nj)))
                    came[(ni,nj)] = cur
    path=[]
    node=goal
    while node in came:
        path.append(node); node=came[node]
    if path: path.append(start)
    return path[::-1]

# ==================== 5️⃣ Streamlit UI ====================
with st.sidebar:
    st.header("📍 航點輸入")
    s_lon = st.number_input("起點經度", min_value=118.0, max_value=124.0, value=120.3, step=0.01)
    s_lat = st.number_input("起點緯度", min_value=21.0, max_value=26.0, value=22.6, step=0.01)
    e_lon = st.number_input("終點經度", min_value=118.0, max_value=124.0, value=122.0, step=0.01)
    e_lat = st.number_input("終點緯度", min_value=21.0, max_value=26.0, value=24.5, step=0.01)
    ship_speed = st.number_input("🚤 船速 km/h", min_value=1.0, max_value=50.0, value=20.0)
    map_type = st.selectbox("底圖類型", ["海流流速", "風速 (ERA5)", "波高 (ERA5)"])
    run_nav = st.button("🚀 計算導航")

# ==================== 6️⃣ 作圖 & 顯示 ====================
if lons is not None and run_nav:
    # 計算路徑
    safety = distance_transform_edt(~land_mask)
    start_idx = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
    goal_idx = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)
    path = astar(start_idx, goal_idx, u, v, land_mask, safety, ship_speed)

    # 顯示儀表板
    if path:
        dist_km = sum(np.sqrt((lats[path[i][0]]-lats[path[i+1][0]])**2 + (lons[path[i][1]]-lons[path[i+1][1]])**2)*111 for i in range(len(path)-1))
        st.metric("📏 航行距離", f"{dist_km:.1f} km")
        st.metric("⏱️ 航行時間", f"{dist_km/ship_speed:.1f} hr")
    st.metric("🛰️ 可見衛星數", f"{get_visible_sats()} SATS")
    st.caption(f"資料時間： {obs_time.strftime('%Y-%m-%d %H:%M')}")

    # 地圖畫布
    fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection':ccrs.PlateCarree()})
    ax.set_extent([118,124,21,26], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)

    # 經緯度標籤（只左下）
    gl = ax.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = False; gl.right_labels = False

    # 底圖切換
    if map_type=="海流流速":
        speed = np.sqrt(u**2+v**2)
        im = ax.pcolormesh(lons, lats, speed, cmap='Blues', shading='auto', alpha=0.8)
        plt.colorbar(im, ax=ax, label='流速 (m/s)')
        ax.streamplot(lons, lats, u, v, color='white', density=1.5, arrowsize=1)

    elif map_type=="風速 (ERA5)":
        im = ax.pcolormesh(era5_lons, era5_lats, era5_ws, cmap='YlOrBr', shading='auto', alpha=0.8)
        plt.colorbar(im, ax=ax, label='風速 (m/s)')
        ax.streamplot(era5_lons, era5_lats, era5_u10, era5_v10, color='yellow', density=1.5, arrowsize=1)

    else:  # 波高
        im = ax.pcolormesh(era5_lons, era5_lats, era5_swh, cmap='viridis', shading='auto', alpha=0.8)
        plt.colorbar(im, ax=ax, label='波高 (m)')

    # 禁航區
    if os.path.exists("no_go_area.geojson"):
        no_go = gpd.read_file("no_go_area.geojson")
        ax.add_geometries(no_go.geometry, crs=ccrs.PlateCarree(), facecolor='red', alpha=0.3, edgecolor=None)

    # 離岸風場
    if os.path.exists("wind_area.geojson"):
        wind_area = gpd.read_file("wind_area.geojson")
        ax.add_geometries(wind_area.geometry, crs=ccrs.PlateCarree(), facecolor='yellow', alpha=0.2, edgecolor=None)

    # 航路
    if path:
        path_lons = [lons[p[1]] for p in path]
        path_lats = [lats[p[0]] for p in path]
        ax.plot(path_lons, path_lats, color='red', linewidth=2)

    plt.title("HELIOS V6 智慧導航")
    st.pyplot(fig)
    
