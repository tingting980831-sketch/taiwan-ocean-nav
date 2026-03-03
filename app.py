import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from shapely.geometry import Point, Polygon
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
import heapq
import time

# ===============================
# 1. 初始化
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.500
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.000
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'start_time' not in st.session_state: st.session_state.start_time = time.time()

# ===============================
# 2. 台灣多邊形 + 緩衝 (5km ≈ 0.045°)
# ===============================
taiwan_poly = Polygon([
    [21.8-0.045, 119.9-0.045],[21.8-0.045, 122.25+0.045],
    [25.35+0.045, 122.25+0.045],[25.35+0.045, 119.9-0.045]
])
def is_on_land(lat, lon):
    return taiwan_poly.contains(Point(lat, lon))

# ===============================
# 3. HYCOM 流場
# ===============================
@st.cache_data(ttl=1800)
def fetch_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        u = subset.water_u.values
        v = subset.water_v.values
        speed = np.sqrt(u**2+v**2)
        u[speed==0] = np.nan
        v[speed==0] = np.nan
        return subset.lon.values, subset.lat.values, u, v, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        lons = np.linspace(118, 126, 80)
        lats = np.linspace(20, 27, 80)
        u_sim = 0.6*np.ones((80,80))
        v_sim = 0.8*np.ones((80,80))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if is_on_land(lat, lon):
                    u_sim[i,j] = np.nan
                    v_sim[i,j] = np.nan
        return lons, lats, u_sim, v_sim, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_ocean_data()

# ===============================
# 4. 距離與航向
# ===============================
def haversine(p1,p2):
    R = 6371
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlat = lat2-lat1
    dlon = lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def calc_bearing(p1,p2):
    y = np.sin(np.radians(p2[1]-p1[1]))*np.cos(np.radians(p2[0]))
    x = np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0])) - np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y,x))+360)%360

# ===============================
# 5. A* + 平滑航線
# ===============================
def astar_smooth(start, end, lons, lats, u, v):
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    h, w = lat_grid.shape
    visited = np.zeros((h,w),dtype=bool)
    
    def closest_idx(lat, lon):
        i = np.abs(lats-lat).argmin()
        j = np.abs(lons-lon).argmin()
        return i,j
    
    si,sj = closest_idx(*start)
    ei,ej = closest_idx(*end)
    
    heap = [(0, (si,sj), None)]
    parents = {}
    g_score = np.full((h,w), np.inf)
    g_score[si,sj] = 0
    
    while heap:
        cost, (i,j), parent = heapq.heappop(heap)
        if visited[i,j]:
            continue
        visited[i,j] = True
        parents[(i,j)] = parent
        if (i,j) == (ei,ej):
            break
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj = i+di,j+dj
                if 0<=ni<h and 0<=nj<w and not visited[ni,nj]:
                    if np.isnan(u[ni,nj]) or np.isnan(v[ni,nj]):
                        continue
                    move_cost = haversine((lat_grid[i,j],lon_grid[i,j]), (lat_grid[ni,nj],lon_grid[ni,nj]))
                    flow_dir = np.arctan2(v[ni,nj],u[ni,nj])
                    bearing = np.radians(calc_bearing((lat_grid[i,j],lon_grid[i,j]), (lat_grid[ni,nj],lon_grid[ni,nj])))
                    dot = np.cos(bearing-flow_dir)
                    move_cost *= 1 - 0.3*dot
                    new_cost = g_score[i,j]+move_cost
                    if new_cost<g_score[ni,nj]:
                        g_score[ni,nj]=new_cost
                        heapq.heappush(heap,(new_cost+(haversine((lat_grid[ni,nj],lon_grid[ni,nj]), end)),(ni,nj),(i,j)))
    # 回溯
    path = []
    node = (ei,ej)
    while node is not None:
        path.append([lat_grid[node], lon_grid[node]])
        node = parents.get(node)
    path = path[::-1]
    
    # 平滑 spline
    if len(path) >= 4:
        lats_path, lons_path = zip(*path)
        tck,u_param = splprep([lats_path, lons_path], s=0.1)
        smooth = splev(np.linspace(0,1, max(len(path)*3,50)), tck)
        path = list(zip(smooth[0], smooth[1]))
        # 過濾陸地
        path = [p for p in path if not is_on_land(*p)]
    return path

# ===============================
# 6. 航線統計
# ===============================
def route_stats(path):
    if len(path)<2:
        return 0,0,0
    dist = sum(haversine(path[i], path[i+1]) for i in range(len(path)-1))
    speed_now = 12 + np.random.normal(0,1)
    eta = dist/speed_now
    return dist, eta, speed_now

# ===============================
# 7. 儀表板
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")
r1 = st.columns(4)
r1[0].metric("🚀 航速", "---")
r1[1].metric("⛽ 省油效益", "---")
r1[2].metric("📡 衛星連線", "---")
r1[3].metric("🌊 流場時間", ocean_time)
r2 = st.columns(4)
r2[0].metric("🧭 建議航向", "---")
r2[1].metric("📏 剩餘路程", "---")
r2[2].metric("🕒 預計抵達", "---")
r2[3].metric("🕒 數據時標", datetime.now().strftime("%H:%M:%S"))

# ===============================
# 8. 側邊欄
# ===============================
with st.sidebar:
    st.header("🚢 導航控制")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")
    if st.button("🚀 啟動智慧航路", use_container_width=True):
        if is_on_land(slat, slon):
            st.error("❌ 起點在陸地")
        elif is_on_land(elat, elon):
            st.error("❌ 終點在陸地")
        else:
            path = astar_smooth([slat, slon],[elat, elon],lons,lats,u,v)
            st.session_state.real_p = path
            st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
            st.session_state.dest_lat, st.session_state.dest_lon = elat, elon
            st.session_state.step_idx = 0
            st.experimental_rerun()

# ===============================
# 9. 地圖
# ===============================
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#2b2b2b')
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2)
speed = np.sqrt(u**2 + v**2)
ax.pcolormesh(lons,lats,speed,cmap='YlGn', alpha=0.7, shading='gouraud')
skip = (slice(None,None,5), slice(None,None,5))
ax.quiver(lons[skip[1]],lats[skip[0]], u[skip], v[skip], color='white', alpha=0.4, scale=30, width=0.002)
if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#FF00FF', linewidth=2.5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80)
    ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color='gold', marker='*', s=200)
ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)
