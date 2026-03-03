import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import time
from heapq import heappush, heappop

# ===============================
# 1. 初始化 session_state
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

default_state = {
    'ship_lat': 25.06, 'ship_lon': 122.2,
    'dest_lat': 22.5, 'dest_lon': 120.0,
    'real_p': [], 'step_idx': 0,
    'start_time': time.time(),
    'rerun_flag': False
}
for key, val in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===============================
# 2. HYCOM 即時流場
# ===============================
@st.cache_data(ttl=900)
def fetch_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        lons = subset.lon.values
        lats = subset.lat.values
        u = subset.water_u.values
        v = subset.water_v.values
        speed = np.sqrt(u**2 + v**2)
        mask = (speed==0) | (speed>5) | np.isnan(speed)
        u[mask]=np.nan
        v[mask]=np.nan
        return lons, lats, u, v, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        # 模擬資料
        lons = np.linspace(118,126,80)
        lats = np.linspace(20,27,80)
        u = 0.6*np.ones((80,80))
        v = 0.8*np.ones((80,80))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if (21.8<=lat<=25.4 and 120<=lon<=122.1) or (lat>=24.3 and lon<=119.6):
                    u[i,j]=np.nan
                    v[i,j]=np.nan
        return lons, lats, u, v, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_ocean_data()

# ===============================
# 3. 陸地判斷
# ===============================
def is_on_land(lat, lon):
    return (21.8<=lat<=25.4 and 120<=lon<=122.1) or (lat>=24.3 and lon<=119.6)

# ===============================
# 4. GPS 模擬
# ===============================
def gps_position():
    return st.session_state.ship_lat + np.random.normal(0,0.002), \
           st.session_state.ship_lon + np.random.normal(0,0.002)

# ===============================
# 5. 距離與航向
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
    x = np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0])) - \
        np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y,x))+360)%360

# ===============================
# 6. 智慧航線（A* + 流場加權）
# ===============================
def smart_route(start, end, lons, lats, u, v):
    lat_grid, lon_grid = lats, lons
    nlat, nlon = len(lat_grid), len(lon_grid)

    def closest_idx(val, arr):
        return int(np.argmin(np.abs(arr-val)))

    start_idx = (closest_idx(start[0], lat_grid), closest_idx(start[1], lon_grid))
    end_idx   = (closest_idx(end[0], lat_grid), closest_idx(end[1], lon_grid))

    visited = np.zeros((nlat,nlon), dtype=bool)
    heap = []
    heappush(heap, (0, start_idx, [start_idx]))

    while heap:
        cost, current, path = heappop(heap)
        if current == end_idx:
            return [[lat_grid[i], lon_grid[j]] for i,j in path], None
        i,j = current
        if visited[i,j]: continue
        visited[i,j]=True
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj = i+di, j+dj
                if 0<=ni<nlat and 0<=nj<nlon and not visited[ni,nj]:
                    if np.isnan(u[ni,nj]) or np.isnan(v[ni,nj]): continue
                    dist = haversine([lat_grid[i],lon_grid[j]], [lat_grid[ni],lon_grid[nj]])
                    flow_bonus = np.sqrt(u[ni,nj]**2+v[ni,nj]**2)
                    heappush(heap, (cost + dist - flow_bonus, (ni,nj), path+[(ni,nj)]))
    return None, "找不到安全航路"

# ===============================
# 7. 路徑統計
# ===============================
def route_stats():
    if len(st.session_state.real_p)<2:
        return 0,0,0
    dist = sum(haversine(st.session_state.real_p[i], st.session_state.real_p[i+1])
               for i in range(len(st.session_state.real_p)-1))
    speed_now = 12 + np.random.rand()*3  # 模擬航速
    eta = dist / speed_now
    return dist, eta, speed_now

distance, eta, speed_now = route_stats()

# ===============================
# 8. 儀表板
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")

elapsed = (time.time() - st.session_state.start_time)/60
fuel_bonus = 20 + 5*np.sin(elapsed/2)
time_bonus = 12 + 4*np.cos(elapsed/3)

r1 = st.columns(4)
r1[0].metric("🚀 航速", f"{speed_now:.1f} kn")
r1[1].metric("⛽ 省油效益", f"{fuel_bonus:.1f}%")
r1[2].metric("⏱️ 省時效益", f"{time_bonus:.1f}%")
r1[3].metric("📡 衛星", f"{10 + int(np.random.rand()*5)} Pcs")

r2 = st.columns(4)
brg = "---"
if len(st.session_state.real_p)>1:
    brg = f"{calc_bearing(st.session_state.real_p[0], st.session_state.real_p[1]):.1f}°"
r2[0].metric("🧭 建議航向", brg)
r2[1].metric("📏 預計距離", f"{distance:.1f} km")
r2[2].metric("🕒 預計時間", f"{eta:.1f} hr")
r2[3].metric("🌊 流場時間", ocean_time)

st.markdown("---")

# ===============================
# 9. 側邊欄
# ===============================
with st.sidebar:
    st.header("🚢 導航控制中心")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    if st.button("📍 GPS定位起點"):
        st.session_state.ship_lat, st.session_state.ship_lon = gps_position()

    if st.button("🚀 啟動智慧航路", use_container_width=True):
        st.session_state.rerun_flag = True  # 標記 rerun
        path, msg = smart_route([slat,slon],[elat,elon], lons,lats,u,v)
        if msg:
            st.error(msg)
        else:
            st.session_state.real_p = path
            st.session_state.step_idx = 0
            st.session_state.ship_lat = slat
            st.session_state.ship_lon = slon
            st.session_state.dest_lat = elat
            st.session_state.dest_lon = elon

# 立即 rerun（安全方式）
if st.session_state.rerun_flag:
    st.session_state.rerun_flag = False
    st.experimental_rerun()

# ===============================
# 10. 地圖
# ===============================
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor="#2b2b2b", zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

speed = np.sqrt(u**2+v**2)
ax.pcolormesh(lons, lats, speed, cmap='YlGn', alpha=0.7, shading='gouraud', zorder=1)

skip = (slice(None,None,5), slice(None,None,5))
ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.4, scale=30, width=0.002, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, zorder=7)
    ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color='gold', marker='*', s=200, zorder=8)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)

# ===============================
# 11. 航行模擬
# ===============================
if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
        st.session_state.step_idx += 6
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.experimental_rerun()
