import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import time

# ===============================
# 1. 初始化
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

# 初始化 session_state
for key, val in {
    'ship_lat': 25.060, 'ship_lon': 122.200,
    'dest_lat': 22.500, 'dest_lon': 120.000,
    'real_p': [], 'step_idx': 0,
    'start_time': time.time()
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===============================
# 2. HYCOM 即時流場抓取
# ===============================
@st.cache_data(ttl=900)  # 15分鐘更新
def fetch_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        lons = subset.lon.values
        lats = subset.lat.values
        u = subset.water_u.values
        v = subset.water_v.values
        speed = np.sqrt(u**2+v**2)
        mask = (speed==0) | (speed>5) | np.isnan(speed)
        u[mask]=np.nan
        v[mask]=np.nan
        return lons, lats, u, v, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        # 模擬資料
        lons = np.linspace(118, 126, 80)
        lats = np.linspace(20, 27, 80)
        u = 0.6*np.ones((80,80))
        v = 0.8*np.ones((80,80))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if (21.8<=lat<=25.4 and 120.0<=lon<=122.1) or (lat>=24.3 and lon<=119.6):
                    u[i,j]=np.nan
                    v[i,j]=np.nan
        return lons, lats, u, v, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_ocean_data()

# ===============================
# 3. 陸地判斷
# ===============================
def is_on_land(lat, lon):
    taiwan = (21.8<=lat<=25.4) and (120.0<=lon<=122.1)
    china = (lat>=24.3 and lon<=119.6)
    return taiwan or china

# ===============================
# 4. GPS 模擬
# ===============================
def gps_position():
    return st.session_state.ship_lat + np.random.normal(0,0.002), \
           st.session_state.ship_lon + np.random.normal(0,0.002)

# ===============================
# 5. 地球距離
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
    return (np.degrees(np.arctan2(y,x)) + 360)%360

# ===============================
# 6. 智慧航線（考慮流場）
# ===============================
def smart_route(start, end, lons, lats, u, v):
    # 簡單格點 A* 演算法
    from heapq import heappush, heappop
    lat_grid = lats
    lon_grid = lons
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
            # 將格點轉回經緯度
            return [[lat_grid[i], lon_grid[j]] for i,j in path], None
        i,j = current
        if visited[i,j]: continue
        visited[i,j] = True

        # 鄰近八格
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni, nj = i+di, j+dj
                if 0<=ni<nlat and 0<=nj<nlon and not visited[ni,nj]:
                    if np.isnan(u[ni,nj]) or np.isnan(v[ni,nj]):
                        continue
                    # 花費=距離-流速加成
                    dist = haversine([lat_grid[i], lon_grid[j]], [lat_grid[ni], lon_grid[nj]])
                    flow_bonus = np.sqrt(u[ni,nj]**2 + v[ni,nj]**2)
                    heappush(heap, (cost + dist/1.0 - flow_bonus, (ni,nj), path+[(ni,nj)]))
    return None, "找不到安全航路"

# ===============================
# 7. 路徑統計
# ===============================
def route_stats():
    if len(st.session_state.real_p)<2:
        return 0,0,0
    dist = sum(haversine(st.session_state.real_p[i], st.session_state.real_p[i+1])
               for i in range(len(st.session_state.real_p)-1))
    speed_now = 12 + 3*np.random.rand()  # 即時航速模擬
    eta = dist / speed_now
    return dist, eta, speed_now

distance, eta, speed_now = route_stats()

# ===============================
# 8. 儀表板（兩行）
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
        path, msg = smart_route([slat, slon],[elat, elon], lons, lats, u, v)
        if msg:
            st.error(msg)
        elif path is not None:
            st.session_state.real_p = path
            st.session_state.step_idx = 0
            st.session_state.ship_lat = slat
            st.session_state.ship_lon = slon
            st.session_state.dest_lat = elat
            st.session_state.dest_lon = elon
            st.experimental_rerun()

# ===============================
# 10. 地圖
# ===============================
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor="#2b2b2b", zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

# 流速
speed = np.sqrt(u**2 + v**2)
ax.pcolormesh(lons, lats, speed, cmap='YlGn', alpha=0.7, shading='gouraud', zorder=1)

# 箭頭
skip = (slice(None,None,5), slice(None,None,5))
ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.4, scale=30, width=0.002, zorder=4)

# 航線
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
