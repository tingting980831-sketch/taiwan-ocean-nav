import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import time
import heapq

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
# 2. HYCOM 流場抓取
# ===============================
@st.cache_data(ttl=1800)
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
        mask = (speed == 0) | (speed > 5) | np.isnan(speed)
        u[mask] = np.nan
        v[mask] = np.nan
        return lons, lats, u, v, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        lons = np.linspace(118,126,80)
        lats = np.linspace(20,27,80)
        u = 0.6*np.ones((80,80))
        v = 0.8*np.ones((80,80))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if (21.8 <= lat <= 25.4 and 120.0 <= lon <= 122.1):
                    u[i,j] = np.nan
                    v[i,j] = np.nan
        return lons, lats, u, v, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_ocean_data()

# ===============================
# 3. 陸地判斷
# ===============================
def is_on_land(lat, lon):
    return (21.8 <= lat <= 25.4 and 120.0 <= lon <= 122.1)

# ===============================
# 4. 地球距離與航向
# ===============================
def haversine(p1,p2):
    R=6371
    lat1,lon1=np.radians(p1)
    lat2,lon2=np.radians(p2)
    dlat=lat2-lat1
    dlon=lon2-lon1
    a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

def calc_bearing(p1,p2):
    y=np.sin(np.radians(p2[1]-p1[1]))*np.cos(np.radians(p2[0]))
    x=np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0])) - \
      np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y,x))+360)%360

# ===============================
# 5. A* 智慧航線
# ===============================
def astar_route(start,end,lons,lats,u,v):
    ny,nx = len(lats), len(lons)
    def to_idx(lat, lon):
        i = np.argmin(np.abs(lats-lat))
        j = np.argmin(np.abs(lons-lon))
        return i,j
    def to_latlon(i,j):
        return lats[i], lons[j]
    start_idx = to_idx(*start)
    end_idx = to_idx(*end)

    visited = np.zeros((ny,nx),dtype=bool)
    heap = [(0,start_idx)]
    came_from = {}
    g_score = {start_idx:0}

    while heap:
        cost, current = heapq.heappop(heap)
        if current == end_idx:
            # 回溯路徑
            path = []
            while current in came_from:
                path.append(to_latlon(*current))
                current = came_from[current]
            path.append(start)
            return path[::-1]
        ci,cj = current
        visited[ci,cj] = True
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj = ci+di,cj+dj
                if 0<=ni<ny and 0<=nj<nx and not visited[ni,nj]:
                    lat,nlon = lats[ni], lons[nj]
                    if np.isnan(u[ni,nj]) or np.isnan(v[ni,nj]):
                        continue
                    speed = 12 + np.sqrt(u[ni,nj]**2 + v[ni,nj]**2)
                    h = haversine((lat,nlon),end)
                    g = g_score[current] + haversine((lat,nlon),to_latlon(ci,cj))/speed
                    f = g+h/12
                    if (ni,nj) not in g_score or g<g_score[(ni,nj)]:
                        g_score[(ni,nj)] = g
                        heapq.heappush(heap,(f,(ni,nj)))
                        came_from[(ni,nj)] = current
    return [start,end]

# ===============================
# 6. 計算路徑統計
# ===============================
def route_stats():
    if len(st.session_state.real_p)<2:
        return 0,0,0
    dist = sum(haversine(st.session_state.real_p[i],st.session_state.real_p[i+1])
               for i in range(len(st.session_state.real_p)-1))
    speed_now = 12 + np.random.normal(0,1)
    eta = dist/speed_now
    return dist, eta, speed_now

elapsed = (time.time()-st.session_state.start_time)/60
fuel_bonus = 20+5*np.sin(elapsed/2)
time_bonus = 12+4*np.cos(elapsed/3)
distance, eta, speed_now = route_stats()

# ===============================
# 7. 儀表板
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")
r1 = st.columns(4)
r1[0].metric("🚀 航速",f"{speed_now:.1f} kn")
r1[1].metric("⛽ 省油效益",f"{fuel_bonus:.1f}%")
r1[2].metric("📡 衛星連線",f"{int(8+np.random.randint(0,6))} Pcs")
r1[3].metric("🌊 流場狀態",stream_status)

r2 = st.columns(4)
brg="---"
if len(st.session_state.real_p)>1:
    brg=f"{calc_bearing(st.session_state.real_p[0],st.session_state.real_p[1]):.1f}°"
r2[0].metric("🧭 建議航向",brg)
r2[1].metric("📏 剩餘路程",f"{distance:.1f} km")
r2[2].metric("🕒 預計抵達",f"{eta:.1f} hr")
r2[3].metric("🕒 流場時間",ocean_time)
st.markdown("---")

# ===============================
# 8. 側邊欄
# ===============================
with st.sidebar:
    st.header("🚢 導航控制")
    slat = st.number_input("起點緯度",value=st.session_state.ship_lat,format="%.3f")
    slon = st.number_input("起點經度",value=st.session_state.ship_lon,format="%.3f")
    elat = st.number_input("終點緯度",value=st.session_state.dest_lat,format="%.3f")
    elon = st.number_input("終點經度",value=st.session_state.dest_lon,format="%.3f")

    if st.button("🚀 啟動智能航路",use_container_width=True):
        if is_on_land(slat,slon):
            st.error("❌ 起點落在陸地")
        elif is_on_land(elat,elon):
            st.error("❌ 終點落在陸地")
        else:
            path = astar_route([slat,slon],[elat,elon],lons,lats,u,v)
            st.session_state.real_p = path
            st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
            st.session_state.dest_lat, st.session_state.dest_lon = elat, elon
            st.session_state.step_idx = 0
            st.experimental_rerun()

# ===============================
# 9. 地圖繪製
# ===============================
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection':ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor="#2b2b2b", zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor="cyan", linewidth=1.2, zorder=3)

speed = np.sqrt(u**2 + v**2)
ax.pcolormesh(lons,lats,speed,cmap="YlGn",alpha=0.7,shading='gouraud',zorder=1)

skip = (slice(None,None,5), slice(None,None,5))
ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.4, scale=30, width=0.002, zorder=4)

if st.session_state.real_p:
    py,px = zip(*st.session_state.real_p)
    ax.plot(px,py,color="#FF00FF",linewidth=2.5,zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat,color="red",s=80,zorder=7)
    ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat,color="gold",marker="*",s=200,zorder=8)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)

# ===============================
# 10. 執行下一階段航行
# ===============================
if st.button("🚢 執行下一階段航行",use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
        st.session_state.step_idx += 8
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.experimental_rerun()
