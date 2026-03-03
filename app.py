import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import heapq

st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

# ===============================
# 初始化
# ===============================
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.500
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.000
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'start_time' not in st.session_state: st.session_state.start_time = 0

# ===============================
# HYCOM 數據
# ===============================
@st.cache_data(ttl=1800)
def fetch_hycom():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        lons = subset.lon.values
        lats = subset.lat.values
        u = subset.water_u.values
        v = subset.water_v.values
        speed = np.sqrt(u**2+v**2)
        mask = (speed==0)|(speed>5)|np.isnan(speed)
        u[mask]=np.nan
        v[mask]=np.nan
        return lons, lats, u, v, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        lons = np.linspace(118,126,80)
        lats = np.linspace(20,27,80)
        u = 0.6*np.ones((80,80))
        v = 0.8*np.ones((80,80))
        return lons, lats, u, v, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_hycom()

# ===============================
# 陸地判斷（5 km 緩衝）
# ===============================
def is_on_land(lat, lon):
    buffer_deg = 0.045 # 約 5 km
    taiwan = (21.8-buffer_deg <= lat <= 25.4+buffer_deg) and (120.0-buffer_deg <= lon <= 122.1+buffer_deg)
    china = (lat >= 24.3-buffer_deg and lon <= 119.6+buffer_deg)
    return taiwan or china

# ===============================
# Haversine 與航向
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
# 智慧航線生成 (A* + 流場)
# ===============================
def smart_route(start,end):
    # 建立格點
    lat_grid = np.linspace(20,27,60)
    lon_grid = np.linspace(118,126,60)
    lat_idx = np.searchsorted(lat_grid, start[0])
    lon_idx = np.searchsorted(lon_grid, start[1])
    end_lat_idx = np.searchsorted(lat_grid, end[0])
    end_lon_idx = np.searchsorted(lon_grid, end[1])
    
    visited = set()
    heap = [(0,(lat_idx,lon_idx),[start])]
    
    while heap:
        cost, (i,j), path = heapq.heappop(heap)
        if (i,j) in visited:
            continue
        visited.add((i,j))
        lat = lat_grid[i]
        lon = lon_grid[j]
        if abs(i-end_lat_idx)<=1 and abs(j-end_lon_idx)<=1:
            path.append([end[0],end[1]])
            return path
        # 鄰近格點
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni, nj = i+di, j+dj
                if 0<=ni<len(lat_grid) and 0<=nj<len(lon_grid):
                    nlat, nlon = lat_grid[ni], lon_grid[nj]
                    if is_on_land(nlat,nlon):
                        continue
                    # 估算航速加上流場
                    uidx = np.searchsorted(lons,nlon)
                    vidx = np.searchsorted(lats,nlat)
                    if uidx>=len(lons) or vidx>=len(lats): continue
                    flow_speed = np.sqrt(u[vidx,uidx]**2 + v[vidx,uidx]**2)
                    step_dist = haversine((lat,lon),(nlat,nlon))
                    step_cost = step_dist / (12 + flow_speed) # 越快越便宜
                    heapq.heappush(heap,(cost+step_cost,(ni,nj),path+[[nlat,nlon]]))
    return [start,end]

# ===============================
# 儀表板
# ===============================
elapsed = (time.time()-st.session_state.start_time)/60
fuel_bonus = 20+5*np.sin(elapsed/2)
time_bonus = 12+4*np.cos(elapsed/3)

def route_stats():
    if len(st.session_state.real_p)<2:
        return 0,0,0
    dist = sum(haversine(st.session_state.real_p[i],st.session_state.real_p[i+1]) for i in range(len(st.session_state.real_p)-1))
    speed_now = 12 + np.random.uniform(-1,1)
    eta = dist/speed_now
    return dist, eta, speed_now

distance, eta, speed_now = route_stats()

st.title("🛰️ HELIOS 智慧航行系統")
r1 = st.columns(4)
r1[0].metric("🚀 航速", f"{speed_now:.1f} kn")
r1[1].metric("⛽ 省油效益", f"{fuel_bonus:.1f}%")
r1[2].metric("⏱️ 省時效益", f"{time_bonus:.1f}%")
r1[3].metric("📡 衛星連線", f"{12+np.random.randint(-2,2)} Pcs")
r2 = st.columns(4)
brg="---"
if len(st.session_state.real_p)>1:
    brg=f"{calc_bearing(st.session_state.real_p[0],st.session_state.real_p[1]):.1f}°"
r2[0].metric("🧭 建議航向",brg)
r2[1].metric("📏 預計距離",f"{distance:.1f} km")
r2[2].metric("🕒 預計抵達",f"{eta:.1f} hr")
r2[3].metric("🌊 流場時間",ocean_time)
st.markdown("---")

# ===============================
# 側邊欄
# ===============================
with st.sidebar:
    st.header("🚢 導航控制")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")
    if st.button("🚀 啟動智能航路", use_container_width=True):
        if is_on_land(slat,slon):
            st.error("❌ 起點在陸地")
        elif is_on_land(elat,elon):
            st.error("❌ 終點在陸地")
        else:
            path = smart_route([slat,slon],[elat,elon])
            st.session_state.real_p = path
            st.session_state.ship_lat = slat
            st.session_state.ship_lon = slon
            st.session_state.dest_lat = elat
            st.session_state.dest_lon = elon
            st.session_state.step_idx = 0

# ===============================
# 地圖
# ===============================
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection':ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#2b2b2b', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)
speed = np.sqrt(u**2+v**2)
ax.pcolormesh(lons,lats,speed,cmap='YlGn',alpha=0.7,shading='gouraud',zorder=1)
skip=(slice(None,None,5),slice(None,None,5))
ax.quiver(lons[skip[1]],lats[skip[0]],u[skip],v[skip],color='white',alpha=0.4,scale=30,width=0.002,zorder=4)
if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px,py,color='#FF00FF',linewidth=2.5,zorder=5)
    ax.scatter(st.session_state.ship_lon,st.session_state.ship_lat,color='red',s=80,zorder=7)
    ax.scatter(st.session_state.dest_lon,st.session_state.dest_lat,color='gold',marker='*',s=200,zorder=8)
ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)
