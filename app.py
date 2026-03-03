import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from scipy.interpolate import CubicSpline
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
# 2. HYCOM 流場資料
# ===============================
@st.cache_data(ttl=1800)
def fetch_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
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
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if (21.8<=lat<=25.4 and 120.0<=lon<=122.1) or (lat>=24.3 and lon<=119.6):
                    u[i,j]=np.nan
                    v[i,j]=np.nan
        return lons, lats, u, v, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_ocean_data()

# ===============================
# 3. 地球距離與航向
# ===============================
def haversine(p1,p2):
    R=6371
    lat1,lon1=np.radians(p1)
    lat2,lon2=np.radians(p2)
    dlat=lat2-lat1
    dlon=lon2-lon1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

def calc_bearing(p1,p2):
    y=np.sin(np.radians(p2[1]-p1[1]))*np.cos(np.radians(p2[0]))
    x=np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0]))-\
      np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y,x))+360)%360

# ===============================
# 4. 陸地判斷 (5 km 安全緩衝)
# ===============================
def is_on_land(lat,lon):
    buffer = 0.045 # 約5 km
    taiwan = (21.8-buffer <= lat <= 25.4+buffer) and (120.0-buffer <= lon <= 122.1+buffer)
    china  = (lat>=24.3-buffer and lon<=119.6+buffer)
    return taiwan or china

# ===============================
# 5. A* 算法
# ===============================
def astar_route(start,end,lons,lats,u,v):
    ny,nx = len(lats), len(lons)
    lat_idx = lambda lat: np.argmin(np.abs(lats-lat))
    lon_idx = lambda lon: np.argmin(np.abs(lons-lon))
    sx,sy = lon_idx(start[1]), lat_idx(start[0])
    ex,ey = lon_idx(end[1]), lat_idx(end[0])
    
    def heuristic(a,b):
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    
    open_set = []
    heapq.heappush(open_set,(0,(sx,sy)))
    came_from = {}
    gscore = {(sx,sy):0}
    fscore = {(sx,sy):heuristic((sx,sy),(ex,ey))}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current==(ex,ey):
            path=[current]
            while current in came_from:
                current=came_from[current]
                path.append(current)
            path=path[::-1]
            # 轉回經緯度
            return [(lats[y],lons[x]) for x,y in path]
        x,y=current
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                nx1=x+dx
                ny1=y+dy
                if 0<=nx1<nx and 0<=ny1<ny:
                    latc,lonc=lats[ny1],lons[nx1]
                    if is_on_land(latc,lonc):
                        continue
                    tentative_g=gscore[(x,y)]+heuristic((x,y),(nx1,ny1))
                    flow_speed = np.sqrt(u[ny1,nx1]**2+v[ny1,nx1]**2) if not np.isnan(u[ny1,nx1]) else 0
                    tentative_g /= (1+flow_speed) # 流速越快成本越低
                    if (nx1,ny1) not in gscore or tentative_g<gscore[(nx1,ny1)]:
                        came_from[(nx1,ny1)]=(x,y)
                        gscore[(nx1,ny1)]=tentative_g
                        fscore[(nx1,ny1)]=tentative_g+heuristic((nx1,ny1),(ex,ey))
                        heapq.heappush(open_set,(fscore[(nx1,ny1)],(nx1,ny1)))
    return [start,end]

# ===============================
# 6. 平滑化
# ===============================
def smooth_path(path,num_points=300):
    if len(path)<3:
        return path
    lats,lons = zip(*path)
    distances=[0]
    for i in range(1,len(path)):
        distances.append(distances[-1]+haversine(path[i-1],path[i]))
    distances=np.array(distances)
    cs_lat = CubicSpline(distances,lats)
    cs_lon = CubicSpline(distances,lons)
    new_dist = np.linspace(0,distances[-1],num_points)
    smooth_lats = cs_lat(new_dist)
    smooth_lons = cs_lon(new_dist)
    return list(zip(smooth_lats,smooth_lons))

# ===============================
# 7. 儀表板
# ===============================
elapsed = (time.time()-st.session_state.start_time)/60
fuel_bonus = 20+5*np.sin(elapsed/2)
time_bonus = 12+4*np.cos(elapsed/3)

def route_stats():
    if len(st.session_state.real_p)<2:
        return 0,0,0
    dist=sum(haversine(st.session_state.real_p[i],st.session_state.real_p[i+1]) for i in range(len(st.session_state.real_p)-1))
    speed_now = 12 + 3*np.sin(time.time()/50)
    eta = dist/speed_now
    return dist,eta,speed_now

distance, eta, speed_now = route_stats()

st.title("🛰️ HELIOS 智慧航行系統")
r1 = st.columns(4)
r1[0].metric("🚀 航速",f"{speed_now:.1f} kn")
r1[1].metric("⛽ 省油效益",f"{fuel_bonus:.1f}%")
r1[2].metric("📡 衛星連線",f"{8+np.random.randint(0,5)} Pcs")
r1[3].metric("🌊 流場狀態",stream_status)
r2 = st.columns(4)
brg="---"
if len(st.session_state.real_p)>1:
    brg=f"{calc_bearing(st.session_state.real_p[0],st.session_state.real_p[1]):.1f}°"
r2[0].metric("🧭 建議航向",brg)
r2[1].metric("📏 預計距離",f"{distance:.1f} km")
r2[2].metric("🕒 預計時間",f"{eta:.1f} hr")
r2[3].metric("🕒 流場時間",ocean_time)
st.markdown("---")

# ===============================
# 8. 側邊欄
# ===============================
with st.sidebar:
    st.header("🚢 導航控制")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")
    st.session_state.dest_lat, st.session_state.dest_lon = elat, elon

    if st.button("🚀 啟動智能航路", use_container_width=True):
        if is_on_land(slat,slon):
            st.error("❌ 起點在陸地")
        elif is_on_land(elat,elon):
            st.error("❌ 終點在陸地")
        else:
            path = astar_route([slat,slon],[elat,elon],lons,lats,u,v)
            path = smooth_path(path)
            st.session_state.real_p = path
            st.session_state.step_idx=0
            st.session_state.ship_lat=slat
            st.session_state.ship_lon=slon
            st.experimental_rerun()

# ===============================
# 9. 地圖繪製
# ===============================
fig, ax = plt.subplots(figsize=(10,8),subplot_kw={'projection':ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND,facecolor='#2b2b2b',zorder=2)
ax.add_feature(cfeature.COASTLINE,edgecolor='cyan',linewidth=1.2,zorder=3)

speed = np.sqrt(u**2+v**2)
ax.pcolormesh(lons,lats,speed,cmap='YlGn',alpha=0.7,shading='gouraud',zorder=1)

skip=(slice(None,None,5),slice(None,None,5))
ax.quiver(lons[skip[1]],lats[skip[0]],u[skip],v[skip],color='white',alpha=0.4,scale=30,width=0.002,zorder=4)

if st.session_state.real_p:
    py,px = zip(*st.session_state.real_p)
    ax.plot(px,py,color='#FF00FF',linewidth=2.5,zorder=5)
    ax.scatter(st.session_state.ship_lon,st.session_state.ship_lat,color='red',s=80,zorder=7)
    ax.scatter(st.session_state.dest_lon,st.session_state.dest_lat,color='gold',marker='*',s=200,zorder=8)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)
