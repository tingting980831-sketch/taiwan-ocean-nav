import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
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

# ===============================
# 2. 讀取 HYCOM 流場
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
        speed = np.sqrt(u**2+v**2)
        mask = (speed==0) | (speed>5) | np.isnan(speed)
        u[mask] = np.nan
        v[mask] = np.nan
        return lons, lats, u, v, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        return None, None, None, None, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_ocean_data()

# ===============================
# 3. 陸地判斷
# ===============================
def is_on_land(lat, lon):
    return (21.8<=lat<=25.4 and 120.0<=lon<=122.1) or (lat>=24.3 and lon<=119.6)

# ===============================
# 4. Haversine & 航向
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
# 5. 智慧航線生成 (A* 考慮流場)
# ===============================
def smart_route(start,end):
    # 格點網格
    lat_grid, lon_grid = lats, lons
    u_grid, v_grid = u, v
    nlat, nlon = len(lat_grid), len(lon_grid)

    # 將陸地設為極高成本
    cost_grid = np.ones((nlat,nlon))
    for i in range(nlat):
        for j in range(nlon):
            if is_on_land(lat_grid[i], lon_grid[j]):
                cost_grid[i,j] = 1e6

    # 找最近格點索引
    def nearest_idx(lat, lon):
        i = np.argmin(np.abs(lat_grid-lat))
        j = np.argmin(np.abs(lon_grid-lon))
        return i,j

    si,sj = nearest_idx(*start)
    ei,ej = nearest_idx(*end)

    # A* 演算法
    heap = [(0, si, sj, [])]  # (f_score, i, j, path)
    visited = set()
    while heap:
        f,i,j,path = heapq.heappop(heap)
        if (i,j) in visited: continue
        visited.add((i,j))
        path = path + [[lat_grid[i], lon_grid[j]]]
        if (i,j)==(ei,ej):
            # 平滑路線
            smooth=[]
            for k in range(len(path)-1):
                for t in np.linspace(0,1,6):
                    lat = path[k][0] + (path[k+1][0]-path[k][0])*t
                    lon = path[k][1] + (path[k+1][1]-path[k][1])*t
                    smooth.append([lat,lon])
            return smooth,None

        # 8 鄰格 (可斜向)
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni,nj = i+di, j+dj
                if 0<=ni<nlat and 0<=nj<nlon and (ni,nj) not in visited:
                    # 成本 = 格距 / (船速+流速順流加成)
                    dist = haversine([lat_grid[i],lon_grid[j]], [lat_grid[ni],lon_grid[nj]])
                    # 流速加成
                    if np.isnan(u_grid[ni,nj]) or np.isnan(v_grid[ni,nj]):
                        continue
                    brg = calc_bearing([lat_grid[i],lon_grid[j]], [lat_grid[ni],lon_grid[nj]])
                    curr_speed = 12 + (u_grid[ni,nj]*np.cos(np.radians(brg)) + v_grid[ni,nj]*np.sin(np.radians(brg)))
                    curr_speed = max(curr_speed,0.5)
                    total_cost = dist/curr_speed + cost_grid[ni,nj]
                    heapq.heappush(heap,(f+total_cost, ni,nj, path))
    return None,"找不到安全航線"

# ===============================
# 6. Sidebar (座標)
# ===============================
with st.sidebar:
    st.header("🚢 導航控制中心")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    if st.button("🚀 啟動智慧航路", use_container_width=True):
        path,msg = smart_route([slat,slon],[elat,elon])
        if msg: st.error(msg)
        else:
            st.session_state.real_p = path
            st.session_state.step_idx = 0
            st.session_state.ship_lat = slat
            st.session_state.ship_lon = slon
            st.session_state.dest_lat = elat
            st.session_state.dest_lon = elon
            st.experimental_rerun()

# ===============================
# 7. 地圖繪製
# ===============================
fig,ax=plt.subplots(figsize=(10,8),subplot_kw={'projection':ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND,facecolor="#2b2b2b")
ax.add_feature(cfeature.COASTLINE,color="cyan")

if u is not None:
    speed = np.sqrt(u**2+v**2)
    ax.pcolormesh(lons,lats,speed,cmap='YlGn',alpha=0.7,shading='gouraud')
    skip = (slice(None,None,5), slice(None,None,5))
    ax.quiver(lons[skip[1]],lats[skip[0]],u[skip],v[skip],color='white',alpha=0.4,scale=30,width=0.002)

if st.session_state.real_p:
    py,px = zip(*st.session_state.real_p)
    ax.plot(px,py,color='#FF00FF',linewidth=2.5)
    ax.scatter(st.session_state.ship_lon,st.session_state.ship_lat,color='red',s=80)
    ax.scatter(st.session_state.dest_lon,st.session_state.dest_lat,color='gold',marker='*',s=200)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)
