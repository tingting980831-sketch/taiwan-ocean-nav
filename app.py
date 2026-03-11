import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path
import xarray as xr

st.set_page_config(layout="wide")
st.title("🛰️ HELIOS System")

# =============================
# PARAMETERS
# =============================
CURRENT_WEIGHT = 0.3
MAX_ASSIST = 0.5
COAST_PENALTY = 2.0
MAX_DIST = 2

LAT_RANGE = slice(21,26)
LON_RANGE = slice(118,124)

# =============================
# MEMORY SAFE HYCOM LOAD
# =============================
@st.cache_data(ttl=3600)
def load_hycom_small():

    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"

    ds = xr.open_dataset(url, decode_times=False)

    # ⭐ 直接遠端裁切（關鍵）
    sub = ds.isel(time=-1).sel(lat=LAT_RANGE, lon=LON_RANGE)

    u = sub["ssu"].to_numpy()
    v = sub["ssv"].to_numpy()

    lats = sub.lat.to_numpy()
    lons = sub.lon.to_numpy()

    land_mask = np.isnan(u)

    return u,v,lats,lons,land_mask

u_field,v_field,lats,lons,land_mask = load_hycom_small()

sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)

# =============================
# SIDEBAR
# =============================
with st.sidebar:

    s_lon = st.number_input("Start Lon",118.0,124.0,120.3)
    s_lat = st.number_input("Start Lat",21.0,26.0,22.6)

    e_lon = st.number_input("End Lon",118.0,124.0,122.0)
    e_lat = st.number_input("End Lat",21.0,26.0,24.5)

    ship_speed = st.number_input("Ship Speed km/h",1.0,60.0,20.0)

    recompute = st.button("Recalculate")
    next_step = st.button("Next Step")

# =============================
# HELPERS
# =============================
def nearest(lon,lat):
    return np.abs(lats-lat).argmin(), np.abs(lons-lon).argmin()

def coast_penalty(y,x):
    d=dist_to_land[y,x]
    if d<MAX_DIST:
        return COAST_PENALTY*(MAX_DIST-d)/MAX_DIST
    return 0

def travel_time(y0,x0,y1,x1):

    dy=lats[y1]-lats[y0]
    dx=lons[x1]-lons[x0]
    dist=np.hypot(dx,dy)*111

    move=np.array([dx,dy])
    n=np.linalg.norm(move)
    if n==0:
        return dist

    move/=n
    current=np.array([u_field[y1,x1],v_field[y1,x1]])

    assist=np.clip(np.dot(move,current),-MAX_ASSIST,MAX_ASSIST)

    eff_speed=max(ship_speed/3.6 + CURRENT_WEIGHT*assist,0.3)

    return dist/(eff_speed*3.6)

# =============================
# ASTAR
# =============================
dirs=[(1,0),(-1,0),(0,1),(0,-1),
      (1,1),(1,-1),(-1,1),(-1,-1)]

def astar(start,goal):

    rows,cols=land_mask.shape
    pq=[(0,start)]
    came={}
    cost={start:0}

    while pq:

        _,cur=heapq.heappop(pq)

        if cur==goal:
            break

        for d in dirs:

            ni,nj=cur[0]+d[0],cur[1]+d[1]

            if 0<=ni<rows and 0<=nj<cols and not land_mask[ni,nj]:

                new=cost[cur]+travel_time(*cur,ni,nj)+coast_penalty(ni,nj)

                if (ni,nj) not in cost or new<cost[(ni,nj)]:
                    cost[(ni,nj)]=new
                    heapq.heappush(pq,(new,(ni,nj)))
                    came[(ni,nj)]=cur

    path=[]
    cur=goal
    while cur in came:
        path.append(cur)
        cur=came[cur]

    path.append(start)
    return path[::-1]

# =============================
# ROUTE
# =============================
start=nearest(s_lon,s_lat)
goal=nearest(e_lon,e_lat)

if recompute or "path" not in st.session_state:
    st.session_state.path=astar(start,goal)
    st.session_state.idx=0

if next_step and st.session_state.idx<len(st.session_state.path)-1:
    st.session_state.idx+=1

# =============================
# MAP
# =============================
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)

speed=np.sqrt(u_field**2+v_field**2)

mesh=ax.pcolormesh(lons,lats,speed,cmap="Blues",shading="auto")
fig.colorbar(mesh,ax=ax,label="Current m/s")

path=st.session_state.path
plon=[lons[p[1]] for p in path]
plat=[lats[p[0]] for p in path]

ax.plot(plon,plat,color="pink",linewidth=2)

idx=st.session_state.idx
ax.plot(plon[:idx+1],plat[:idx+1],color="red",linewidth=2)

y,x=path[idx]
ax.scatter(lons[x],lats[y],color="gray",s=150,marker="^")

st.pyplot(fig)
