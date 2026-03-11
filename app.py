import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(layout="wide", page_title="HELIOS System")
st.title("🛰️ HELIOS System")

# ------------------------------
# Parameters (IMPORTANT TUNING)
# ------------------------------
CURRENT_WEIGHT = 0.3   # 海流影響程度
MAX_ASSIST = 0.5       # 防止繞遠路
COAST_PENALTY = 2.0
MAX_DIST = 2

# ------------------------------
# No-Go Zones & Offshore Wind
# ------------------------------
NO_GO_ZONES = [
    [[22.953536,120.171678],[22.934628,120.175472],[22.933136,120.170942],[22.95781,120.16078]],
]

OFFSHORE_WIND = [
    [[24.18,120.12],[24.22,120.28],[24.05,120.35],[24.00,120.15]],
]
OFFSHORE_COST = 10

# ------------------------------
# Load HYCOM
# ------------------------------
@st.cache_data(ttl=3600)
def load_hycom_data():
    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)

    if 'time_origin' in ds['time'].attrs:
        t0 = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = t0 + pd.to_timedelta(ds['time'].values[-1], unit='h')
    else:
        latest_time = pd.Timestamp.now()

    sub = ds[['ssu','ssv']].sel(lat=slice(21,26),lon=slice(118,124))

    lons = sub.lon.values
    lats = sub.lat.values
    land_mask = np.isnan(sub['ssu'].isel(time=0).values)

    return sub, lons, lats, land_mask, latest_time

ds, lons, lats, land_mask, obs_time = load_hycom_data()

sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Route Settings")

    s_lon = st.number_input("Start Longitude",118.0,124.0,120.3)
    s_lat = st.number_input("Start Latitude",21.0,26.0,22.6)
    e_lon = st.number_input("End Longitude",118.0,124.0,122.0)
    e_lat = st.number_input("End Latitude",21.0,26.0,24.5)
    ship_speed = st.number_input("Ship Speed (km/h)",1.0,60.0,20.0)

    recompute = st.button("🔄 Recalculate Route")
    next_step = st.button("Next Step")

# ------------------------------
# Helpers
# ------------------------------
def nearest_ocean_cell(lon,lat):
    lon_idx=np.abs(lons-lon).argmin()
    lat_idx=np.abs(lats-lat).argmin()
    return lat_idx,lon_idx

def offshore_penalty(y,x):
    lat_val=lats[y]
    lon_val=lons[x]
    for zone in OFFSHORE_WIND:
        if Path(zone).contains_point([lon_val,lat_val]):
            return OFFSHORE_COST
    return 0

def coast_penalty(y,x):
    d=dist_to_land[y,x]
    if d<MAX_DIST:
        return COAST_PENALTY*(MAX_DIST-d)/MAX_DIST
    return 0

# ⭐ 海流時間成本（核心）
def travel_time_cost(y0,x0,y1,x1,time_idx):

    u = ds['ssu'].isel(time=time_idx).values
    v = ds['ssv'].isel(time=time_idx).values

    dy=lats[y1]-lats[y0]
    dx=lons[x1]-lons[x0]

    dist_km=np.hypot(dy,dx)*111

    move_vec=np.array([dx,dy])
    norm=np.linalg.norm(move_vec)
    if norm==0:
        return dist_km

    move_unit=move_vec/norm
    current_vec=np.array([u[y1,x1],v[y1,x1]])

    assist=np.dot(move_unit,current_vec)
    assist=np.clip(assist,-MAX_ASSIST,MAX_ASSIST)

    eff_speed=ship_speed/3.6 + CURRENT_WEIGHT*assist
    eff_speed=max(eff_speed,0.3)

    return dist_km/(eff_speed*3.6)

# ------------------------------
# A* Routing
# ------------------------------
dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

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

                time_cost=travel_time_cost(cur[0],cur[1],ni,nj,0)

                new=cost[cur] \
                    + time_cost \
                    + offshore_penalty(ni,nj) \
                    + coast_penalty(ni,nj)

                if (ni,nj) not in cost or new<cost[(ni,nj)]:
                    cost[(ni,nj)]=new
                    heapq.heappush(pq,(new,(ni,nj)))
                    came[(ni,nj)]=cur

    path=[]
    curr=goal
    while curr in came:
        path.append(curr)
        curr=came[curr]
    if path:
        path.append(start)

    return path[::-1]

# ------------------------------
# Route Compute
# ------------------------------
start=nearest_ocean_cell(s_lon,s_lat)
goal=nearest_ocean_cell(e_lon,e_lat)

if recompute or "full_path" not in st.session_state:
    st.session_state.full_path=astar(start,goal)
    st.session_state.ship_step_idx=0

# ------------------------------
# Move Ship
# ------------------------------
if next_step:
    if st.session_state.ship_step_idx < len(st.session_state.full_path)-1:
        st.session_state.ship_step_idx+=1

# ------------------------------
# Remaining Distance
# ------------------------------
def calc_remaining(path,idx):
    dist=0
    for i in range(idx,len(path)-1):
        y0,x0=path[i]
        y1,x1=path[i+1]
        dist+=np.hypot(lats[y1]-lats[y0],lons[x1]-lons[x0])*111
    return dist

remaining_dist=calc_remaining(
    st.session_state.full_path,
    st.session_state.ship_step_idx
)

remaining_time=remaining_dist/ship_speed

# ------------------------------
# Dashboard
# ------------------------------
st.subheader("Navigation Dashboard")

c1,c2=st.columns(2)
c1.metric("Remaining Distance (km)",f"{remaining_dist:.2f}")
c2.metric("Remaining Time (hr)",f"{remaining_time:.2f}")

st.caption(f"HYCOM observation time: {obs_time}")

# ------------------------------
# Map
# ------------------------------
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND,facecolor="#b0b0b0")
ax.add_feature(cfeature.COASTLINE)

time_idx=0
u=ds['ssu'].isel(time=time_idx).values
v=ds['ssv'].isel(time=time_idx).values
speed=np.sqrt(u**2+v**2)

mesh=ax.pcolormesh(lons,lats,speed,
                   cmap="Blues",
                   shading="auto",
                   vmin=0,vmax=1.6)

fig.colorbar(mesh,ax=ax,label="Current Speed (m/s)")

# Path
path=st.session_state.full_path
full_lons=[lons[p[1]] for p in path]
full_lats=[lats[p[0]] for p in path]

ax.plot(full_lons,full_lats,color="pink",linewidth=2)

idx=st.session_state.ship_step_idx
ax.plot(full_lons[:idx+1],full_lats[:idx+1],
        color="red",linewidth=2)

current_pos=path[idx]

ax.scatter(lons[current_pos[1]],
           lats[current_pos[0]],
           color="gray",s=150,marker="^")

ax.scatter(s_lon,s_lat,color="#B15BFF",s=80,edgecolors="black")
ax.scatter(e_lon,e_lat,color="yellow",marker="*",s=200,edgecolors="black")

plt.title("HELIOS System")
st.pyplot(fig)
