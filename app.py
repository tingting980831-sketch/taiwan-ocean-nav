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
# Zones
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
    ds=xr.open_dataset(url,decode_times=False)

    if 'time_origin' in ds['time'].attrs:
        origin=pd.to_datetime(ds['time'].attrs['time_origin'])
        latest=origin+pd.to_timedelta(ds['time'].values[-1],unit='h')
    else:
        latest=pd.Timestamp.now()

    sub=ds.sel(lat=slice(21,26),lon=slice(118,124))
    lons=sub.lon.values
    lats=sub.lat.values
    land_mask=np.isnan(sub['ssu'].isel(time=0).values)

    return ds,lons,lats,land_mask,latest

ds,lons,lats,land_mask,obs_time=load_hycom_data()

sea_mask=~land_mask
dist_to_land=distance_transform_edt(sea_mask)

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Route Settings")
    s_lon=st.number_input("Start Lon",118.0,124.0,120.3)
    s_lat=st.number_input("Start Lat",21.0,26.0,22.6)
    e_lon=st.number_input("End Lon",118.0,124.0,122.0)
    e_lat=st.number_input("End Lat",21.0,26.0,24.5)
    ship_speed=st.number_input("Ship Speed (km/h)",1.0,60.0,20.0)

# ------------------------------
# Helpers
# ------------------------------
def nearest_ocean_cell(lon,lat):
    return np.abs(lats-lat).argmin(),np.abs(lons-lon).argmin()

def offshore_penalty(y,x):
    for zone in OFFSHORE_WIND:
        if Path(zone).contains_point([lons[x],lats[y]]):
            return OFFSHORE_COST
    return 0

def coast_penalty(y,x):
    d=dist_to_land[y,x]
    return 2*(2-d)/2 if d<2 else 0

# -------- TRUE SOG ----------
def sog_speed(y0,x0,y1,x1,ship_speed,time_idx):

    dx=lons[x1]-lons[x0]
    dy=lats[y1]-lats[y0]
    norm=np.hypot(dx,dy)
    if norm==0:
        return ship_speed

    dirx,diry=dx/norm,dy/norm

    u=ds['ssu'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=time_idx).values
    v=ds['ssv'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=time_idx).values

    flow=u[y0,x0]*dirx+v[y0,x0]*diry
    flow_kmh=flow*3.6

    return max(ship_speed+flow_kmh,ship_speed*0.3)

dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

# -------- Dynamic A* ----------
def astar_with_wind_and_coast(start,goal):

    rows,cols=land_mask.shape
    pq=[(0,start,0)]
    came={}
    cost={start:0}

    while pq:
        _,cur,t=heapq.heappop(pq)
        if cur==goal:
            break

        for d in dirs:
            ni,nj=cur[0]+d[0],cur[1]+d[1]
            if not(0<=ni<rows and 0<=nj<cols):continue
            if land_mask[ni,nj]:continue

            next_t=min(t+1,len(ds['time'])-1)

            dist=np.hypot(d[0],d[1])*111
            sog=sog_speed(cur[0],cur[1],ni,nj,ship_speed,next_t)

            travel_time=dist/sog

            new=cost[cur]+travel_time+offshore_penalty(ni,nj)+coast_penalty(ni,nj)

            if (ni,nj) not in cost or new<cost[(ni,nj)]:
                cost[(ni,nj)]=new
                heapq.heappush(pq,(new,(ni,nj),next_t))
                came[(ni,nj)]=cur

    path=[]
    curr=goal
    while curr in came:
        path.append(curr)
        curr=came[curr]
    if path:path.append(start)
    return path[::-1]

# ------------------------------
# Dynamic Replan Trigger
# ------------------------------
start=nearest_ocean_cell(s_lon,s_lat)
goal=nearest_ocean_cell(e_lon,e_lat)

if st.session_state.get("last_start")!=start or \
   st.session_state.get("last_goal")!=goal:

    st.session_state.full_path=astar_with_wind_and_coast(start,goal)
    st.session_state.ship_step_idx=0
    st.session_state.last_start=start
    st.session_state.last_goal=goal

path=st.session_state.full_path

# ------------------------------
# Remaining Distance
# ------------------------------
def calc_remaining(path,idx,speed):
    dist=0
    for i in range(idx,len(path)-1):
        y0,x0=path[i]
        y1,x1=path[i+1]
        dist+=np.hypot(lats[y1]-lats[y0],lons[x1]-lons[x0])*111

    if idx<len(path)-1:
        y0,x0=path[idx]
        y1,x1=path[idx+1]
        heading=np.degrees(np.arctan2(lats[y1]-lats[y0],lons[x1]-lons[x0]))
    else:
        heading=0

    return dist,dist/speed,heading

remaining_dist,remaining_time,heading=calc_remaining(
    path,
    st.session_state.ship_step_idx,
    ship_speed
)

# ------------------------------
# Dashboard
# ------------------------------
st.subheader("Navigation Dashboard")

c1,c2,c3,c4=st.columns(4)
c1.metric("Remaining Distance (km)",f"{remaining_dist:.2f}")
c2.metric("Remaining Time (hr)",f"{remaining_time:.2f}")
c3.metric("Heading (°)",f"{heading:.1f}")
c4.metric("Satellites",np.random.randint(3,13))

st.caption(f"HYCOM observation time: {obs_time}")

# ------------------------------
# Map
# ------------------------------
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND,facecolor="#b0b0b0")
ax.add_feature(cfeature.COASTLINE)

time_idx=min(st.session_state.ship_step_idx,len(ds['time'])-1)

u=ds['ssu'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=time_idx).values
v=ds['ssv'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=time_idx).values
speed=np.sqrt(u**2+v**2)

mesh=ax.pcolormesh(lons,lats,speed,cmap="Blues",shading="auto",vmin=0,vmax=1.6)
fig.colorbar(mesh,ax=ax,label="Current Speed (m/s)")

# zones
for z in NO_GO_ZONES:
    p=np.array(z)
    ax.fill(p[:,1],p[:,0],color="red",alpha=0.4)

for z in OFFSHORE_WIND:
    p=np.array(z)
    ax.fill(p[:,1],p[:,0],color="yellow",alpha=0.4)

# path
full_lons=[lons[p[1]] for p in path]
full_lats=[lats[p[0]] for p in path]
ax.plot(full_lons,full_lats,color="pink",linewidth=2)

done_lons=full_lons[:st.session_state.ship_step_idx+1]
done_lats=full_lats[:st.session_state.ship_step_idx+1]
ax.plot(done_lons,done_lats,color="red",linewidth=2)

cur=path[st.session_state.ship_step_idx]
ax.scatter(lons[cur[1]],lats[cur[0]],color="gray",s=150,marker="^")

ax.scatter(s_lon,s_lat,color="#B15BFF",s=80,edgecolors="black")
ax.scatter(e_lon,e_lat,color="yellow",marker="*",s=200,edgecolors="black")

plt.title("HELIOS System")
st.pyplot(fig)
