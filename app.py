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
st.title("🛰️ HELIOS Dynamic Ocean Navigation")

# ===============================
# No-Go Zones
# ===============================
NO_GO_ZONES = [
    [[22.953536,120.171678],[22.934628,120.175472],[22.933136,120.170942],[22.95781,120.16078]],
    [[22.943956,120.172358],[22.939717,120.173944],[22.928353,120.157372],[22.936636,120.153547]],
    [[22.933136,120.170942],[22.924847,120.172583],[22.915003,120.159022],[22.931536,120.155772]],
]

# ===============================
# Offshore Wind
# ===============================
OFFSHORE_WIND = [
    [[24.18,120.12],[24.22,120.28],[24.05,120.35],[24.00,120.15]],
]

OFFSHORE_COST = 10

# ===============================
# Load HYCOM Data
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():

    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds=xr.open_dataset(url,decode_times=False)

    if 'time_origin' in ds['time'].attrs:
        time_origin=pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time=time_origin+pd.to_timedelta(ds['time'].values[-1],unit='h')
    else:
        latest_time=pd.Timestamp.now()

    subset=ds['ssu'].sel(lat=slice(21,26),lon=slice(118,124))

    lons=subset.lon.values
    lats=subset.lat.values
    land_mask=np.isnan(subset.isel(time=0).values)

    return ds,lons,lats,land_mask,latest_time

ds,lons,lats,land_mask,obs_time=load_hycom_data()

sea_mask=~land_mask
dist_to_land=distance_transform_edt(sea_mask)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("Route Settings")

    s_lon=st.number_input("Start Longitude",118.0,124.0,120.3)
    s_lat=st.number_input("Start Latitude",21.0,26.0,22.6)

    e_lon=st.number_input("End Longitude",118.0,124.0,122.0)
    e_lat=st.number_input("End Latitude",21.0,26.0,24.5)

    ship_speed=st.number_input("Ship Speed (km/h)",1.0,60.0,20.0)

    next_step=st.button("Next Step")

# ===============================
# Helper
# ===============================
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

MAX_DIST=2
COAST_PENALTY=2.0

def coast_penalty(y,x):
    d=dist_to_land[y,x]
    if d<MAX_DIST:
        return COAST_PENALTY*(MAX_DIST-d)/MAX_DIST
    return 0.0

# ===============================
# Dynamic A*
# ===============================
def astar_with_cost(start,goal):

    rows,cols=land_mask.shape
    dirs=[(1,0),(-1,0),(0,1),(0,-1),
          (1,1),(1,-1),(-1,1),(-1,-1)]

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

                new_cost=cost[cur]+1
                new_cost+=coast_penalty(ni,nj)
                new_cost+=offshore_penalty(ni,nj)

                if (ni,nj) not in cost or new_cost<cost[(ni,nj)]:
                    cost[(ni,nj)]=new_cost
                    heapq.heappush(pq,(new_cost,(ni,nj)))
                    came[(ni,nj)]=cur

    path=[]
    curr=goal
    while curr in came:
        path.append(curr)
        curr=came[curr]

    if path:
        path.append(start)

    return path[::-1]

# ===============================
# Initialize Path
# ===============================
start=nearest_ocean_cell(s_lon,s_lat)
goal=nearest_ocean_cell(e_lon,e_lat)

if "full_path" not in st.session_state:
    st.session_state.full_path=astar_with_cost(start,goal)

if "ship_idx" not in st.session_state:
    st.session_state.ship_idx=0

if next_step and st.session_state.ship_idx<len(st.session_state.full_path)-1:
    st.session_state.ship_idx+=1

# ===============================
# Remaining Distance
# ===============================
def calc_remaining(path,idx):

    dist=0
    for i in range(idx,len(path)-1):
        y0,x0=path[i]
        y1,x1=path[i+1]
        dist+=np.hypot(lats[y1]-lats[y0],lons[x1]-lons[x0])*111

    time=dist/ship_speed

    if idx<len(path)-1:
        y0,x0=path[idx]
        y1,x1=path[idx+1]
        heading=np.degrees(np.arctan2(
            lats[y1]-lats[y0],
            lons[x1]-lons[x0]))
    else:
        heading=0

    return dist,time,heading

remaining_dist,remaining_time,heading_deg=calc_remaining(
    st.session_state.full_path,
    st.session_state.ship_idx
)

# ===============================
# Dashboard
# ===============================
st.subheader("Navigation Dashboard")

c1,c2,c3=st.columns(3)
c1.metric("Remaining Distance (km)",f"{remaining_dist:.2f}")
c2.metric("Remaining Time (hr)",f"{remaining_time:.2f}")
c3.metric("Heading (°)",f"{heading_deg:.1f}")

st.caption(f"HYCOM observation time: {obs_time}")

# ===============================
# Map
# ===============================
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND,facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE)

time_idx=st.session_state.ship_idx
if time_idx>=len(ds['time']):
    time_idx=-1

u=ds['ssu'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=time_idx).values
v=ds['ssv'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=time_idx).values

speed=np.sqrt(u**2+v**2)

mesh=ax.pcolormesh(lons,lats,speed,cmap="Blues",
                   shading="auto",vmin=0,vmax=1.6)

fig.colorbar(mesh,ax=ax,label="Current Speed (m/s)")

# zones
for zone in NO_GO_ZONES:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="red",alpha=0.4)

for zone in OFFSHORE_WIND:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="yellow",alpha=0.4)

# path
full_lons=[lons[p[1]] for p in st.session_state.full_path]
full_lats=[lats[p[0]] for p in st.session_state.full_path]

ax.plot(full_lons,full_lats,color="pink",linewidth=2)

done_lons=full_lons[:st.session_state.ship_idx+1]
done_lats=full_lats[:st.session_state.ship_idx+1]

ax.plot(done_lons,done_lats,color="red",linewidth=2)

current=st.session_state.full_path[st.session_state.ship_idx]
ax.scatter(lons[current[1]],lats[current[0]],
           color="gray",marker="^",s=150)

ax.scatter(s_lon,s_lat,color="purple",s=80,edgecolors="black")
ax.scatter(e_lon,e_lat,color="yellow",marker="*",s=200,edgecolors="black")

plt.title("HELIOS Dynamic Navigation")
st.pyplot(fig)
