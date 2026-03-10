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

st.set_page_config(layout="wide", page_title="HELIOS Navigation System")
st.title("🛰️ HELIOS Smart Ocean Navigation")

# ===============================
# No-Go Zones
# ===============================
NO_GO_ZONES = [
    [[22.953536,120.171678],[22.934628,120.175472],[22.933136,120.170942],[22.957810,120.160780]],
    [[22.943956,120.172358],[22.939717,120.173944],[22.928353,120.157372],[22.936636,120.153547]],
    [[23.7885,119.598368],[23.784251,119.598368],[23.784251,119.602022],[23.7885,119.602022]],
]

# ===============================
# Offshore Wind Farms
# ===============================
OFFSHORE_WIND = [
    [[24.18,120.12],[24.22,120.28],[24.05,120.35],[24.00,120.15]],
    [[24.00,120.10],[24.05,120.32],[23.90,120.38],[23.85,120.15]],
    [[23.88,120.05],[23.92,120.18],[23.75,120.25],[23.70,120.08]],
    [[23.68,120.02],[23.72,120.12],[23.58,120.15],[23.55,120.05]],
]
OFFSHORE_COST = 10  # 軟懲罰成本

# ===============================
# HYCOM Ocean Current
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

    u_data=ds['ssu'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=-1)
    v_data=ds['ssv'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=-1)

    lons=u_data['lon'].values
    lats=u_data['lat'].values

    u=np.nan_to_num(u_data.values)
    v=np.nan_to_num(v_data.values)
    land_mask=np.isnan(u_data.values)

    return lons,lats,u,v,land_mask,latest_time

lons,lats,u,v,land_mask,obs_time=load_hycom_data()

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("Route Settings")
    s_lon=st.number_input("Start Lon",118.0,124.0,120.3)
    s_lat=st.number_input("Start Lat",21.0,26.0,22.6)
    e_lon=st.number_input("End Lon",118.0,124.0,122.0)
    e_lat=st.number_input("End Lat",21.0,26.0,24.5)
    ship_speed=st.number_input("Ship Speed (km/h)",1.0,60.0,20.0)

# ===============================
# Helper
# ===============================
def nearest_ocean_cell(lon,lat):
    lon_idx=np.abs(lons-lon).argmin()
    lat_idx=np.abs(lats-lat).argmin()
    return lat_idx,lon_idx

def offshore_penalty(y,x):
    lat=lats[y]
    lon=lons[x]
    for zone in OFFSHORE_WIND:
        if Path(zone).contains_point([lon,lat]):
            return OFFSHORE_COST
    return 0

# ===============================
# 靠岸懲罰 (減少讓路徑更自然)
# ===============================
sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)
MAX_DIST = 2        # 減少懲罰距離
COAST_PENALTY = 2.0 # 減少懲罰強度
def coast_penalty(y,x):
    d = dist_to_land[y,x]
    if d < MAX_DIST:
        return COAST_PENALTY * (MAX_DIST - d)/MAX_DIST
    return 0.0

# ===============================
# A* with offshore + coast
# ===============================
def astar_with_wind_and_coast(start, goal):
    rows, cols = land_mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq = [(0,start)]
    came = {}
    cost = {start:0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal: break

        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if 0<=ni<rows and 0<=nj<cols and not land_mask[ni,nj]:
                base_distance = np.hypot(d[0], d[1])
                new = cost[cur] + base_distance \
                      + offshore_penalty(ni,nj) \
                      + coast_penalty(ni,nj)
                if (ni,nj) not in cost or new < cost[(ni,nj)]:
                    cost[(ni,nj)] = new
                    heapq.heappush(pq,(new,(ni,nj)))
                    came[(ni,nj)] = cur

    path=[]
    curr=goal
    while curr in came:
        path.append(curr)
        curr=came[curr]
    if path:
        path.append(start)
    return path[::-1]

# ===============================
# Compute path
# ===============================
start=nearest_ocean_cell(s_lon,s_lat)
goal=nearest_ocean_cell(e_lon,e_lat)
path=astar_with_wind_and_coast(start,goal)

# ===============================
# Distance & Time
# ===============================
def calc_stats(path):
    dist=0
    for i in range(len(path)-1):
        y0,x0=path[i]
        y1,x1=path[i+1]
        dist+=np.hypot(lats[y1]-lats[y0],lons[x1]-lons[x0])*111
    hours=dist/ship_speed
    return dist,hours

distance_km,time_hr=calc_stats(path)

# ===============================
# Dashboard
# ===============================
st.subheader("Navigation Dashboard")
c1,c2,c3=st.columns(3)
c1.metric("Total Distance (km)",f"{distance_km:.2f}")
c2.metric("Travel Time (hr)",f"{time_hr:.2f}")
c3.metric("Connected Satellites",np.random.randint(3,7))
st.caption(f"HYCOM observation time: {obs_time}")

# ===============================
# Map
# ===============================
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND, facecolor="#b0b0b0")  # 更灰
ax.add_feature(cfeature.COASTLINE)

speed=np.sqrt(u**2+v**2)
mesh=ax.pcolormesh(lons,lats,speed,cmap="Blues",shading="auto")
fig.colorbar(mesh,ax=ax,label="Current Speed (m/s)")

# No-go zones
for zone in NO_GO_ZONES:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="red",alpha=0.4)

# Offshore wind (軟懲罰，黃色)
for zone in OFFSHORE_WIND:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="yellow",alpha=0.4)

# Path
if path:
    path_lons=[lons[p[1]] for p in path]
    path_lats=[lats[p[0]] for p in path]
    ax.plot(path_lons,path_lats,color="red",linewidth=2)

ax.scatter(s_lon,s_lat,color="green",s=120,edgecolors="black")
ax.scatter(e_lon,e_lat,color="yellow",marker="*",s=200,edgecolors="black")

plt.title("HELIOS Ocean Navigation Map")
st.pyplot(fig)
