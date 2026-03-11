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
# Load HYCOM (LATEST ONLY)
# ------------------------------
@st.cache_data(ttl=1800)
def load_hycom_data():

    url = (
        "https://tds.hycom.org/thredds/dodsC/"
        "FMRC_ESPC-D-V02_uv3z/"
        "FMRC_ESPC-D-V02_uv3z_best.ncd"
    )

    ds = xr.open_dataset(url, engine="netcdf4")

    sub = ds.isel(time=-1).sel(
        lat=slice(21,26),
        lon=slice(118,124)
    )

    u = sub["water_u"].values
    v = sub["water_v"].values

    lons = sub.lon.values
    lats = sub.lat.values

    land_mask = np.isnan(u)

    obs_time = pd.Timestamp.now()

    return u, v, lons, lats, land_mask, obs_time


u_field, v_field, lons, lats, land_mask, obs_time = load_hycom_data()

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

    st.button("Next Step", key="next_step_button")

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

MAX_DIST=2
COAST_PENALTY=2.0

def coast_penalty(y,x):
    d=dist_to_land[y,x]
    if d<MAX_DIST:
        return COAST_PENALTY*(MAX_DIST-d)/MAX_DIST
    return 0

# ------------------------------
# Ocean-aware travel cost
# ------------------------------
CURRENT_WEIGHT=0.35
MAX_ASSIST=0.6

def travel_cost(y0,x0,y1,x1):

    dy=lats[y1]-lats[y0]
    dx=lons[x1]-lons[x0]

    dist_km=np.hypot(dx,dy)*111

    move=np.array([dx,dy])
    n=np.linalg.norm(move)
    if n==0:
        return dist_km

    move/=n

    current=np.array([
        u_field[y1,x1],
        v_field[y1,x1]
    ])

    assist=np.dot(move,current)
    assist=np.clip(assist,-MAX_ASSIST,MAX_ASSIST)

    eff_speed=max(
        ship_speed/3.6 + CURRENT_WEIGHT*assist,
        0.4
    )

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

            ni, nj = cur[0]+d[0], cur[1]+d[1]

            if 0<=ni<rows and 0<=nj<cols and not land_mask[ni,nj]:

                move_cost=travel_cost(cur[0],cur[1],ni,nj)

                new_cost=(
                    cost[cur]
                    + move_cost
                    + offshore_penalty(ni,nj)
                    + coast_penalty(ni,nj)
                )

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

# ------------------------------
# Dynamic recompute
# ------------------------------
start=nearest_ocean_cell(s_lon,s_lat)
goal=nearest_ocean_cell(e_lon,e_lat)

if (
    "full_path" not in st.session_state
    or st.session_state.get("last_start")!=start
    or st.session_state.get("last_goal")!=goal
):

    st.session_state.full_path=astar(start,goal)
    st.session_state.ship_step_idx=0
    st.session_state.last_start=start
    st.session_state.last_goal=goal

# ------------------------------
# Ship movement
# ------------------------------
if st.session_state.get("next_step_button",False):
    if st.session_state.ship_step_idx < len(st.session_state.full_path)-1:
        st.session_state.ship_step_idx+=1

path=st.session_state.full_path
step=st.session_state.ship_step_idx

# ------------------------------
# Remaining distance/time
# ------------------------------
def calc_remaining():

    dist=0
    for i in range(step,len(path)-1):
        y0,x0=path[i]
        y1,x1=path[i+1]
        dist+=np.hypot(lats[y1]-lats[y0],lons[x1]-lons[x0])*111

    if step<len(path)-1:
        y0,x0=path[step]
        y1,x1=path[step+1]
        heading=np.degrees(np.arctan2(
            lats[y1]-lats[y0],
            lons[x1]-lons[x0]
        ))
    else:
        heading=0

    return dist, dist/ship_speed, heading

remaining_dist,remaining_time,heading_deg=calc_remaining()

# ------------------------------
# Dashboard
# ------------------------------
st.subheader("Navigation Dashboard")

c1,c2,c3=st.columns(3)
c1.metric("Remaining Distance (km)",f"{remaining_dist:.2f}")
c2.metric("Remaining Time (hr)",f"{remaining_time:.2f}")
c3.metric("Heading (°)",f"{heading_deg:.1f}")

st.caption(f"HYCOM observation time: {obs_time}")

# ------------------------------
# Map
# ------------------------------
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND,facecolor="#b0b0b0")
ax.add_feature(cfeature.COASTLINE)

speed=np.sqrt(u_field**2+v_field**2)

mesh=ax.pcolormesh(
    lons,lats,speed,
    cmap="Blues",
    shading="auto",
    vmin=0,vmax=1.6
)

fig.colorbar(mesh,ax=ax,label="Current Speed (m/s)")

# path
full_lons=[lons[p[1]] for p in path]
full_lats=[lats[p[0]] for p in path]

ax.plot(full_lons,full_lats,color="pink",linewidth=2)

done_lons=full_lons[:step+1]
done_lats=full_lats[:step+1]

ax.plot(done_lons,done_lats,color="red",linewidth=2)

current_pos=path[step]

ax.scatter(
    lons[current_pos[1]],
    lats[current_pos[0]],
    color="gray",
    s=150,
    marker="^"
)

ax.scatter(s_lon,s_lat,color="#B15BFF",s=80,edgecolors="black")
ax.scatter(e_lon,e_lat,color="yellow",marker="*",s=200,edgecolors="black")

plt.title("HELIOS System")
st.pyplot(fig)
