import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

# ===============================
# Page
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS System")
st.title("🛰️ HELIOS System")

# ===============================
# No-Go Zones
# ===============================
NO_GO_ZONES = [
    [[22.953536,120.171678],[22.934628,120.175472],[22.933136,120.170942],[22.95781,120.16078]],
    [[22.943956,120.172358],[22.939717,120.173944],[22.928353,120.157372],[22.936636,120.153547]],
    [[22.933136,120.170942],[22.924847,120.172583],[22.915003,120.159022],[22.931536,120.155772]],
]

OFFSHORE_WIND = [
    [[24.18,120.12],[24.22,120.28],[24.05,120.35],[24.00,120.15]],
    [[24.00,120.10],[24.05,120.32],[23.90,120.38],[23.85,120.15]],
    [[23.88,120.05],[23.92,120.18],[23.75,120.25],[23.70,120.08]],
    [[23.68,120.02],[23.72,120.12],[23.58,120.15],[23.55,120.05]],
]

# ===============================
# Load HYCOM (REALTIME)
# ===============================
@st.cache_data(ttl=1800)
def load_hycom():

    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest"

    ds = xr.open_dataset(
        url,
        engine="pydap",
        decode_times=False
    )

    latest_time = float(ds.time.values[-1])

    ds_time = ds.sel(time=latest_time)

    sub = ds_time.sel(
        lat=slice(21,27),
        lon=slice(117,125)
    )

    lons = sub.lon.values
    lats = sub.lat.values

    u_latest = sub["water_u"][0,:,:].values
    land_mask = np.isnan(u_latest)

    # HYCOM epoch
    base_time = pd.Timestamp("2000-01-01")
    obs_time = base_time + pd.to_timedelta(latest_time, unit="h")

    return sub, lons, lats, land_mask, obs_time


ds_sub, lons, lats, land_mask, obs_time = load_hycom()

# ===============================
# Distance from land
# ===============================
sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("Route Settings")

    s_lon = st.number_input("Start Lon",118.0,124.0,120.3)
    s_lat = st.number_input("Start Lat",21.0,26.0,22.6)

    e_lon = st.number_input("End Lon",118.0,124.0,122.0)
    e_lat = st.number_input("End Lat",21.0,26.0,24.5)

    ship_speed = st.number_input("Ship Speed (km/h)",1.0,60.0,20.0)

    if st.button("Next Step"):
        st.session_state.ship_step_idx += 1

# ===============================
# Helpers
# ===============================
def nearest_cell(lon, lat):
    return (
        np.abs(lats-lat).argmin(),
        np.abs(lons-lon).argmin()
    )

def astar(start, goal):

    rows, cols = land_mask.shape
    pq=[(0,start)]
    came={}
    cost={start:0}

    dirs=[(1,0),(-1,0),(0,1),(0,-1),
          (1,1),(1,-1),(-1,1),(-1,-1)]

    while pq:
        _,cur=heapq.heappop(pq)
        if cur==goal: break

        for d in dirs:
            ni,nj=cur[0]+d[0],cur[1]+d[1]

            if 0<=ni<rows and 0<=nj<cols and not land_mask[ni,nj]:

                penalty=5 if dist_to_land[ni,nj]<2 else 0
                new_cost=cost[cur]+np.hypot(d[0],d[1])+penalty

                if (ni,nj) not in cost or new_cost<cost[(ni,nj)]:
                    cost[(ni,nj)]=new_cost
                    came[(ni,nj)]=cur
                    priority=new_cost+np.hypot(ni-goal[0],nj-goal[1])
                    heapq.heappush(pq,(priority,(ni,nj)))

    path=[]
    curr=goal
    while curr in came:
        path.append(curr)
        curr=came[curr]

    return path[::-1]

# ===============================
# Route Update
# ===============================
start_node = nearest_cell(s_lon,s_lat)
goal_node  = nearest_cell(e_lon,e_lat)
route_key=(s_lon,s_lat,e_lon,e_lat)

if "route_key" not in st.session_state or st.session_state.route_key!=route_key:
    st.session_state.full_path=astar(start_node,goal_node)
    st.session_state.ship_step_idx=0
    st.session_state.route_key=route_key

path=st.session_state.full_path
st.session_state.ship_step_idx=min(
    st.session_state.ship_step_idx,
    len(path)-1
)

current_pos=path[st.session_state.ship_step_idx]

# ===============================
# Dashboard
# ===============================
st.subheader("Navigation Dashboard")

dist_rem=(len(path)-st.session_state.ship_step_idx)*0.08*111

c1,c2,c3=st.columns(3)
c1.metric("Remaining Distance (km)",f"{dist_rem:.2f}")
c2.metric("Remaining Time (hr)",f"{dist_rem/ship_speed:.2f}")
c3.metric("Satellite Status","Connected (12)")

st.caption(f"HYCOM Latest Analysis Time: {obs_time:%Y-%m-%d %H:%M UTC}")

# ===============================
# Map
# ===============================
fig=plt.figure(figsize=(12,9))
ax=plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118,124,21,26])

# currents
u_plot=ds_sub["water_u"][0,:,:]
v_plot=ds_sub["water_v"][0,:,:]
speed_plot=np.sqrt(u_plot**2+v_plot**2)

speed_plot.plot.pcolormesh(
    ax=ax,
    x="lon",
    y="lat",
    transform=ccrs.PlateCarree(),
    cmap="Blues",
    vmin=0,
    vmax=1.6,
    add_colorbar=True,
    cbar_kwargs={"label":"Current Speed (m/s)"}
)

ax.add_feature(cfeature.LAND,facecolor="#b0b0b0")
ax.add_feature(cfeature.COASTLINE)

# zones
for z in NO_GO_ZONES:
    ax.fill([p[1] for p in z],[p[0] for p in z],
            color="red",alpha=0.4,transform=ccrs.PlateCarree())

for z in OFFSHORE_WIND:
    ax.fill([p[1] for p in z],[p[0] for p in z],
            color="yellow",alpha=0.4,transform=ccrs.PlateCarree())

# route
if len(path)>0:
    f_lons=[lons[p[1]] for p in path]
    f_lats=[lats[p[0]] for p in path]

    ax.plot(f_lons,f_lats,color="pink",linewidth=3,
            transform=ccrs.PlateCarree())

    idx=st.session_state.ship_step_idx

    ax.plot(f_lons[:idx+1],f_lats[:idx+1],
            color="red",linewidth=3,
            transform=ccrs.PlateCarree())

    ax.scatter(f_lons[idx],f_lats[idx],
               color="gray",marker="^",s=200,
               transform=ccrs.PlateCarree())

# start/end
ax.scatter(s_lon,s_lat,color="#B15BFF",s=100,
           transform=ccrs.PlateCarree())
ax.scatter(e_lon,e_lat,color="yellow",marker="*",s=300,
           transform=ccrs.PlateCarree())

plt.title(f"HELIOS System | Data Time: {obs_time}")
st.pyplot(fig)
