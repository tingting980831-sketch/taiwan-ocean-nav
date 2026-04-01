import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq

st.set_page_config(layout="wide")
st.title("🛰️ HELIOS Dynamic Ocean Navigation (Stable)")

# =====================================================
# HYCOM LOADER  (RESOURCE CACHE — 不可 hash)
# =====================================================

@st.cache_resource(show_spinner="Loading HYCOM...")
def load_hycom():

    url = (
        "https://tds.hycom.org/thredds/dodsC/"
        "GLBy0.08/latest"
    )

    ds = xr.open_dataset(
        url,
        engine="pydap",
        decode_times=False
    )

    lons = ds["lon"].values
    lats = ds["lat"].values

    # land mask (NaN = land)
    sample = ds["water_u"][0,0,:,:].values
    land_mask = np.isnan(sample)

    obs_time = float(ds["time"].values[-1])

    return ds, lons, lats, land_mask, obs_time


ds, lons, lats, land_mask, obs_time = load_hycom()

# =====================================================
# CURRENT EXTRACTION (DATA CACHE — 可 hash)
# =====================================================

@st.cache_data(show_spinner="Extracting currents...")
def get_valid_currents():

    u = ds["water_u"][-1,0,:,:].values
    v = ds["water_v"][-1,0,:,:].values

    u = np.nan_to_num(u)
    v = np.nan_to_num(v)

    speed = np.sqrt(u**2 + v**2)

    return u, v, speed


u, v, speed = get_valid_currents()

# =====================================================
# A* PATHFINDING
# =====================================================

def heuristic(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])


def astar(cost, start, goal):

    neighbors = [
        (1,0),(-1,0),(0,1),(0,-1),
        (1,1),(1,-1),(-1,1),(-1,-1)
    ]

    close_set=set()
    came_from={}
    gscore={start:0}
    fscore={start:heuristic(start,goal)}

    oheap=[]
    heapq.heappush(oheap,(fscore[start],start))

    while oheap:

        current=heapq.heappop(oheap)[1]

        if current==goal:
            data=[]
            while current in came_from:
                data.append(current)
                current=came_from[current]
            return data[::-1]

        close_set.add(current)

        for i,j in neighbors:
            neighbor=(current[0]+i,current[1]+j)

            if not (
                0<=neighbor[0]<cost.shape[0]
                and 0<=neighbor[1]<cost.shape[1]
            ):
                continue

            if land_mask[neighbor]:
                continue

            tentative=gscore[current]+cost[neighbor]

            if neighbor in close_set and tentative>=gscore.get(neighbor,0):
                continue

            if tentative<gscore.get(neighbor,np.inf):
                came_from[neighbor]=current
                gscore[neighbor]=tentative
                fscore[neighbor]=tentative+heuristic(neighbor,goal)
                heapq.heappush(oheap,(fscore[neighbor],neighbor))

    return []


# =====================================================
# UI INPUT
# =====================================================

col1,col2=st.columns(2)

with col1:
    start_lat=st.number_input("Start Lat",value=22.0)
    start_lon=st.number_input("Start Lon",value=120.0)

with col2:
    end_lat=st.number_input("End Lat",value=25.0)
    end_lon=st.number_input("End Lon",value=122.0)

# =====================================================
# GRID INDEX
# =====================================================

def nearest_idx(lat,lon):
    iy=np.abs(lats-lat).argmin()
    ix=np.abs(lons-lon).argmin()
    return iy,ix

start=nearest_idx(start_lat,start_lon)
goal=nearest_idx(end_lat,end_lon)

# =====================================================
# COST FIELD
# =====================================================

dist_land=distance_transform_edt(~land_mask)

cost=1/(speed+0.05)+1/(dist_land+1)

# =====================================================
# PATH COMPUTE
# =====================================================

with st.spinner("Computing optimal path..."):
    path=astar(cost,start,goal)

# =====================================================
# PLOT
# =====================================================

fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([
    start_lon-5,end_lon+5,
    start_lat-5,end_lat+5
])

ax.coastlines()
ax.add_feature(cfeature.LAND)

skip=15

ax.quiver(
    lons[::skip],
    lats[::skip],
    u[::skip,::skip],
    v[::skip,::skip],
    transform=ccrs.PlateCarree()
)

if path:
    py=[lats[p[0]] for p in path]
    px=[lons[p[1]] for p in path]

    ax.plot(px,py,"r-",linewidth=3)

ax.plot(start_lon,start_lat,"go",markersize=10)
ax.plot(end_lon,end_lat,"ro",markersize=10)

st.pyplot(fig)

st.success("Route computed successfully.")
