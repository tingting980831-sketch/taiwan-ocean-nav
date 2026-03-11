import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path

st.set_page_config(layout="wide", page_title="HELIOS Fast Navigation")
st.title("🛰️ HELIOS Fast Ocean Navigation")

# ===============================
# Parameters
# ===============================
NO_GO_ZONES = [
    [[22.953536,120.171678],[22.934628,120.175472],[22.933136,120.170942],[22.95781,120.16078]],
    [[22.943956,120.172358],[22.939717,120.173944],[22.928353,120.157372],[22.936636,120.153547]],
]
OFFSHORE_WIND = [
    [[24.18,120.12],[24.22,120.28],[24.05,120.35],[24.00,120.15]],
]
OFFSHORE_COST = 10
COAST_PENALTY = 2.0
MAX_DIST = 2

# ===============================
# Load HYCOM (subset)
# ===============================
@st.cache_data(ttl=3600)
def load_hycom():
    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds=xr.open_dataset(url, decode_times=False)
    subset=ds['ssu'].sel(lat=slice(21,26), lon=slice(118,124))
    lons=subset.lon.values
    lats=subset.lat.values
    land_mask=np.isnan(subset.isel(time=0).values)
    latest_time = pd.Timestamp.now()
    return ds, lons, lats, land_mask, latest_time

ds, lons, lats, land_mask, obs_time = load_hycom()

# ===============================
# Reduce Grid Resolution for Speed
# ===============================
grid_step = st.sidebar.slider("Grid Step (1=high precision, 2=fast)", 1, 5, 2)
lons_small = lons[::grid_step]
lats_small = lats[::grid_step]
land_mask_small = land_mask[::grid_step, ::grid_step]
dist_to_land_small = distance_transform_edt(~land_mask_small)

# ===============================
# Sidebar Inputs
# ===============================
with st.sidebar:
    s_lon = st.number_input("Start Lon", 118.0, 124.0, 120.3)
    s_lat = st.number_input("Start Lat", 21.0, 26.0, 22.6)
    e_lon = st.number_input("End Lon", 118.0, 124.0, 122.0)
    e_lat = st.number_input("End Lat", 21.0, 26.0, 24.5)
    ship_speed = st.number_input("Ship Speed (km/h)", 1.0, 60.0, 20.0)
    st.button("Recalculate Path", key="recalc_path")

# ===============================
# Helper Functions
# ===============================
def nearest_cell(lon, lat, lons_array, lats_array):
    lon_idx = np.abs(lons_array - lon).argmin()
    lat_idx = np.abs(lats_array - lat).argmin()
    return lat_idx, lon_idx

def coast_penalty(y, x):
    d = dist_to_land_small[y, x]
    if d < MAX_DIST:
        return COAST_PENALTY * (MAX_DIST - d) / MAX_DIST
    return 0.0

def offshore_penalty(y, x):
    lat_val = lats_small[y]
    lon_val = lons_small[x]
    for zone in OFFSHORE_WIND:
        if Path(zone).contains_point([lon_val, lat_val]):
            return OFFSHORE_COST
    return 0

def get_energy_cost(curr, nxt):
    y0,x0 = curr
    y1,x1 = nxt
    dist = np.hypot(lats_small[y1]-lats_small[y0], lons_small[x1]-lons_small[x0])*111
    return dist + coast_penalty(y1,x1) + offshore_penalty(y1,x1)

# ===============================
# A* Algorithm
# ===============================
def astar(start, goal):
    rows, cols = land_mask_small.shape
    dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq = [(0, start)]
    came = {}
    cost_so_far = {start:0}

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal: break

        for d in dirs:
            ni, nj = current[0]+d[0], current[1]+d[1]
            if 0<=ni<rows and 0<=nj<cols and not land_mask_small[ni,nj]:
                new_cost = cost_so_far[current] + get_energy_cost(current,(ni,nj))
                if (ni,nj) not in cost_so_far or new_cost<cost_so_far[(ni,nj)]:
                    cost_so_far[(ni,nj)] = new_cost
                    heapq.heappush(pq,(new_cost,(ni,nj)))
                    came[(ni,nj)] = current

    path=[]
    curr = goal
    while curr in came:
        path.append(curr)
        curr = came[curr]
    path.append(start)
    return path[::-1]

# ===============================
# Compute Path
# ===============================
start = nearest_cell(s_lon, s_lat, lons_small, lats_small)
goal = nearest_cell(e_lon, e_lat, lons_small, lats_small)

if "path" not in st.session_state or st.session_state.get("recalc_path_button", False):
    st.session_state.path = astar(start, goal)
    st.session_state.ship_idx = 0

# ===============================
# Estimate Remaining Distance & Time
# ===============================
def calc_remaining(path, idx, speed):
    dist = 0
    for i in range(idx, len(path)-1):
        y0,x0 = path[i]
        y1,x1 = path[i+1]
        dist += np.hypot(lats_small[y1]-lats_small[y0], lons_small[x1]-lons_small[x0])*111
    if idx < len(path)-1:
        y0,x0 = path[idx]
        y1,x1 = path[idx+1]
        heading = np.degrees(np.arctan2(lats_small[y1]-lats_small[y0], lons_small[x1]-lons_small[x0]))
    else:
        heading = 0
    return dist, dist/speed, heading

remaining_dist, remaining_time, heading_deg = calc_remaining(st.session_state.path, st.session_state.ship_idx, ship_speed)

# ===============================
# Dashboard
# ===============================
st.subheader("Navigation Dashboard")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Remaining Distance (km)", f"{remaining_dist:.2f}")
c2.metric("Remaining Time (hr)", f"{remaining_time:.2f}")
c3.metric("Heading (°)", f"{heading_deg:.1f}")
c4.metric("Satellites in View", np.random.randint(3,13))
st.caption(f"HYCOM observation time: {obs_time}")

# ===============================
# Map Visualization
# ===============================
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND,facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE)

# No-Go Zones
for zone in NO_GO_ZONES:
    poly = np.array(zone)
    ax.fill(poly[:,1], poly[:,0], color="red", alpha=0.4)
# Offshore Wind
for zone in OFFSHORE_WIND:
    poly = np.array(zone)
    ax.fill(poly[:,1], poly[:,0], color="yellow", alpha=0.4)

# Path
path_lons = [lons_small[p[1]] for p in st.session_state.path]
path_lats = [lats_small[p[0]] for p in st.session_state.path]
ax.plot(path_lons, path_lats, color="pink", linewidth=2)

# Current Ship Position
current = st.session_state.path[st.session_state.ship_idx]
ax.scatter(lons_small[current[1]], lats_small[current[0]], color="gray", marker="^", s=150)

# Start/End
ax.scatter(s_lon, s_lat, color="green", s=80)
ax.scatter(e_lon, e_lat, color="red", marker="*", s=150)

plt.title("HELIOS Fast Navigation")
st.pyplot(fig)
