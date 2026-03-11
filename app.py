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

# ===============================
# ------------------------------
# No-Go Zones & Offshore Wind
# 更新為 Prohibited_Sea_Area.json 中的座標 [緯度, 經度]
# ===============================
# ------------------------------
NO_GO_ZONES = [
    [[22.953536, 120.171678], [22.934628, 120.175472], [22.933136, 120.170942], [22.95781, 120.16078]], # 臺南市南區鯤鯓、喜樹近岸海域
    [[22.943956, 120.172358], [22.939717, 120.173944], [22.928353, 120.157372], [22.936636, 120.153547]], # 臺南市南區鯤鯓、喜樹部分海域
    [[22.933136, 120.170942], [22.924847, 120.172583], [22.915003, 120.159022], [22.931536, 120.155772]], # 臺南市南區灣裡近岸海域
    [[22.953536, 120.171678], [22.934628, 120.175472], [22.933136, 120.170942], [22.95781, 120.16078]],
    [[22.943956, 120.172358], [22.939717, 120.173944], [22.928353, 120.157372], [22.936636, 120.153547]],
    [[22.933136, 120.170942], [22.924847, 120.172583], [22.915003, 120.159022], [22.931536, 120.155772]],
]

OFFSHORE_WIND = [
@@ -30,61 +32,61 @@
]
OFFSHORE_COST = 10

# ===============================
# Load HYCOM Ocean Current
# ===============================
# ------------------------------
# Load HYCOM Ocean Data
# ------------------------------
@st.cache_data(ttl=3600)
def load_hycom_data():
    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds=xr.open_dataset(url,decode_times=False)
    ds = xr.open_dataset(url, decode_times=False)
    if 'time_origin' in ds['time'].attrs:
        time_origin=pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time=time_origin+pd.to_timedelta(ds['time'].values[-1],unit='h')
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
    else:
        latest_time=pd.Timestamp.now()
        latest_time = pd.Timestamp.now()

    lons = ds['ssu'].sel(lat=slice(21,26),lon=slice(118,124)).lon.values
    lats = ds['ssu'].sel(lat=slice(21,26),lon=slice(118,124)).lat.values
    land_mask = np.isnan(ds['ssu'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=0).values)
    lons = ds['ssu'].sel(lat=slice(21,26), lon=slice(118,124)).lon.values
    lats = ds['ssu'].sel(lat=slice(21,26), lon=slice(118,124)).lat.values
    land_mask = np.isnan(ds['ssu'].sel(lat=slice(21,26), lon=slice(118,124)).isel(time=0).values)

    return ds, lons, lats, land_mask, latest_time

ds, lons, lats, land_mask, obs_time = load_hycom_data()
sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)

# ===============================
# Sidebar Settings + Next Step
# ===============================
# ------------------------------
# Sidebar Settings
# ------------------------------
with st.sidebar:
    st.header("Route Settings")
    s_lon=st.number_input("Start Longitude",118.0,124.0,120.3)
    s_lat=st.number_input("Start Latitude",21.0,26.0,22.6)
    e_lon=st.number_input("End Longitude",118.0,124.0,122.0)
    e_lat=st.number_input("End Latitude",21.0,26.0,24.5)
    ship_speed=st.number_input("Ship Speed (km/h)",1.0,60.0,20.0)
    s_lon = st.number_input("Start Longitude", 118.0, 124.0, 120.3)
    s_lat = st.number_input("Start Latitude", 21.0, 26.0, 22.6)
    e_lon = st.number_input("End Longitude", 118.0, 124.0, 122.0)
    e_lat = st.number_input("End Latitude", 21.0, 26.0, 24.5)
    ship_speed = st.number_input("Ship Speed (km/h)", 1.0, 60.0, 20.0)
    st.button("Next Step", key="next_step_button")

# ===============================
# ------------------------------
# Helper Functions
# ===============================
def nearest_ocean_cell(lon,lat):
    lon_idx=np.abs(lons-lon).argmin()
    lat_idx=np.abs(lats-lat).argmin()
    return lat_idx,lon_idx

def offshore_penalty(y,x):
    lat=lats[y]
    lon=lons[x]
# ------------------------------
def nearest_ocean_cell(lon, lat):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()
    return lat_idx, lon_idx

def offshore_penalty(y, x):
    lat_val = lats[y]
    lon_val = lons[x]
    for zone in OFFSHORE_WIND:
        if Path(zone).contains_point([lon,lat]):
        if Path(zone).contains_point([lon_val, lat_val]):
            return OFFSHORE_COST
    return 0

MAX_DIST = 2
COAST_PENALTY = 2.0
def coast_penalty(y,x):
    d = dist_to_land[y,x]
def coast_penalty(y, x):
    d = dist_to_land[y, x]
    if d < MAX_DIST:
        return COAST_PENALTY * (MAX_DIST - d)/MAX_DIST
    return 0.0
@@ -109,45 +111,48 @@
                    heapq.heappush(pq,(new,(ni,nj)))
                    came[(ni,nj)] = cur
    path=[]
    curr=goal
    curr = goal
    while curr in came:
        path.append(curr)
        curr=came[curr]
        curr = came[curr]
    if path:
        path.append(start)
    return path[::-1]

# ===============================
# ------------------------------
# Initialize Session State
# ===============================
start = nearest_ocean_cell(s_lon,s_lat)
goal = nearest_ocean_cell(e_lon,e_lat)
# ------------------------------
start = nearest_ocean_cell(s_lon, s_lat)
goal = nearest_ocean_cell(e_lon, e_lat)

if "full_path" not in st.session_state:
    st.session_state.full_path = astar_with_wind_and_coast(start, goal)
if "ship_step_idx" not in st.session_state:
    st.session_state.ship_step_idx = 0

# Estimate total travel time
def estimate_total_time(path):
# Estimate total travel time in hours
def estimate_total_time(path, speed):
    total_dist = 0
    for i in range(len(path)-1):
        y0,x0 = path[i]
        y1,x1 = path[i+1]
        total_dist += np.hypot(lats[y1]-lats[y0], lons[x1]-lons[x0])*111
    return total_dist/ship_speed
    return total_dist / speed

if "total_time" not in st.session_state:
    st.session_state.total_time = estimate_total_time(st.session_state.full_path)
    st.session_state.total_time = estimate_total_time(st.session_state.full_path, ship_speed)

# ------------------------------
# Move Ship One Step
# ------------------------------
if st.session_state.get("next_step_button", False):
    if st.session_state.ship_step_idx < len(st.session_state.full_path)-1:
        st.session_state.ship_step_idx += 1

# ===============================
# Distance & Remaining Time
# ===============================
def calc_remaining(path, step_idx):
# ------------------------------
# Calculate Remaining Distance & Time
# ------------------------------
def calc_remaining(path, step_idx, speed):
    dist_remaining = 0
    for i in range(step_idx, len(path)-1):
        y0,x0 = path[i]
@@ -160,59 +165,62 @@
        angle_deg = np.degrees(np.arctan2(lats[y1]-lats[y0], lons[x1]-lons[x0]))
    else:
        angle_deg = 0
    return dist_remaining, angle_deg
    remaining_time = dist_remaining / speed
    return dist_remaining, remaining_time, angle_deg

remaining_dist, heading_deg = calc_remaining(st.session_state.full_path, st.session_state.ship_step_idx)
remaining_time = st.session_state.total_time * (remaining_dist / st.session_state.total_time) if remaining_dist>0 else 0
remaining_dist, remaining_time, heading_deg = calc_remaining(
    st.session_state.full_path, st.session_state.ship_step_idx, ship_speed
)

# ===============================
# ------------------------------
# Satellite Visibility
# ===============================
# ------------------------------
def visible_sats(ship_lat, ship_lon):
    # Random 3–12 visible satellites (simplified)
    return np.random.randint(3,13)

current_pos = st.session_state.full_path[st.session_state.ship_step_idx]
sat_count = visible_sats(lats[current_pos[0]], lons[current_pos[1]])

# ===============================
# ------------------------------
# Dashboard
# ===============================
# ------------------------------
st.subheader("Navigation Dashboard")
c1,c2,c3,c4=st.columns(4)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Remaining Distance (km)", f"{remaining_dist:.2f}")
c2.metric("Remaining Time (hr)", f"{remaining_time:.2f}")
c2.metric("Remaining Time (hr)", f"{remaining_time:.2f}" if remaining_time>0 else "-")
c3.metric("Heading (°)", f"{heading_deg:.1f}")
c4.metric("Satellites in View", sat_count)
st.caption(f"HYCOM observation time: {obs_time}")

# ===============================
# ------------------------------
# Map
# ===============================
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())
# ------------------------------
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND, facecolor="#b0b0b0")
ax.add_feature(cfeature.COASTLINE)

# Flow field (fixed color scale 0~1.6)
# Flow field (fixed 0~1.6)
time_idx = st.session_state.ship_step_idx
if time_idx >= len(ds['time']):
    time_idx = -1
u_data = ds['ssu'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=time_idx).values
v_data = ds['ssv'].sel(lat=slice(21,26),lon=slice(118,124)).isel(time=time_idx).values
u_data = ds['ssu'].sel(lat=slice(21,26), lon=slice(118,124)).isel(time=time_idx).values
v_data = ds['ssv'].sel(lat=slice(21,26), lon=slice(118,124)).isel(time=time_idx).values
speed = np.sqrt(u_data**2 + v_data**2)
mesh = ax.pcolormesh(lons, lats, speed, cmap="Blues", shading="auto", vmin=0, vmax=1.6)
fig.colorbar(mesh, ax=ax, label="Current Speed (m/s)")

# No-go zones
for zone in NO_GO_ZONES:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="red",alpha=0.4)
    poly = np.array(zone)
    ax.fill(poly[:,1], poly[:,0], color="red", alpha=0.4)

# Offshore wind zones
for zone in OFFSHORE_WIND:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="yellow",alpha=0.4)
    poly = np.array(zone)
    ax.fill(poly[:,1], poly[:,0], color="yellow", alpha=0.4)

# Full path (pink)
full_lons = [lons[p[1]] for p in st.session_state.full_path]
@@ -224,12 +232,12 @@
done_lats = full_lats[:st.session_state.ship_step_idx+1]
ax.plot(done_lons, done_lats, color="red", linewidth=2)

# Ship icon (gray, heading)
# Ship icon (gray, pointing along heading)
ax.scatter(lons[current_pos[1]], lats[current_pos[0]], color="gray", s=150, marker="^")

# Start/End
ax.scatter(s_lon, s_lat, color="#B15BFF", s=80, edgecolors="black")  # Start
ax.scatter(e_lon, e_lat, color="yellow", marker="*", s=200, edgecolors="black")  # End

plt.title("HELIOS System")
st.pyplot(fig)
