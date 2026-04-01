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

# ===============================
# Page Config
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS System")
st.title("🛰️ HELIOS System")

# ===============================
# No-Go Zones & Offshore Wind
# ===============================
NO_GO_ZONES = [
    [[22.953536, 120.171678], [22.934628, 120.175472], [22.933136, 120.170942], [22.95781, 120.16078]],
    [[22.943956, 120.172358], [22.939717, 120.173944], [22.928353, 120.157372], [22.936636, 120.153547]],
    [[22.933136, 120.170942], [22.924847, 120.172583], [22.915003, 120.159022], [22.931536, 120.155772]],
]

OFFSHORE_WIND = [
    [[24.18, 120.12], [24.22, 120.28], [24.05, 120.35], [24.00, 120.15]],
    [[24.00, 120.10], [24.05, 120.32], [23.90, 120.38], [23.85, 120.15]],
    [[23.88, 120.05], [23.92, 120.18], [23.75, 120.25], [23.70, 120.08]],
    [[23.68, 120.02], [23.72, 120.12], [23.58, 120.15], [23.55, 120.05]],
]

OFFSHORE_COST = 10

# ===============================
# Load HYCOM
# ===============================
@st.cache_data(ttl=3600)
def load_hycom():
    # 這裡的 URL 需確保在 2026 年依然有效或使用當前的 OpenDAP 連結
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
    except Exception as e:
        st.error(f"無法連接到 HYCOM 數據庫: {e}")
        st.stop()

    if 'time_origin' in ds['time'].attrs:
        origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        obs_time = origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
    else:
        obs_time = pd.Timestamp.now()

    sub = ds.sel(lat=slice(21, 26), lon=slice(118, 124))
    lons = sub.lon.values
    lats = sub.lat.values
    # 使用第一層深度和第一個時間點作為陸地遮罩參考
    land_mask = np.isnan(sub['ssu'].isel(time=0).values)

    return ds, lons, lats, land_mask, obs_time

ds, lons, lats, land_mask, obs_time = load_hycom()

sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("Route Settings")

    s_lon = st.number_input("Start Lon", 118.0, 124.0, 120.3)
    s_lat = st.number_input("Start Lat", 21.0, 26.0, 22.6)

    e_lon = st.number_input("End Lon", 118.0, 124.0, 122.0)
    e_lat = st.number_input("End Lat", 21.0, 26.0, 24.5)

    ship_speed = st.number_input("Ship Speed (km/h)", 1.0, 60.0, 20.0)

    st.button("Next Step", key="next_step")

# ===============================
# Helpers
# ===============================
def nearest_cell(lon, lat):
    return (
        np.abs(lats - lat).argmin(),
        np.abs(lons - lon).argmin()
    )

def offshore_penalty(y, x):
    for zone in OFFSHORE_WIND:
        if Path(zone).contains_point([lons[x], lats[y]]):
            return OFFSHORE_COST
    return 0

def coast_penalty(y, x):
    d = dist_to_land[y, x]
    if d < 2:
        return (2 - d) * 2
    return 0

# ===============================
# A* Pathfinding
# ===============================
dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)]

def astar(start, goal):
    rows, cols = land_mask.shape
    pq = [(0, start)]
    came = {}
    cost = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)

        if cur == goal:
            break

        for d in dirs:
            ni, nj = cur[0] + d[0], cur[1] + d[1]

            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni, nj]:
                base = np.hypot(d[0], d[1])
                new = cost[cur] + base + offshore_penalty(ni, nj) + coast_penalty(ni, nj)

                if (ni, nj) not in cost or new < cost[(ni, nj)]:
                    cost[(ni, nj)] = new
                    came[(ni, nj)] = cur
                    heapq.heappush(pq, (new, (ni, nj)))

    path = []
    cur = goal
    while cur in came:
        path.append(cur)
        cur = came[cur]

    if path:
        path.append(start)

    return path[::-1]

# ===============================
# ROUTE LOGIC
# ===============================
start = nearest_cell(s_lon, s_lat)
goal = nearest_cell(e_lon, e_lat)

route_key = (round(s_lon, 4), round(s_lat, 4),
             round(e_lon, 4), round(e_lat, 4))

if "route_key" not in st.session_state or st.session_state.route_key != route_key:
    with st.spinner("Computing optimal route..."):
        new_path = astar(start, goal)

    if len(new_path) == 0:
        st.error("❌ No valid route found")
        st.stop()

    st.session_state.full_path = new_path
    st.session_state.ship_step_idx = 0
    st.session_state.route_key = route_key

# ===============================
# State Management
# ===============================
path = st.session_state.full_path

# 防止索引越界
st.session_state.ship_step_idx = min(
    st.session_state.ship_step_idx,
    len(path) - 1
)

if st.session_state.get("next_step"):
    if st.session_state.ship_step_idx < len(path) - 1:
        st.session_state.ship_step_idx += 1

current_pos = path[st.session_state.ship_step_idx]

# ===============================
# Calculations
# ===============================
def calc_remaining(path, idx):
    dist = 0
    for i in range(idx, len(path) - 1):
        y0, x0 = path[i]
        y1, x1 = path[i + 1]
        # 簡單估算：1度緯度約 111km
        dist += np.hypot(lats[y1] - lats[y0], lons[x1] - lons[x0]) * 111

    if idx < len(path) - 1:
        y0, x0 = path[idx]
        y1, x1 = path[idx + 1]
        heading = np.degrees(np.arctan2(lats[y1] - lats[y0], lons[x1] - lons[x0]))
    else:
        heading = 0

    return dist, dist / ship_speed, heading

remaining_dist, remaining_time, heading = calc_remaining(
    path, st.session_state.ship_step_idx
)

# ===============================
# Dashboard
# ===============================
st.subheader("Navigation Dashboard")

c1, c2, c3 = st.columns(3)
c1.metric("Remaining Distance (km)", f"{remaining_dist:.2f}")
c2.metric("Remaining Time (hr)", f"{remaining_time:.2f}")
c3.metric("Heading", f"{heading:.1f}°")

st.caption(f"HYCOM observation time: {obs_time}")

# ===============================
# Map Visualization
# ===============================
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118, 124, 21, 26])
ax.add_feature(cfeature.LAND, facecolor="#b0b0b0")
ax.add_feature(cfeature.COASTLINE)

# 獲取海流數據
try:
    time_idx = -1
    u = ds['ssu'].sel(lat=slice(21, 26), lon=slice(118, 124)).isel(time=time_idx)
    v = ds['ssv'].sel(lat=slice(21, 26), lon=slice(118, 124)).isel(time=time_idx)
    speed = np.sqrt(u**2 + v**2)
    mesh = ax.pcolormesh(lons, lats, speed, cmap="Blues", shading="auto", vmin=0, vmax=1.6)
    fig.colorbar(mesh, ax=ax, label="Current Speed (m/s)")
except:
    st.warning("Could not overlay current data.")

# 繪製禁航區與風場
for zone in NO_GO_ZONES:
    poly = np.array(zone)
    ax.fill(poly[:, 1], poly[:, 0], color="red", alpha=0.4)

for zone in OFFSHORE_WIND:
    poly = np.array(zone)
    ax.fill(poly[:, 1], poly[:, 0], color="yellow", alpha=0.4)

# 繪製路徑
full_lons = [lons[p[1]] for p in path]
full_lats = [lats[p[0]] for p in path]
ax.plot(full_lons, full_lats, color="pink", linewidth=2, label="Planned Route")

# 繪製已行駛路徑
done_lons = full_lons[:st.session_state.ship_step_idx + 1]
done_lats = full_lats[:st.session_state.ship_step_idx + 1]
ax.plot(done_lons, done_lats, color="red", linewidth=2)

# 標記船隻當前位置、起點與終點
ax.scatter(lons[current_pos[1]], lats[current_pos[0]], color="black", marker="^", s=150, zorder=5)
ax.scatter(s_lon, s_lat, color="#B15BFF", s=80, edgecolors="black", label="Start")
ax.scatter(e_lon, e_lat, color="yellow", marker="*", s=200, edgecolors="black", label="Goal")

plt.title("HELIOS Navigation Map")
st.pyplot(fig)
