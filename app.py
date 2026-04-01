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
# Page Config (保持原本樣式)
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS System")
st.title("🛰️ HELIOS System")

# ===============================
# No-Go Zones & Offshore Wind
# ===============================
NO_GO_ZONES = [
    [[22.953536,120.171678],[22.934628,120.175472],[22.933136,120.170942],[22.95781,120.16078]],
    [[22.943956,120.172358],[22.939717,120.173944],[22.928353,120.157372],[22.936636,120.153547]],
    [[22.933136,120.170942],[22.924847,120.172583],[22.915003,120.159022],[22.931536,120.155772]],
]

OFFSHORE_WIND = [
    [[24.18,120.12],[24.22,120.28],[24.05,120.35],[24.00,120.15]],
    [[24.00,120.10],[24.05,120.32],[23.90,120.38],[23.85,120.15]],
]

# ===============================
# Load HYCOM (⭐ 強化時間對接：目標 2026-04-01)
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_target():
    # 使用 2026 資料源
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)
    
    # 1. 處理時間轉換
    origin = pd.to_datetime(ds['time'].attrs['time_origin'])
    target_date = pd.Timestamp("2026-04-01")
    
    # 計算目標時間與基準時間的差距(小時)
    target_hours = (target_date - origin).total_seconds() / 3600
    
    # 2. 擷取目標時間與區域的流場 (使用 nearest 確保對接到 2026-04-01)
    sub = ds.sel(
        time=target_hours, 
        lat=slice(21, 26), 
        lon=slice(118, 124), 
        method="nearest"
    )
    
    # 3. 獲取實際對接到的時間點
    actual_time = origin + pd.to_timedelta(sub.time.values, unit='h')
    
    lons = sub.lon.values
    lats = sub.lat.values
    land_mask = np.isnan(sub['ssu'].values) # 陸地遮罩
    
    return sub, lons, lats, land_mask, actual_time

# 執行讀取
sub_data, lons, lats, land_mask, obs_time = load_hycom_target()

# 避障距離計算
sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)

# ===============================
# Sidebar (原本格式)
# ===============================
with st.sidebar:
    st.header("Route Settings")
    s_lon = st.number_input("Start Lon", 118.0, 124.0, 120.3, format="%.2f")
    s_lat = st.number_input("Start Lat", 21.0, 26.0, 22.6, format="%.2f")
    e_lon = st.number_input("End Lon", 118.0, 124.0, 122.0, format="%.2f")
    e_lat = st.number_input("End Lat", 21.0, 26.0, 24.5, format="%.2f")
    ship_speed = st.number_input("Ship Speed (km/h)", 1.0, 60.0, 20.0)
    
    if st.button("Next Step"):
        if "ship_step_idx" in st.session_state:
            st.session_state.ship_step_idx += 1

# ===============================
# A* Pathfinding (對接最近流場網格)
# ===============================
def nearest_cell(lon, lat):
    return (np.abs(lats - lat).argmin(), np.abs(lons - lon).argmin())

def astar_engine(start, goal):
    rows, cols = land_mask.shape
    pq = [(0, start)]
    came, cost = {}, {start: 0}
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal: break
        for d in dirs:
            ni, nj = cur[0] + d[0], cur[1] + d[1]
            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni, nj]:
                # 岸邊懲罰權重
                penalty = 10 if dist_to_land[ni, nj] < 3 else 0
                new_cost = cost[cur] + np.hypot(d[0], d[1]) + penalty
                if (ni, nj) not in cost or new_cost < cost[(ni, nj)]:
                    cost[(ni, nj)] = new_cost
                    came[(ni, nj)] = cur
                    priority = new_cost + np.hypot(ni-goal[0], nj-goal[1])
                    heapq.heappush(pq, (priority, (ni, nj)))
    
    path = []
    curr = goal
    while curr in came:
        path.append(curr); curr = came[curr]
    return path[::-1]

# 路徑狀態管理
start_node = nearest_cell(s_lon, s_lat)
goal_node = nearest_cell(e_lon, e_lat)
route_key = (s_lon, s_lat, e_lon, e_lat)

if "route_key" not in st.session_state or st.session_state.route_key != route_key:
    st.session_state.full_path = astar_engine(start_node, goal_node)
    st.session_state.ship_step_idx = 0
    st.session_state.route_key = route_key

path = st.session_state.full_path
st.session_state.ship_step_idx = min(st.session_state.ship_step_idx, len(path)-1)
curr_p = path[st.session_state.ship_step_idx]

# ===============================
# Dashboard
# ===============================
st.subheader("Navigation Dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("Remaining Distance (km)", f"{(len(path)-st.session_state.ship_step_idx)*8.8:.1f}")
c2.metric("Target Date", "2026-04-01")
c3.metric("Observed Time", f"{obs_time.strftime('%Y-%m-%d %H:%M')}")

# ===============================
# Map (⭐ 解決條紋與對齊問題)
# ===============================
fig = plt.figure(figsize=(11, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118, 124, 21, 26], crs=ccrs.PlateCarree())

# 1. 繪製流速底圖
speed = np.sqrt(sub_data['ssu']**2 + sub_data['ssv']**2)

# 使用 xarray 的 plot 功能，強制對齊 lon/lat
speed.plot.pcolormesh(
    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
    cmap="Blues", vmin=0, vmax=1.5, zorder=1,
    cbar_kwargs={'label': 'Current Speed (m/s)', 'pad': 0.02}
)

# 2. 地理特徵
ax.add_feature(cfeature.LAND, facecolor="#b0b0b0", zorder=2)
ax.add_feature(cfeature.COASTLINE, zorder=3)

# 3. 禁航區與航線
for zone in NO_GO_ZONES:
    ax.fill([p[1] for p in zone], [p[0] for p in zone], color="red", alpha=0.4, transform=ccrs.PlateCarree(), zorder=4)

if len(path) > 0:
    f_lons = [lons[p[1]] for p in path]
    f_lats = [lats[p[0]] for p in path]
    ax.plot(f_lons, f_lats, color="pink", linewidth=3, transform=ccrs.PlateCarree(), zorder=5)
    ax.plot(f_lons[:st.session_state.ship_step_idx+1], f_lats[:st.session_state.ship_step_idx+1], 
            color="red", linewidth=3, transform=ccrs.PlateCarree(), zorder=6)
    ax.scatter(f_lons[st.session_state.ship_step_idx], f_lats[st.session_state.ship_step_idx], 
               color="black", marker="^", s=150, transform=ccrs.PlateCarree(), zorder=10)

ax.scatter(s_lon, s_lat, color="purple", s=100, transform=ccrs.PlateCarree(), zorder=11)
ax.scatter(e_lon, e_lat, color="gold", marker="*", s=300, transform=ccrs.PlateCarree(), zorder=11)

st.pyplot(fig)
