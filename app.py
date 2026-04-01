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
    [[23.88,120.05],[23.92,120.18],[23.75,120.25],[23.70,120.08]],
    [[23.68,120.02],[23.72,120.12],[23.58,120.15],[23.55,120.05]],
]

OFFSHORE_COST = 10

# ===============================
# Load HYCOM (強化最近點抓取)
# ===============================
@st.cache_data(ttl=3600)
def load_hycom():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)
    
    # 這裡先切出大範圍，避免運算過慢
    sub = ds.sel(lat=slice(21, 27), lon=slice(117, 125))
    lons = sub.lon.values
    lats = sub.lat.values
    
    # 抓取最新時間的 U/V 分量來做陸地遮罩
    u_latest = sub['ssu'].isel(time=-1).values
    land_mask = np.isnan(u_latest)
    
    # 獲取觀測時間
    if 'time_origin' in ds['time'].attrs:
        origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        obs_time = origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
    else:
        obs_time = pd.Timestamp.now()

    return sub, lons, lats, land_mask, obs_time

ds_sub, lons, lats, land_mask, obs_time = load_hycom()

# 距離轉換（避障邏輯）
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
    
    if st.button("Next Step", key="next_step"):
        if "ship_step_idx" in st.session_state:
            st.session_state.ship_step_idx += 1

# ===============================
# Helpers & A* (確保對接到最近流場)
# ===============================
def nearest_cell(lon, lat):
    # 使用 argmin 找到物理距離最近的網格索引
    return (np.abs(lats - lat).argmin(), np.abs(lons - lon).argmin())

def astar(start, goal):
    rows, cols = land_mask.shape
    pq = [(0, start)]
    came = {}
    cost = {start: 0}
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal: break
        for d in dirs:
            ni, nj = cur[0] + d[0], cur[1] + d[1]
            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni, nj]:
                # 增加避開陸地的權重
                penalty = 0
                if dist_to_land[ni, nj] < 2: penalty = 5
                
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

# 路由更新邏輯
start_node = nearest_cell(s_lon, s_lat)
goal_node = nearest_cell(e_lon, e_lat)
route_key = (s_lon, s_lat, e_lon, e_lat)

if "route_key" not in st.session_state or st.session_state.route_key != route_key:
    st.session_state.full_path = astar(start_node, goal_node)
    st.session_state.ship_step_idx = 0
    st.session_state.route_key = route_key

path = st.session_state.full_path
st.session_state.ship_step_idx = min(st.session_state.ship_step_idx, len(path)-1)
current_pos = path[st.session_state.ship_step_idx]

# ===============================
# Dashboard (原本格式)
# ===============================
st.subheader("Navigation Dashboard")
dist_rem = (len(path) - st.session_state.ship_step_idx) * 0.08 * 111 # 粗估距離
c1, c2, c3 = st.columns(3)
c1.metric("Remaining Distance (km)", f"{dist_rem:.2f}")
c2.metric("Remaining Time (hr)", f"{dist_rem/ship_speed:.2f}")
c3.metric("Satellite Status", "Connected (12)") # 根據你的 Baseline 研究

# ===============================
# Map (核心修正：確保座標對齊且無條紋)
# ===============================
fig = plt.figure(figsize=(12, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118, 124, 21, 26], crs=ccrs.PlateCarree())

# 1. 抓取流場 (⭐ 強制轉置解決條紋問題)
u_plot = ds_sub['ssu'].isel(time=-1)
v_plot = ds_sub['ssv'].isel(time=-1)
speed_plot = np.sqrt(u_plot**2 + v_plot**2)

# 使用 xarray 的 plot 功能自動對接 lon/lat，這是最不會出錯的方式
speed_plot.plot.pcolormesh(
    ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
    cmap="Blues", vmin=0, vmax=1.6, zorder=1, add_colorbar=True,
    cbar_kwargs={'label': 'Current Speed (m/s)', 'pad': 0.02}
)

# 2. 地理特徵 (蓋在流場上)
ax.add_feature(cfeature.LAND, facecolor="#b0b0b0", zorder=2)
ax.add_feature(cfeature.COASTLINE, zorder=3)

# 3. 禁航區與風場
for zone in NO_GO_ZONES:
    ax.fill([p[1] for p in zone], [p[0] for p in zone], color="red", alpha=0.4, transform=ccrs.PlateCarree(), zorder=4)
for zone in OFFSHORE_WIND:
    ax.fill([p[1] for p in zone], [p[0] for p in zone], color="yellow", alpha=0.4, transform=ccrs.PlateCarree(), zorder=4)

# 4. 航線繪製 (⭐ 從數據座標提取)
if len(path) > 0:
    f_lons = [lons[p[1]] for p in path]
    f_lats = [lats[p[0]] for p in path]
    ax.plot(f_lons, f_lats, color="pink", linewidth=3, transform=ccrs.PlateCarree(), zorder=5)
    
    idx = st.session_state.ship_step_idx
    ax.plot(f_lons[:idx+1], f_lats[:idx+1], color="red", linewidth=3, transform=ccrs.PlateCarree(), zorder=6)
    
    # 船隻圖標
    ax.scatter(f_lons[idx], f_lats[idx], color="gray", marker="^", s=200, transform=ccrs.PlateCarree(), zorder=10)

# 5. 起點與終點
ax.scatter(s_lon, s_lat, color="#B15BFF", s=100, transform=ccrs.PlateCarree(), zorder=11, label='Start')
ax.scatter(e_lon, e_lat, color="yellow", marker="*", s=300, transform=ccrs.PlateCarree(), zorder=11, label='Goal')

plt.title(f"HELIOS System | Data Time: {obs_time}")
st.pyplot(fig)
