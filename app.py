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
# 1️⃣ 系統設定與資料載入
# ===============================
st.set_page_config(layout="wide", page_title="HELONS 2026 Accelerated")
st.title("🛰️ HELONS: 衛星導航與能耗優化系統")

@st.cache_data(ttl=3600)
def load_hycom_2026():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)
    time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
    time_values = time_origin + pd.to_timedelta(ds['time'].values, unit='h')
    sub = ds.sel(lat=slice(21, 26), lon=slice(118, 124))
    u = sub['ssu'].isel(time=-1).values
    v = sub['ssv'].isel(time=-1).values
    return u, v, sub.lon.values, sub.lat.values, time_values[-1]

u_raw, v_raw, lons_raw, lats_raw, obs_time = load_hycom_2026()

# ===============================
# 2️⃣ 側邊欄設定
# ===============================
with st.sidebar:
    st.header("⚙️ 航道設定與加速選項")
    grid_step = st.slider("網格運算精細度 (步長越大越快)", 1, 4, 2)
    
    s_lon = st.number_input("起點經度", 118.0, 124.0, 120.3)
    s_lat = st.number_input("起點緯度", 21.0, 26.0, 22.6)
    e_lon = st.number_input("終點經度", 118.0, 124.0, 122.0)
    e_lat = st.number_input("終點緯度", 21.0, 26.0, 24.5)
    ship_speed = st.number_input("航速 (km/h)", 10.0, 50.0, 25.0)
    
    recalc = st.button("🚀 計算最佳航路")

# 小格點切片加速
u = u_raw[::grid_step, ::grid_step]
v = v_raw[::grid_step, ::grid_step]
lons = lons_raw[::grid_step]
lats = lats_raw[::grid_step]
land_mask = np.isnan(u)
dist_to_land = distance_transform_edt(~land_mask)

# 離岸風區域
WIND_ZONES = [[[24.18, 120.12], [24.22, 120.28], [24.05, 120.35], [24.00, 120.15]]]

wind_mask = np.zeros((len(lats), len(lons)), dtype=bool)
for zone in WIND_ZONES:
    p = Path(zone)
    for i in range(len(lats)):
        for j in range(len(lons)):
            if p.contains_point([lons[j], lats[i]]):
                wind_mask[i,j] = True

# ===============================
# 3️⃣ Helper Functions
# ===============================
def nearest_ocean_cell(lon, lat, lons_arr, lats_arr):
    lon_idx = np.abs(lons_arr - lon).argmin()
    lat_idx = np.abs(lats_arr - lat).argmin()
    return lat_idx, lon_idx

def get_cost(curr, nxt):
    y0, x0 = curr
    y1, x1 = nxt
    dist = np.hypot(lons[x1]-lons[x0], lats[y1]-lats[y0]) * 111.0
    move_vec = np.array([lons[x1]-lons[x0], lats[y1]-lats[y0]])
    move_vec /= np.linalg.norm(move_vec)
    v_ocean = np.array([u[y1,x1], v[y1,x1]])
    v_assist = np.dot(move_vec, v_ocean)
    v_g = ship_speed / 3.6
    v_w = max(v_g - v_assist, 0.5)
    energy = (v_w**3) * (dist / ship_speed)
    penalty = 0
    if wind_mask[y1, x1]: penalty += 50
    if dist_to_land[y1,x1] < 2: penalty += (2 - dist_to_land[y1,x1]) * 20
    return energy + penalty

def astar(start, goal):
    pq = [(0, start)]
    came, cost = {}, {start:0}
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    while pq:
        _, curr = heapq.heappop(pq)
        if curr == goal: break
        for dy, dx in dirs:
            nxt = (curr[0]+dy, curr[1]+dx)
            if 0 <= nxt[0] < len(lats) and 0 <= nxt[1] < len(lons) and not land_mask[nxt]:
                new_cost = cost[curr] + get_cost(curr, nxt)
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    heapq.heappush(pq, (new_cost, nxt))
                    came[nxt] = curr
    path=[]
    curr = goal
    while curr in came:
        path.append(curr); curr = came[curr]
    return path[::-1]

# ===============================
# 4️⃣ 計算路徑
# ===============================
s_idx = nearest_ocean_cell(s_lon, s_lat, lons, lats)
e_idx = nearest_ocean_cell(e_lon, e_lat, lons, lats)

if "final_path" not in st.session_state or recalc:
    with st.spinner("正在優化航路..."):
        st.session_state.final_path = astar(s_idx, e_idx)

path = st.session_state.final_path

# ===============================
# 5️⃣ 計算剩餘距離與航向
# ===============================
def calc_remaining(path, step_idx, speed):
    dist_remaining = 0
    angle_deg = 0
    for i in range(step_idx, len(path)-1):
        y0,x0 = path[i]
        y1,x1 = path[i+1]
        dist_remaining += np.hypot(lats[y1]-lats[y0], lons[x1]-lons[x0]) * 111
    if step_idx < len(path)-1:
        y0,x0 = path[step_idx]
        y1,x1 = path[step_idx+1]
        angle_deg = np.degrees(np.arctan2(lats[y1]-lats[y0], lons[x1]-lons[x0]))
    return dist_remaining, dist_remaining/speed, angle_deg

if "ship_step_idx" not in st.session_state:
    st.session_state.ship_step_idx = 0

if st.button("➡️ 下一步"):
    if st.session_state.ship_step_idx < len(path)-1:
        st.session_state.ship_step_idx += 1

remaining_dist, remaining_time, heading_deg = calc_remaining(path, st.session_state.ship_step_idx, ship_speed)

# ===============================
# 6️⃣ 儀表板
# ===============================
st.subheader("導航儀表板")
c1,c2,c3 = st.columns(3)
c1.metric("剩餘距離 (km)", f"{remaining_dist:.2f}")
c2.metric("剩餘時間 (hr)", f"{remaining_time:.2f}")
c3.metric("航向 (°)", f"{heading_deg:.1f}")

# ===============================
# 7️⃣ 地圖繪製
# ===============================
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118.5,123.5,21.5,25.5])
ax.add_feature(cfeature.LAND, facecolor="#d3d3d3")
ax.add_feature(cfeature.COASTLINE)

# 流速背景
speed = np.sqrt(u**2 + v**2)
im = ax.pcolormesh(lons, lats, speed, cmap='YlGnBu', shading='auto', alpha=0.7)
plt.colorbar(im, ax=ax, label="海流流速 (m/s)", orientation='horizontal', pad=0.05)

# 航路
if path:
    path_lons = [lons[i[1]] for i in path]
    path_lats = [lats[i[0]] for i in path]
    ax.plot(path_lons, path_lats, color="orange", linewidth=3, label="HELONS 優化航路")

# 船位置
current = path[st.session_state.ship_step_idx]
ax.scatter(lons[current[1]], lats[current[0]], color="red", marker="^", s=150)

# 起點/終點
ax.scatter(s_lon, s_lat, color="green", s=80, edgecolors="black")
ax.scatter(e_lon, e_lat, color="yellow", marker="*", s=200, edgecolors="black")

plt.title("HELONS Dynamic Navigation")
ax.legend()
st.pyplot(fig)
