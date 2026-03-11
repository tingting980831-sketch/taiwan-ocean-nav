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
# 1. 頁面設定
# ------------------------------
st.set_page_config(layout="wide", page_title="HELONS: Smart Navigation")
st.title("🛰️ HELONS: 智慧衛星航行與海流輔助系統 (2026)")

# ------------------------------
# 2. 禁航區/離岸風場
# ------------------------------
OFFSHORE_WIND = [
    [[24.18,120.12],[24.22,120.28],[24.05,120.35],[24.00,120.15]]
]
OFFSHORE_COST = 50.0

# ------------------------------
# 3. 載入最新 HYCOM 2026 ssu/ssv
# ------------------------------
@st.cache_data(ttl=3600)
def load_hycom():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)
    
    time_origin = pd.to_datetime(ds['time'].attrs.get('time_origin', pd.Timestamp.now()))
    time_vals = time_origin + pd.to_timedelta(ds['time'].values, unit='h')
    
    sub = ds.sel(lat=slice(21,26), lon=slice(118,124))
    u = sub['ssu'].isel(time=-1).values
    v = sub['ssv'].isel(time=-1).values
    lons = sub.lon.values
    lats = sub.lat.values
    obs_time = time_vals[-1]
    land_mask = np.isnan(u)
    
    sea_mask = ~land_mask
    dist_to_land = distance_transform_edt(sea_mask)
    
    return u, v, lons, lats, land_mask, dist_to_land, obs_time

data = load_hycom()
u_field, v_field, lons, lats, land_mask, dist_to_land, obs_time = data

# ------------------------------
# 4. 側邊欄
# ------------------------------
with st.sidebar:
    st.header("🚢 航線設定")
    s_lon = st.number_input("起點經度", 118.0, 124.0, 120.3)
    s_lat = st.number_input("起點緯度", 21.0, 26.0, 22.6)
    e_lon = st.number_input("終點經度", 118.0, 124.0, 122.0)
    e_lat = st.number_input("終點緯度", 21.0, 26.0, 24.5)
    ship_speed_kmh = st.number_input("船對地速度 km/h", 5.0, 60.0, 25.0)
    st.button("重新計算路徑", key="recalc")

# ------------------------------
# 5. 輔助函數
# ------------------------------
def nearest_idx(lon, lat):
    i = np.abs(lats - lat).argmin()
    j = np.abs(lons - lon).argmin()
    return i, j

def offshore_penalty(y, x):
    lon_val = lons[x]; lat_val = lats[y]
    for zone in OFFSHORE_WIND:
        if Path([(p[1], p[0]) for p in zone]).contains_point([lon_val, lat_val]):
            return OFFSHORE_COST
    return 0.0

def energy_cost(y0, x0, y1, x1):
    dy = lats[y1]-lats[y0]
    dx = lons[x1]-lons[x0]
    dist_km = np.hypot(dx, dy)*111.0
    n = np.hypot(dx, dy)
    unit_vec = np.array([dx/n, dy/n]) if n>0 else np.array([0,0])
    v_ocean = np.array([u_field[y1, x1], v_field[y1, x1]])
    v_assist = np.dot(unit_vec, v_ocean)
    v_g = ship_speed_kmh / 3.6
    v_w = max(v_g - v_assist, 0.5)
    energy = (v_w**3)*(dist_km/ship_speed_kmh)
    penalty = offshore_penalty(y1, x1)
    if dist_to_land[y1,x1]<2: penalty += (2-dist_to_land[y1,x1])*10
    return energy + penalty

# ------------------------------
# 6. 限制 A* 搜索區域
# ------------------------------
def astar(start, goal, radius=50):
    rows, cols = land_mask.shape
    r_min = max(0, start[0]-radius); r_max = min(rows, start[0]+radius)
    c_min = max(0, start[1]-radius); c_max = min(cols, start[1]+radius)
    
    pq = [(0, start)]
    came = {}; cost = {start:0}
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    
    while pq:
        _, cur = heapq.heappop(pq)
        if cur==goal: break
        for dr, dc in dirs:
            ny, nx = cur[0]+dr, cur[1]+dc
            if r_min<=ny<r_max and c_min<=nx<c_max:
                if not land_mask[ny,nx]:
                    new_cost = cost[cur]+energy_cost(cur[0],cur[1],ny,nx)
                    if (ny,nx) not in cost or new_cost<cost[(ny,nx)]:
                        cost[(ny,nx)] = new_cost
                        heapq.heappush(pq,(new_cost,(ny,nx)))
                        came[(ny,nx)] = cur
    path=[]
    curr = goal
    while curr in came:
        path.append(curr)
        curr = came[curr]
    if path: path.append(start)
    return path[::-1]

# ------------------------------
# 7. 計算路徑
# ------------------------------
s_idx = nearest_idx(s_lon, s_lat)
e_idx = nearest_idx(e_lon, e_lat)

if "route" not in st.session_state or st.session_state.get("recalc", False):
    with st.spinner("計算最佳節能航線..."):
        st.session_state.route = astar(s_idx, e_idx)

# ------------------------------
# 8. 儀表板
# ------------------------------
def calc_distance_time(path):
    dist = 0
    for i in range(len(path)-1):
        dy = lats[path[i+1][0]] - lats[path[i][0]]
        dx = lons[path[i+1][1]] - lons[path[i][1]]
        dist += np.hypot(dx,dy)*111
    t = dist / ship_speed_kmh
    return dist, t

if st.session_state.route:
    dist, t = calc_distance_time(st.session_state.route)
else:
    dist, t = 0,0

c1,c2,c3 = st.columns(3)
c1.metric("航程距離 (km)", f"{dist:.2f}")
c2.metric("預估航行時間 (h)", f"{t:.2f}")
c3.metric("資料時間", obs_time.strftime("%m/%d %H:%M"))

# ------------------------------
# 9. Map
# ------------------------------
fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118,124,21,26])
speed = np.sqrt(u_field**2 + v_field**2)
ax.pcolormesh(lons,lats,speed,cmap='YlGnBu',shading='auto',alpha=0.8)
ax.add_feature(cfeature.LAND, facecolor='#E0E0E0')
ax.add_feature(cfeature.COASTLINE, linewidth=1)

if st.session_state.route:
    p_lons = [lons[p[1]] for p in st.session_state.route]
    p_lats = [lats[p[0]] for p in st.session_state.route]
    ax.plot(p_lons,p_lats,color='#FF5722',linewidth=3,label='節能航線')
    ax.scatter(s_lon,s_lat,color='green',s=100,label='起點')
    ax.scatter(e_lon,e_lat,color='red',s=100,marker='*',label='終點')

ax.legend()
plt.title("HELONS: Dynamic Path with Sea Current Assistance")
st.pyplot(fig)
