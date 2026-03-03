import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from heapq import heappush, heappop
from scipy.ndimage import distance_transform_edt

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS V28 最終防撞版", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0


# ===============================
# HYCOM 讀取
# ===============================
@st.cache_data(ttl=3600)
def fetch_hycom_v28():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(
            lat=slice(20.0, 27.0),
            lon=slice(118.0, 126.0),
            depth=0
        ).isel(time=-1).load()

        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        return None, "N/A", "OFFLINE"


ocean_data, data_clock, stream_status = fetch_hycom_v28()

# ===============================
# ⭐ 建立避障成本地圖 (核心升級)
# ===============================
def build_cost_map(ocean_data):

    lats = ocean_data.lat.values
    lons = ocean_data.lon.values

    speed = np.sqrt(
        ocean_data.water_u.values**2 +
        ocean_data.water_v.values**2
    )

    # 海洋 = True
    ocean_mask = np.isfinite(speed)

    # 距離陸地
    dist = distance_transform_edt(ocean_mask)

    cost = np.ones_like(dist, dtype=float)

    # 靠岸懲罰（避免貼岸）
    cost += np.exp(-dist/3) * 50

    # 陸地不可走
    cost[~ocean_mask] = np.inf

    return cost, lats, lons


# ===============================
# ⭐ A* 導航
# ===============================
def astar_route(start, goal, cost):

    h, w = cost.shape

    def heuristic(a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    open_set = []
    heappush(open_set, (0, start))

    came_from = {}
    g = {start: 0}

    dirs = [
        (1,0),(-1,0),(0,1),(0,-1),
        (1,1),(1,-1),(-1,1),(-1,-1)
    ]

    while open_set:

        _, current = heappop(open_set)

        if current == goal:
            path=[current]
            while current in came_from:
                current=came_from[current]
                path.append(current)
            return path[::-1]

        for d in dirs:
            nx = current[0]+d[0]
            ny = current[1]+d[1]

            if nx<0 or ny<0 or nx>=h or ny>=w:
                continue

            if np.isinf(cost[nx,ny]):
                continue

            new_g = g[current] + cost[nx,ny]

            if (nx,ny) not in g or new_g < g[(nx,ny)]:
                g[(nx,ny)] = new_g
                f = new_g + heuristic((nx,ny), goal)
                heappush(open_set,(f,(nx,ny)))
                came_from[(nx,ny)] = current

    return []


# ===============================
# 航向計算（原樣保留）
# ===============================
def calc_bearing(p1, p2):
    y = np.sin(np.radians(p2[1]-p1[1])) * np.cos(np.radians(p2[0]))
    x = np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0])) - \
        np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y,x)) + 360) % 360


# ===============================
# 儀表板（完全未修改）
# ===============================
st.title("🛰️ HELIOS V28 (絕對避障與定位版)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🚀 航速", "15.8 kn")
c2.metric("⛽ 能源紅利", "24.5%")

brg="---"
if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
    brg=f"{calc_bearing(st.session_state.real_p[st.session_state.step_idx],st.session_state.real_p[st.session_state.step_idx+1]):.1f}°"

c3.metric("🧭 建議航向", brg)
c4.metric("📡 衛星", "12 Pcs")
c5.metric("🕒 數據時標", data_clock)

st.markdown("---")

m1,m2,m3=st.columns(3)
m1.success(f"🌊 流場: {stream_status}")
m2.info("📡 GNSS: 訊號鎖定")
m3.write(f"📍 目前位置: {st.session_state.ship_lat:.3f}N, {st.session_state.ship_lon:.3f}E")

st.markdown("---")

# ===============================
# Sidebar（UI保留，只換導航核心）
# ===============================
with st.sidebar:

    st.header("🚢 導航控制器")

    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=22.500, format="%.3f")
    elon = st.number_input("終點經度", value=120.000, format="%.3f")

    if st.button("🚀 啟動智能航路", use_container_width=True):

        cost, lats, lons = build_cost_map(ocean_data)

        def nearest(lat, lon):
            iy = np.abs(lats-lat).argmin()
            ix = np.abs(lons-lon).argmin()
            return (iy, ix)

        start = nearest(slat, slon)
        goal = nearest(elat, elon)

        path_idx = astar_route(start, goal, cost)

        if len(path_idx)==0:
            st.error("找不到安全航路")
        else:
            final_p = [[lats[i], lons[j]] for i,j in path_idx]

            st.session_state.real_p = final_p
            st.session_state.ship_lat = slat
            st.session_state.ship_lon = slon
            st.session_state.step_idx = 0
            st.rerun()

# ===============================
# 地圖（完全保留）
# ===============================
fig, ax = plt.subplots(figsize=(10,8),
        subplot_kw={'projection':ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.5, zorder=3)

if ocean_data is not None:
    lons, lats = ocean_data.lon, ocean_data.lat
    speed=np.sqrt(ocean_data.water_u**2+ocean_data.water_v**2)

    ax.pcolormesh(lons,lats,speed,cmap='YlGn',alpha=0.6,zorder=1)

    skip=(slice(None,None,4),slice(None,None,4))
    ax.quiver(
        lons[skip[1]],lats[skip[0]],
        ocean_data.water_u[skip],
        ocean_data.water_v[skip],
        color='white',alpha=0.3,scale=22,zorder=4)

if st.session_state.real_p:
    py,px=zip(*st.session_state.real_p)
    ax.plot(px,py,color='#FF00FF',linewidth=3,zorder=6)
    ax.scatter(st.session_state.ship_lon,st.session_state.ship_lat,
               color='red',s=80,edgecolors='white',zorder=7)
    ax.scatter(elon,elat,color='gold',marker='*',s=250,zorder=8)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)

# ===============================
# 航行動畫（保留）
# ===============================
if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
        st.session_state.step_idx=min(
            st.session_state.step_idx+10,
            len(st.session_state.real_p)-1
        )
        st.session_state.ship_lat,st.session_state.ship_lon=\
            st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
