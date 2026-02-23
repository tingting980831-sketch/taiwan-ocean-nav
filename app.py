import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

# ===============================
# 基本設定
# ===============================
st.set_page_config(page_title="AI 海象導航系統", layout="wide")
st.title("⚓ 智慧避障導航系統")
st.write("A* 演算法 × HYCOM 海流 × AIS × 低軌衛星通訊")

DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"

# ===============================
# Session State 初始化
# ===============================
if "start_lon" not in st.session_state:
    st.session_state.start_lon = 121.750
    st.session_state.start_lat = 25.150

# ===============================
# Sidebar：出發點設定
# ===============================
st.sidebar.header("📍 出發點設定")

def get_ais_position():
    # 模擬 AIS 即時定位
    return (
        np.random.uniform(121.6, 122.1),
        np.random.uniform(24.8, 25.3)
    )

if st.sidebar.button("📡 立即定位（AIS）"):
    lon, lat = get_ais_position()
    st.session_state.start_lon = lon
    st.session_state.start_lat = lat
    st.sidebar.success("已取得 AIS 即時位置")

s_lon = st.sidebar.number_input(
    "起點經度", value=st.session_state.start_lon, format="%.3f"
)
s_lat = st.sidebar.number_input(
    "起點緯度", value=st.session_state.start_lat, format="%.3f"
)

st.session_state.start_lon = s_lon
st.session_state.start_lat = s_lat

# ===============================
# Sidebar：終點設定
# ===============================
st.sidebar.header("🎯 目的地設定")
e_lon = st.sidebar.number_input("終點經度", value=121.900, format="%.3f")
e_lat = st.sidebar.number_input("終點緯度", value=24.600, format="%.3f")

# ===============================
# AIS → LEO 通訊檢查
# ===============================
st.sidebar.header("🛰️ AIS → 低軌衛星")

def check_leo_comm(lat, lon):
    sat_lat, sat_lon = 25.0, 121.8
    radius_km = 1200
    d = np.sqrt((lat - sat_lat)**2 + (lon - sat_lon)**2) * 111
    if d < radius_km:
        return True, round(0.9 - d / radius_km * 0.3, 2)
    return False, 0.0

if st.sidebar.button("🔍 檢查通訊"):
    ok, q = check_leo_comm(s_lat, s_lon)
    if ok:
        st.sidebar.success(f"通訊可行 ✅ 穩定度 {q}")
    else:
        st.sidebar.error("通訊不可行 ❌")

# ===============================
# A* 演算法
# ===============================
def astar_search(grid, safety_map, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dy, dx in neighbors:
            nb = (current[0]+dy, current[1]+dx)
            if 0 <= nb[0] < grid.shape[0] and 0 <= nb[1] < grid.shape[1]:
                if grid[nb] == 1:
                    continue
                step = 1.414 if dy != 0 and dx != 0 else 1.0
                cost = step + safety_map[nb]*1.5
                new_g = g[current] + cost
                if new_g < g.get(nb, 1e12):
                    g[nb] = new_g
                    f = new_g + np.linalg.norm(np.array(nb)-np.array(goal))
                    came_from[nb] = current
                    heapq.heappush(open_set, (f, nb))
    return []

# ===============================
# 路徑規劃
# ===============================
if st.sidebar.button("🚀 規劃路徑"):
    with st.spinner("讀取 HYCOM 資料中..."):
        ds = xr.open_dataset(DATA_URL, decode_times=False)

        margin = 1.0
        sub = ds.sel(
            lon=slice(min(s_lon,e_lon)-margin, max(s_lon,e_lon)+margin),
            lat=slice(min(s_lat,e_lat)-margin, max(s_lat,e_lat)+margin),
            depth=0
        ).isel(time=-1).load()

        lons = sub.lon.values
        lats = sub.lat.values

        grid = np.where(np.isnan(sub.water_u.values), 1, 0)
        dist = distance_transform_edt(1-grid)
        safety = np.exp(-dist/0.5)

        iy_s, ix_s = np.abs(lats-s_lat).argmin(), np.abs(lons-s_lon).argmin()
        iy_e, ix_e = np.abs(lats-e_lat).argmin(), np.abs(lons-e_lon).argmin()

        path = astar_search(grid, safety, (iy_s,ix_s), (iy_e,ix_e))

        if not path:
            st.error("找不到路徑")
        else:
            path_lon = [s_lon] + [lons[p[1]] for p in path] + [e_lon]
            path_lat = [s_lat] + [lats[p[0]] for p in path] + [e_lat]

            # 正確比例地圖
            lon_min, lon_max = min(path_lon)-0.4, max(path_lon)+0.4
            lat_min, lat_max = min(path_lat)-0.4, max(path_lat)+0.4

            aspect = ((lon_max-lon_min)*np.cos(np.deg2rad((lat_min+lat_max)/2))) / (lat_max-lat_min)
            fig_h = 8
            fig_w = fig_h * aspect

            fig, ax = plt.subplots(figsize=(fig_w,fig_h),
                                    subplot_kw={"projection":ccrs.PlateCarree()})
            ax.set_extent([lon_min,lon_max,lat_min,lat_max])

            speed = np.sqrt(sub.water_u**2 + sub.water_v**2)
            ax.pcolormesh(lons,lats,speed,cmap="viridis",shading="auto",alpha=0.7)

            ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#222222")
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="white")

            ax.plot(path_lon,path_lat,color="magenta",lw=3,label="AI Path")
            ax.scatter(s_lon,s_lat,c="yellow",s=100,label="Start")
            ax.scatter(e_lon,e_lat,c="red",marker="*",s=200,label="Goal")

            ax.set_title("AI Marine Navigation & Obstacle Avoidance", fontsize=14)
            ax.legend(loc="lower right")
            st.pyplot(fig)

            # 下一步航點
            next_p = path[1]
            next_lon = lons[next_p[1]]
            next_lat = lats[next_p[0]]
            dist_km = np.sqrt((next_lon-s_lon)**2+(next_lat-s_lat)**2)*111

            st.subheader("🧭 下一步航行建議")
            st.write(f"➡ ({next_lat:.4f}, {next_lon:.4f})")
            st.write(f"📏 距離：約 {dist_km:.2f} km")
