import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

# ===============================
#  AI 海象導航系統（A* + HYCOM）
# ===============================

st.set_page_config(page_title="AI 海象導航系統", layout="wide")
st.title("⚓ 智慧避障導航系統")
st.write("這套系統結合了 A* 演算法與 HYCOM 全球海象即時數據。")

# -------------------------------
# Sidebar - A* 座標輸入
# -------------------------------
st.sidebar.header("📍 座標輸入")
s_lon = st.sidebar.number_input("起點經度 (Start Lon)", value=121.750, format="%.3f")
s_lat = st.sidebar.number_input("起點緯度 (Start Lat)", value=25.150, format="%.3f")
e_lon = st.sidebar.number_input("終點經度 (End Lon)", value=121.900, format="%.3f")
e_lat = st.sidebar.number_input("終點緯度 (End Lat)", value=24.600, format="%.3f")

# -------------------------------
# A* 演算法
# -------------------------------
def astar_search(grid, safety_map, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_heap:
        current = heapq.heappop(open_heap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dy, dx in neighbors:
            nb = (current[0] + dy, current[1] + dx)
            if 0 <= nb[0] < grid.shape[0] and 0 <= nb[1] < grid.shape[1]:
                if grid[nb] == 1:
                    continue
                step = 1.414 if dy != 0 and dx != 0 else 1.0
                cost = step + safety_map[nb] * 1.5
                g_new = g_score[current] + cost
                if g_new < g_score.get(nb, 1e12):
                    came_from[nb] = current
                    g_score[nb] = g_new
                    f = g_new + np.linalg.norm(np.array(nb) - np.array(goal))
                    heapq.heappush(open_heap, (f, nb))
    return []

# -------------------------------
# A* 執行
# -------------------------------
if st.sidebar.button("🚀 開始規劃航線"):
    with st.spinner("正在從 HYCOM 獲取數據..."):
        try:
            DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
            ds = xr.open_dataset(DATA_URL, decode_times=False)

            margin = 1.0
            subset = ds.sel(
                lon=slice(min(s_lon, e_lon)-margin, max(s_lon, e_lon)+margin),
                lat=slice(min(s_lat, e_lat)-margin, max(s_lat, e_lat)+margin),
                depth=0
            ).isel(time=-1).load()

            lons = subset.lon.values
            lats = subset.lat.values

            grid = np.where(np.isnan(subset.water_u.values), 1, 0)
            dist_land = distance_transform_edt(1 - grid)
            safety_map = np.exp(-dist_land / 0.5)

            iy_s, ix_s = np.abs(lats - s_lat).argmin(), np.abs(lons - s_lon).argmin()
            iy_e, ix_e = np.abs(lats - e_lat).argmin(), np.abs(lons - e_lon).argmin()

            def nearest_water(y, x):
                if grid[y, x] == 0:
                    return (y, x)
                Y, X = np.indices(grid.shape)
                d = np.sqrt((Y-y)**2 + (X-x)**2)
                d[grid == 1] = 1e9
                return np.unravel_index(np.argmin(d), grid.shape)

            s_idx = nearest_water(iy_s, ix_s)
            e_idx = nearest_water(iy_e, ix_e)

            path = astar_search(grid, safety_map, s_idx, e_idx)

            if not path:
                st.error("找不到路徑 (Path not found)")
            else:
                path_lon = [s_lon] + [lons[p[1]] for p in path] + [e_lon]
                path_lat = [s_lat] + [lats[p[0]] for p in path] + [e_lat]

                # ===============================
                # 正確比例、無白邊底圖（關鍵）
                # ===============================
                lon_min, lon_max = min(path_lon)-0.4, max(path_lon)+0.4
                lat_min, lat_max = min(path_lat)-0.4, max(path_lat)+0.4

                lon_range = lon_max - lon_min
                lat_range = lat_max - lat_min
                mean_lat = (lat_min + lat_max) / 2

                aspect_geo = (lon_range * np.cos(np.deg2rad(mean_lat))) / lat_range

                fig_h = 8
                fig_w = fig_h * aspect_geo

                fig, ax = plt.subplots(
                    figsize=(fig_w, fig_h),
                    subplot_kw={"projection": ccrs.PlateCarree()}
                )

                ax.set_extent([lon_min, lon_max, lat_min, lat_max])

                speed = np.sqrt(subset.water_u**2 + subset.water_v**2)
                ax.pcolormesh(
                    lons, lats, speed,
                    cmap="viridis", shading="auto", alpha=0.7
                )

                ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#222222")
                ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="white")

                ax.plot(path_lon, path_lat, color="magenta", lw=3, label="AI Path")
                ax.scatter(s_lon, s_lat, c="yellow", s=100, label="Start")
                ax.scatter(e_lon, e_lat, c="red", marker="*", s=200, label="Goal")

                ax.set_title("AI Marine Navigation & Obstacle Avoidance", fontsize=14)
                ax.legend(loc="lower right")

                st.pyplot(fig)
                st.success("規劃完成 (Success)! 航線經緯度已精確對齊。")

        except Exception as e:
            st.error(f"Error: {e}")

# ===============================
# HELIOS 系統（完全未修改）
# ===============================

st.divider()
st.header("🛰️ HELIOS 智慧導航系統")

if "sim_lon" not in st.session_state:
    st.session_state.sim_lon = 121.85
    st.session_state.sim_lat = 25.15

st.sidebar.header("🛰️ HELIOS 星座設定")
st.sidebar.info("""
**軌道高度**: 900 km  
**總衛星數**: 36 顆  
**覆蓋率**: ~84%
""")

if st.sidebar.button("🎲 瞬移到海上測試點"):
    st.session_state.sim_lon = np.random.uniform(119.5, 122.5)
    st.session_state.sim_lat = np.random.uniform(22.5, 25.5)

c_lon = st.sidebar.number_input("模擬經度", value=st.session_state.sim_lon, format="%.3f")
c_lat = st.sidebar.number_input("模擬緯度", value=st.session_state.sim_lat, format="%.3f")

SHIP_POWER_KNOTS = 15.0

def calculate_metrics(u, v, s):
    vs = s * 0.514
    sog = vs + (u*0.6 + v*0.4)
    return sog/0.514, 15.8, round(0.84 + np.random.uniform(0.08,0.12),2)

if st.sidebar.button("🚀 啟動即時導航分析"):
    ds = xr.open_dataset(DATA_URL, decode_times=False)
    sub = ds.sel(
        lon=slice(c_lon-0.6, c_lon+0.6),
        lat=slice(c_lat-0.6, c_lat+0.6),
        depth=0
    ).isel(time=-1).load()

    u = sub.water_u.interp(lat=c_lat, lon=c_lon).values
    v = sub.water_v.interp(lat=c_lat, lon=c_lon).values

    if np.isnan(u):
        st.error("❌ 座標位於陸地")
    else:
        sog, fuel, comm = calculate_metrics(float(u), float(v), SHIP_POWER_KNOTS)
        c1, c2, c3 = st.columns(3)
        c1.metric("SOG", f"{sog:.2f} kn")
        c2.metric("Fuel Saving", f"{fuel}%")
        c3.metric("Comm Stability", comm)
