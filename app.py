import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from datetime import datetime

# ===============================
# Page
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

# ===============================
# Session State Defaults
# ===============================
defaults = {
    "ship_lat": 25.06,
    "ship_lon": 122.2,
    "dest_lat": 22.5,
    "dest_lon": 120.0,
    "real_p": [],
    "step_idx": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===============================
# HYCOM 即時流場
# ===============================
@st.cache_data(ttl=1800)
def fetch_ocean():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)

        sub = ds.sel(
            lat=slice(20, 27),
            lon=slice(118, 126),
            depth=0
        ).isel(time=-1).load()

        return (
            sub.lon.values,
            sub.lat.values,
            sub.water_u.values,
            sub.water_v.values,
            str(ds.time.values[-1]),
            "ONLINE"
        )
    except:
        lon = np.linspace(118, 126, 80)
        lat = np.linspace(20, 27, 80)
        u = np.zeros((80, 80))
        v = np.zeros((80, 80))
        return lon, lat, u, v, "N/A", "OFFLINE"

lons, lats, u, v, raw_time, status = fetch_ocean()

# 將時間轉成人類可讀格式
if status == "ONLINE":
    try:
        ocean_time = np.datetime_as_string(np.datetime64(raw_time), unit='s')
    except:
        ocean_time = str(raw_time)
else:
    ocean_time = "N/A"

# ===============================
# 台灣陸地 + 5km buffer
# ===============================
BUFFER_DEG = 0.045
def is_land(lat, lon):
    taiwan = (21.8 - BUFFER_DEG <= lat <= 25.4 + BUFFER_DEG) and (120.0 - BUFFER_DEG <= lon <= 122.1 + BUFFER_DEG)
    china = (lat >= 24.2 - BUFFER_DEG and lon <= 119.6 + BUFFER_DEG)
    return taiwan or china

# ===============================
# 距離計算
# ===============================
def haversine(a, b):
    R = 6371
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(h), np.sqrt(1 - h))

# ===============================
# 取得流速
# ===============================
def current_vec(lat, lon):
    i = np.abs(lats - lat).argmin()
    j = np.abs(lons - lon).argmin()
    return u[i, j], v[i, j]

# ===============================
# A* 智慧航線（考慮流場）
# ===============================
def smart_route(start, end):
    step = 0.12
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def h(p): return haversine(p, end)

    open_set = []
    heapq.heappush(open_set, (0, start))

    came = {}
    g = {tuple(start): 0}

    while open_set:
        _, cur = heapq.heappop(open_set)

        if haversine(cur, end) < 20:
            path = [cur]
            while tuple(cur) in came:
                cur = came[tuple(cur)]
                path.append(cur)
            return path[::-1]

        for dx, dy in dirs:
            nxt = [cur[0] + dx * step, cur[1] + dy * step]
            if is_land(*nxt):
                continue

            base_speed = 12
            cu, cv = current_vec(*nxt)
            current_speed = np.sqrt(cu**2 + cv**2)
            effective_speed = max(4, base_speed + current_speed * 2)
            cost = step / effective_speed
            new_g = g[tuple(cur)] + cost

            if tuple(nxt) not in g or new_g < g[tuple(nxt)]:
                g[tuple(nxt)] = new_g
                f = new_g + h(nxt)
                heapq.heappush(open_set, (f, nxt))
                came[tuple(nxt)] = cur

    return []

# ===============================
# Sidebar 控制
# ===============================
with st.sidebar:
    st.header("🚢 導航控制")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    if st.button("📍 定位目前位置"):
        st.session_state.ship_lat = slat
        st.session_state.ship_lon = slon

    if st.button("🚀 啟動智慧航線", use_container_width=True):
        if is_land(slat, slon):
            st.error("起點在陸地")
        elif is_land(elat, elon):
            st.error("終點在陸地")
        else:
            path = smart_route([slat, slon], [elat, elon])
            st.session_state.ship_lat = slat
            st.session_state.ship_lon = slon
            st.session_state.dest_lat = elat
            st.session_state.dest_lon = elon
            st.session_state.real_p = path
            st.session_state.step_idx = 0
            st.rerun()

# ===============================
# 儀表板
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")

# 模擬即時航速 & 衛星
speed_now = 12 + np.random.uniform(-1.5, 1.5)
satellite = int(9 + np.random.randint(0, 6))

# 計算距離 & ETA & 建議航向
if st.session_state.real_p:
    start = [st.session_state.ship_lat, st.session_state.ship_lon]
    end = [st.session_state.dest_lat, st.session_state.dest_lon]
    total_distance = haversine(start, end)
    eta_hours = total_distance / speed_now
    eta_str = f"{int(eta_hours)} h {int((eta_hours%1)*60)} m"

    # 建議航向
    dy = end[0] - start[0]
    dx = end[1] - start[1]
    heading = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
else:
    total_distance = 0
    eta_str = "--"
    heading = "--"

r1 = st.columns(4)
r1[0].metric("🚀 即時航速", f"{speed_now:.1f} kn")
r1[1].metric("📡 衛星數量", satellite)
r1[2].metric("🕒 流場時間", ocean_time)
r1[3].metric("🧭 建議航向", f"{heading} °")

r2 = st.columns(3)
r2[0].metric("📏 剩餘距離", f"{total_distance:.1f} km")
r2[1].metric("🕒 預計航行時間", eta_str)
r2[2].metric("🌊 流場狀態", status)

st.markdown("---")

# ===============================
# 地圖
# ===============================
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND, facecolor="#2b2b2b")
ax.add_feature(cfeature.COASTLINE, edgecolor="cyan")

speed = np.sqrt(u**2 + v**2)
ax.pcolormesh(lons, lats, speed, cmap="YlGn", alpha=0.7)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color="#ff00ff", linewidth=3)

ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color="red", s=80, zorder=6)
ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color="gold", marker="*", s=250, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)
