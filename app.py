import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from datetime import datetime, timedelta
import pandas as pd

# ===============================
# 1. 頁面設定與狀態初始化
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統 V30", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.500
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.000
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# ===============================
# 2. HYCOM 流場抓取與「人類可讀時間」解碼
# ===============================
@st.cache_data(ttl=1800)
def fetch_ocean_v30():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)

        sub = ds.sel(
            lat=slice(20, 27),
            lon=slice(118, 126),
            depth=0
        ).isel(time=-1).load()

        # --- 核心時間解碼邏輯 ---
        raw_time = float(ds.time.values[-1])
        # HYCOM 標準基準: 2000-01-01 00:00:00
        base_time = datetime(2000, 1, 1, 0, 0, 0)
        # 轉換為台灣時間 (UTC+8)
        actual_time = base_time + timedelta(hours=raw_time) + timedelta(hours=8)
        # 格式化為您要求的：年-月-日 時:分:秒
        ocean_time_str = actual_time.strftime("%Y-%m-%d %H:%M:%S")

        return sub.lon.values, sub.lat.values, sub.water_u.values, sub.water_v.values, ocean_time_str, "ONLINE"
    except Exception as e:
        lon = np.linspace(118, 126, 80)
        lat = np.linspace(20, 27, 80)
        return lon, lat, np.zeros((80,80)), np.zeros((80,80)), "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, status = fetch_ocean_v30()

# ===============================
# 3. 陸地限制與導航工具
# ===============================
BUFFER_DEG = 0.05
def is_land(lat, lon):
    # 台灣本島與主要離島遮罩 (含 5km 緩衝)
    taiwan = (21.85 - BUFFER_DEG <= lat <= 25.38 + BUFFER_DEG) and \
             (120.00 - BUFFER_DEG <= lon <= 122.10 + BUFFER_DEG)
    china = (lat >= 24.0 and lon <= 119.5)
    return taiwan or china

def haversine(a, b):
    R = 6371
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat, dlon = lat2-lat1, lon2-lon1
    h = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(h), np.sqrt(1-h))

def current_vec(lat, lon):
    i = np.abs(lats - lat).argmin()
    j = np.abs(lons - lon).argmin()
    return u[i, j], v[i, j]

# ===============================
# 4. A* 智慧航線演算法
# ===============================
def smart_route(start, end):
    step = 0.15
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-0.7,-0.7),(-0.7,0.7),(0.7,-0.7),(0.7,0.7)]
    def h(p): return haversine(p, end)
    
    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))
    came, g = {}, {tuple(start): 0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if haversine(cur, end) < 18:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]

        for dx, dy in dirs:
            nxt = (cur[0] + dx * step, cur[1] + dy * step)
            if is_land(*nxt): continue
            
            cu, cv = current_vec(*nxt)
            eff_speed = max(5, 12 + (cu*dx + cv*dy)*2.5)
            new_g = g[cur] + (step / eff_speed)
            
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                f = new_g + h(nxt) / 10
                heapq.heappush(open_set, (f, nxt))
                came[nxt] = cur
    return []

# ===============================
# 5. Sidebar 介面
# ===============================
with st.sidebar:
    st.header("🚢 導航控制器")
    if st.button("📍 抓取目前紅點位置為起點"):
        st.session_state.ship_lat = st.session_state.ship_lat
        st.session_state.ship_lon = st.session_state.ship_lon
        st.toast("定位已更新")

    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    on_land = is_land(slat, slon) or is_land(elat, elon)
    if on_land:
        st.error("🚫 起點或終點位於陸地禁區")
        st.button("🚀 啟動智慧航線", disabled=True, use_container_width=True)
    else:
        if st.button("🚀 啟動智慧航線", use_container_width=True):
            path = smart_route([slat, slon], [elat, elon])
            if path:
                st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
                st.session_state.dest_lat, st.session_state.dest_lon = elat, elon
                st.session_state.real_p, st.session_state.step_idx = path, 0
                st.rerun()
            else:
                st.warning("無法計算路徑，請嘗試開放水域")

# ===============================
# 6. Dashboard 與地圖
# ===============================
st.title("🛰️ HELIOS 智慧航行系統 V30")

def get_stats():
    if not st.session_state.real_p or st.session_state.step_idx >= len(st.session_state.real_p)-1:
        return "---", "---", "---"
    rem = st.session_state.real_p[st.session_state.step_idx:]
    dist = sum(haversine(rem[i], rem[i+1]) for i in range(len(rem)-1))
    p1, p2 = rem[0], rem[1]
    head = (np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0])) + 360) % 360
    return f"{dist:.1f} km", f"{dist/15.8:.1f} hr", f"{head:.1f}°"

dist_v, eta_v, head_v = get_stats()

c = st.columns(6)
c[0].metric("🚀 航速", "15.8 kn")
c[1].metric("🧭 建議航向", head_v)
c[2].metric("🌊 流場", status)
c[3].metric("🕒 流場時間", ocean_time) # 這裡現在顯示 年-月-日 時:分:秒
c[4].metric("🛣️ 剩餘里程", dist_v)
c[5].metric("🕑 ETA", eta_v)
st.markdown("---")

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor="#333333", zorder=2)
ax.add_feature(cfeature.COASTLINE, color="cyan", linewidth=1.2, zorder=3)
ax.pcolormesh(lons, lats, np.sqrt(u**2+v**2), cmap="YlGn", alpha=0.7, zorder=1)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color="#ff00ff", linewidth=3, zorder=5)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color="white", linestyle="--", zorder=4)

ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color="red", s=100, edgecolors='white', zorder=6)
ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color="gold", marker="*", s=300, zorder=7)
ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
        st.session_state.step_idx += 1
        curr = st.session_state.real_p[st.session_state.step_idx]
        st.session_state.ship_lat, st.session_state.ship_lon = curr[0], curr[1]
        st.rerun()
