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
# Page Configuration
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統 V29", layout="wide")

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
# HYCOM 流場抓取與時間解碼 (核心修正)
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

        # --- 時間解析修正邏輯 ---
        raw_time = ds.time.values[-1]
        units = ds.time.attrs.get('units', "hours since 2000-01-01 00:00:00")
        origin_str = units.replace("hours since ", "")
        
        # 轉換為台灣時間 (UTC+8)
        dt_base = pd.to_datetime(raw_time, unit='h', origin=origin_str)
        dt_taiwan = dt_base + timedelta(hours=8)
        ocean_time_str = dt_taiwan.strftime("%Y-%m-%d %H:%M")

        return sub.lon.values, sub.lat.values, sub.water_u.values, sub.water_v.values, ocean_time_str, "ONLINE"
    except Exception as e:
        lon = np.linspace(118, 126, 80)
        lat = np.linspace(20, 27, 80)
        u = np.zeros((80, 80))
        v = np.zeros((80, 80))
        return lon, lat, u, v, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, status = fetch_ocean()

# ===============================
# 台灣陸地 + 5km 限制 (鐵律)
# ===============================
BUFFER_DEG = 0.05  # 約 5.5 公里緩衝
def is_land(lat, lon):
    # 台灣本島主要陸地遮罩
    taiwan = (21.85 - BUFFER_DEG <= lat <= 25.35 + BUFFER_DEG) and \
             (120.05 - BUFFER_DEG <= lon <= 122.05 + BUFFER_DEG)
    # 簡單過濾中國大陸沿岸
    china = (lat >= 24.0 and lon <= 119.5)
    return taiwan or china

# ===============================
# 導航工具計算
# ===============================
def haversine(a, b):
    R = 6371
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(h), np.sqrt(1-h))

def current_vec(lat, lon):
    i = np.abs(lats - lat).argmin()
    j = np.abs(lons - lon).argmin()
    return u[i, j], v[i, j]

# ===============================
# A* 智慧航線 (防亂繞優化)
# ===============================
def smart_route(start, end):
    step = 0.15 # 步長
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-0.7,-0.7),(-0.7,0.7),(0.7,-0.7),(0.7,0.7)]
    def h(p): return haversine(p, end)
    
    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))
    came = {}
    g = {tuple(start): 0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if haversine(cur, end) < 18: # 接近目標則視為到達
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]

        for dx, dy in dirs:
            nxt = (cur[0] + dx * step, cur[1] + dy * step)
            if is_land(*nxt): continue
            
            # 流場影響權重
            cu, cv = current_vec(*nxt)
            eff_speed = max(5, 12 + (cu*dx + cv*dy)*2) # 考慮流向貢獻
            cost = step / eff_speed
            
            new_g = g[cur] + cost
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                f = new_g + h(nxt) / 12 # 啟發函數權重調整
                heapq.heappush(open_set, (f, nxt))
                came[nxt] = cur
    return []

# ===============================
# Sidebar (找回定位功能 + 陸地限制)
# ===============================
with st.sidebar:
    st.header("🚢 導航控制中心")
    if st.button("📍 抓取目前紅點位置為起點", use_container_width=True):
        st.session_state.ship_lat = st.session_state.ship_lat
        st.session_state.ship_lon = st.session_state.ship_lon
        st.toast("定位已鎖定")

    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    # 檢查是否位於陸地
    locked = is_land(slat, slon) or is_land(elat, elon)
    if locked:
        st.error("🚫 錯誤：起點或終點位於陸地禁區")
        st.button("🚀 啟動智慧航線", disabled=True, use_container_width=True)
    else:
        if st.button("🚀 啟動智慧航線", use_container_width=True):
            with st.spinner("正在計算最佳避障路徑..."):
                path = smart_route([slat, slon], [elat, elon])
                if path:
                    st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
                    st.session_state.dest_lat, st.session_state.dest_lon = elat, elon
                    st.session_state.real_p = path
                    st.session_state.step_idx = 0
                    st.rerun()
                else:
                    st.warning("無法找到可行路徑，請嘗試調整位置")

# ===============================
# Dashboard Indicators
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")

def route_stats():
    if not st.session_state.real_p or st.session_state.step_idx >= len(st.session_state.real_p)-1:
        return "---", "---", "---"
    
    # 剩餘里程
    rem_path = st.session_state.real_p[st.session_state.step_idx:]
    dist = sum(haversine(rem_path[i], rem_path[i+1]) for i in range(len(rem_path)-1))
    
    # 建議航向 (下一跳)
    p1, p2 = rem_path[0], rem_path[1]
    dx, dy = p2[1] - p1[1], p2[0] - p1[0]
    heading = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    
    eta = dist / 15.8
    return f"{dist:.1f} km", f"{eta:.1f} hr", f"{heading:.1f}°"

dist_val, eta_val, head_val = route_stats()

c = st.columns(6)
c[0].metric("🚀 即時航速", "15.8 kn")
c[1].metric("🧭 建議航向", head_val)
c[2].metric("🌊 流場狀態", status)
c[3].metric("🕒 流場時間", ocean_time)
c[4].metric("🛣️ 剩餘航程", dist_val)
c[5].metric("🕑 預計抵達", eta_val)
st.markdown("---")

# ===============================
# Map (顏色與圖標完全還原)
# ===============================
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor="#333333", zorder=2)
ax.add_feature(cfeature.COASTLINE, color="cyan", linewidth=1.2, zorder=3)

speed_m = np.sqrt(u**2 + v**2)
im = ax.pcolormesh(lons, lats, speed_m, cmap="YlGn", alpha=0.7, zorder=1)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color="#ff00ff", linewidth=3, alpha=0.9, zorder=5)
    # 繪製已行駛過的虛線
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color="white", linewidth=1, linestyle="--", zorder=4)

ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color="red", s=100, edgecolors='white', zorder=6)
ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color="gold", marker="*", s=300, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

# ===============================
# Bottom Control
# ===============================
if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        curr = st.session_state.real_p[st.session_state.step_idx]
        st.session_state.ship_lat, st.session_state.ship_lon = curr[0], curr[1]
        st.rerun()
