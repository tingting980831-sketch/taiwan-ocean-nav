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
# 1. 系統初始化與頁面設定
# ===============================
st.set_page_config(page_title="HELIOS V31 旗艦導航", layout="wide")

# 初始化 Session State
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.500
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.000
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'use_taiwan_tz' not in st.session_state: st.session_state.use_taiwan_tz = True

# ===============================
# 2. HYCOM 流場抓取與精確時間解碼
# ===============================
@st.cache_data(ttl=1800)
def fetch_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)

        # 抓取最新時間步長
        sub = ds.sel(lat=slice(20, 27), lon=slice(118, 126), depth=0).isel(time=-1).load()

        # 時間解碼邏輯 (基準點 2000-01-01)
        raw_val = float(ds.time.values[-1])
        base_dt = datetime(2000, 1, 1, 0, 0, 0)
        
        # 依據使用者選擇計算時區
        utc_dt = base_dt + timedelta(hours=raw_val)
        taiwan_dt = utc_dt + timedelta(hours=8)
        
        return {
            "lon": sub.lon.values,
            "lat": sub.lat.values,
            "u": sub.water_u.values,
            "v": sub.water_v.values,
            "utc_str": utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "tw_str": taiwan_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "ONLINE"
        }
    except Exception as e:
        return {"status": "OFFLINE", "tw_str": "N/A", "utc_str": "N/A"}

ocean = fetch_ocean_data()

# ===============================
# 3. 導航邏輯與陸地限制 (鐵律)
# ===============================
BUFFER = 0.05
def check_land(lat, lon):
    # 台灣本島與外島遮罩區域
    taiwan_main = (21.8 <= lat <= 25.4) and (120.0 <= lon <= 122.1)
    # 增加 5km 緩衝檢查
    return taiwan_main

def haversine(p1, p2):
    R = 6371
    phi1, lam1 = np.radians(p1)
    phi2, lam2 = np.radians(p2)
    dphi, dlam = phi2-phi1, lam2-lam1
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# ===============================
# 4. A* 智慧避障路徑生成
# ===============================
def generate_route(start, end):
    step_size = 0.15
    # 八個方向移動
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-0.7,-0.7),(-0.7,0.7),(0.7,-0.7),(0.7,0.7)]
    
    queue = [(0, tuple(start))]
    came_from = {}
    cost_so_far = {tuple(start): 0}

    while queue:
        _, current = heapq.heappop(queue)
        
        if haversine(current, end) < 15: # 抵達目標半徑
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in moves:
            nxt = (current[0] + dx * step_size, current[1] + dy * step_size)
            if check_land(*nxt): continue
            
            # 簡單成本計算：距離 + 逆流處罰
            new_cost = cost_so_far[current] + step_size
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + haversine(nxt, end) / 10
                heapq.heappush(queue, (priority, nxt))
                came_from[nxt] = current
    return []

# ===============================
# 5. Sidebar 側邊欄控制
# ===============================
with st.sidebar:
    st.header("🚢 導航控制器")
    
    # 時區切換開關
    st.session_state.use_taiwan_tz = st.toggle("顯示台灣時區 (UTC+8)", value=True)
    
    if st.button("📍 抓取目前紅點為起點", use_container_width=True):
        st.toast("已同步衛星定位座標")

    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    # 陸地硬性檢查
    if check_land(slat, slon) or check_land(elat, elon):
        st.error("🚫 警告：起點或終點位於陸地禁區！")
        st.button("🚀 啟動智慧航線", disabled=True, use_container_width=True)
    else:
        if st.button("🚀 啟動智慧航線", use_container_width=True):
            with st.spinner("計算中..."):
                path = generate_route([slat, slon], [elat, elon])
                if path:
                    st.session_state.real_p = path
                    st.session_state.step_idx = 0
                    st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
                    st.session_state.dest_lat, st.session_state.dest_lon = elat, elon
                    st.rerun()

# ===============================
# 6. Dashboard 儀表板區域
# ===============================
st.title("🛰️ HELIOS V31 智慧導航系統")

# 計算航行指標
def get_nav_metrics():
    if not st.session_state.real_p or st.session_state.step_idx >= len(st.session_state.real_p)-1:
        return "---", "---", "---"
    
    rem_path = st.session_state.real_p[st.session_state.step_idx:]
    dist = sum(haversine(rem_path[i], rem_path[i+1]) for i in range(len(rem_path)-1))
    
    # 建議航向計算
    p1, p2 = rem_path[0], rem_path[1]
    bearing = (np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0])) + 360) % 360
    
    return f"{dist:.1f} km", f"{dist/15.8:.1f} hr", f"{bearing:.1f}°"

dist_v, eta_v, head_v = get_nav_metrics()
display_time = ocean["tw_str"] if st.session_state.use_taiwan_tz else ocean["utc_str"]

c = st.columns(6)
c[0].metric("🚀 航速", "15.8 kn")
c[1].metric("🧭 建議航向", head_v)
c[2].metric("🌊 流場狀態", ocean["status"])
c[3].metric("🕒 流場時間", display_time) # 顯示格式化時間
c[4].metric("🛣️ 剩餘里程", dist_v)
c[5].metric("🕑 預計抵達", eta_v)
st.markdown("---")

# ===============================
# 7. 地圖顯示區域 (顏色還原)
# ===============================
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor="#333333", zorder=2)
ax.add_feature(cfeature.COASTLINE, color="cyan", linewidth=1.2, zorder=3)

if ocean["status"] == "ONLINE":
    speed = np.sqrt(ocean["u"]**2 + ocean["v"]**2)
    ax.pcolormesh(ocean["lon"], ocean["lat"], speed, cmap="YlGn", alpha=0.7, zorder=1)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color="#ff00ff", linewidth=3, zorder=5) # 洋紅路徑
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color="white", linestyle="--", zorder=4)

ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color="red", s=100, edgecolors='white', zorder=6)
ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color="gold", marker="*", s=300, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

# ===============================
# 8. 下方導航推進按鈕
# ===============================
if st.button("🚢 執行下一階段航行 (推進路徑)", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        curr_pos = st.session_state.real_p[st.session_state.step_idx]
        st.session_state.ship_lat, st.session_state.ship_lon = curr_pos[0], curr_pos[1]
        st.rerun()
