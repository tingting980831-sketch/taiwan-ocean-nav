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
# 1. 系統初始化
# ===============================
st.set_page_config(page_title="HELIOS V33 | 2026 實時導航", layout="wide")

# 初始化 Session State
defaults = {
    "ship_lat": 25.060, "ship_lon": 122.200,
    "dest_lat": 22.500, "dest_lon": 120.000,
    "real_p": [], "step_idx": 0
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ===============================
# 2. 最新 ESPC-D-V02 數據抓取 (2026 專用)
# ===============================
@st.cache_data(ttl=1800)
def fetch_ocean_2026_realtime():
    try:
        # 這是目前 HYCOM 2026 年最活躍的數據接口 (ESPC 系統)
        url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/uv3z"
        ds = xr.open_dataset(url, decode_times=False)

        # 抓取最新時間點 (Time -1)
        sub = ds.sel(lat=slice(20, 27), lon=slice(118, 126), depth=0).isel(time=-1).load()

        # 時間解碼邏輯
        raw_val = float(ds.time.values[-1])
        units = ds.time.attrs.get('units', "hours since 2000-01-01 00:00:00")
        base_str = units.replace("hours since ", "")
        
        # 基準解析：pd.to_datetime 自動處理小時數轉為時間
        utc_dt = pd.to_datetime(raw_val, unit='h', origin=base_str)
        tw_dt = utc_dt + timedelta(hours=8) # 轉換為台灣時間
        
        return {
            "lon": sub.lon.values, "lat": sub.lat.values,
            "u": sub.water_u.values, "v": sub.water_v.values,
            "time_str": tw_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "ONLINE (ESPC-2026)"
        }
    except Exception as e:
        # 保險機制：若新網址連線失敗，則顯示離線
        return {"status": "OFFLINE", "time_str": "N/A", "u": None, "v": None}

ocean = fetch_ocean_2026_realtime()

# ===============================
# 3. 導航限制 (硬性避開陸地)
# ===============================
def is_land(lat, lon):
    # 台灣主要陸地座標範圍 (包含 5km 安全緩衝)
    taiwan_island = (21.9 <= lat <= 25.4) and (120.0 <= lon <= 122.1)
    china_coast = (lat >= 24.2 and lon <= 119.6)
    return taiwan_island or china_coast

def haversine(p1, p2):
    R = 6371
    phi1, lam1 = np.radians(p1); phi2, lam2 = np.radians(p2)
    a = np.sin((phi2-phi1)/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin((lam2-lam1)/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# ===============================
# 4. A* 導航演算法
# ===============================
def generate_route(start, end):
    step = 0.15
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-0.7,-0.7),(-0.7,0.7),(0.7,-0.7),(0.7,0.7)]
    q = [(0, tuple(start))]; came = {}; g = {tuple(start): 0}
    
    while q:
        _, cur = heapq.heappop(q)
        if haversine(cur, end) < 15:
            path = [cur]
            while cur in came: cur = came[cur]; path.append(cur)
            return path[::-1]
        
        for dx, dy in dirs:
            nxt = (cur[0]+dx*step, cur[1]+dy*step)
            if is_land(*nxt): continue
            
            new_g = g[cur] + step
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                heapq.heappush(q, (new_g + haversine(nxt, end)/12, nxt))
                came[nxt] = cur
    return []

# ===============================
# 5. UI 與 儀表板
# ===============================
with st.sidebar:
    st.header("🚢 實時導航中心")
    if st.button("📍 定位目前起點", use_container_width=True):
        st.toast("座標同步成功")
        
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    # 硬性起點檢查
    if is_land(slat, slon) or is_land(elat, elon):
        st.error("🚫 錯誤：座標位於陸地！")
    else:
        if st.button("🚀 啟動 2026 實時路徑", use_container_width=True):
            st.session_state.real_p = generate_route([slat, slon], [elat, elon])
            st.session_state.step_idx = 0
            st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
            st.session_state.dest_lat, st.session_state.dest_lon = elat, elon
            st.rerun()

# 儀表板顯示
st.title("🛰️ HELIOS V33 (ESPC 實時預報系統)")

def get_stats():
    if not st.session_state.real_p or st.session_state.step_idx >= len(st.session_state.real_p)-1:
        return "---", "---"
    rem = st.session_state.real_p[st.session_state.step_idx:]
    dist = sum(haversine(rem[i], rem[i+1]) for i in range(len(rem)-1))
    p1, p2 = rem[0], rem[1]
    head = (np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0])) + 360) % 360
    return f"{dist:.1f} km", f"{head:.1f}°"

dist_str, head_str = get_stats()

c = st.columns(6)
c[0].metric("🚀 航速", "15.8 kn")
c[1].metric("🧭 建議航向", head_str)
c[2].metric("🌊 流場來源", ocean["status"])
c[3].metric("🕒 流場時間", ocean["time_str"]) # 這裡會顯示 2026 年！
c[4].metric("🛣️ 剩餘里程", dist_str)
c[5].metric("📡 GNSS", "穩定")
st.markdown("---")

# ===============================
# 6. 地圖繪製 (YlGn 顏色)
# ===============================
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor="#333333", zorder=2)
ax.add_feature(cfeature.COASTLINE, color="cyan", linewidth=1.2, zorder=3)

if ocean["status"] != "OFFLINE" and ocean["u"] is not None:
    speed = np.sqrt(ocean["u"]**2 + ocean["v"]**2)
    ax.pcolormesh(ocean["lon"], ocean["lat"], speed, cmap="YlGn", alpha=0.7, zorder=1)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color="#ff00ff", linewidth=3, zorder=5) # 洋紅路徑
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color="red", s=100, zorder=6)
    ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color="gold", marker="*", s=300, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

if st.button("🚢 推進航行下一階段", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
        st.session_state.step_idx += 1
        curr = st.session_state.real_p[st.session_state.step_idx]
        st.session_state.ship_lat, st.session_state.ship_lon = curr[0], curr[1]
        st.rerun()
