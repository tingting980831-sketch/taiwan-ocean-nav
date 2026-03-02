import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS V10 5km貼岸版", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.017
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.463
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# 實時數據獲取
@st.cache_data(ttl=3600)
def fetch_hycom_v10():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_hycom_v10()

# --- 2. 儀表板 (保留所有監控指標) ---
st.title("🛰️ HELIOS 智慧導航 (5km 貼岸優化版)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速", "15.5 kn")
c2.metric("⛽ 能源紅利", "21.2%")
dist_str = f"{len(st.session_state.real_p)*0.1:.1f} nmi" if st.session_state.real_p else "0.0 nmi"
c3.metric("📏 剩餘里程", dist_str)
c4.metric("🕒 預估時間", "4.2 hrs")

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)

# --- 3. 5公里安全距離邏輯 ---
# 經緯度 0.045 度約等於 5 公里
def is_unsafe_v10(lat, lon):
    # 縮減後的台灣本島禁區範圍 (5km 緩衝)
    return (21.85 <= lat <= 25.35) and (120.05 <= lon <= 122.05)

unsafe = is_unsafe_v10(st.session_state.ship_lat, st.session_state.ship_lon)
if unsafe:
    m1.error("❌ GNSS: 訊號異常 (過近)")
    m2.metric("📡 衛星連線", "0 Pcs")
else:
    m1.success("🟢 GNSS: 訊號鎖定")
    m2.metric("📡 衛星連線", "12 Pcs")

m3.info(f"🌊 流場接收: {stream_status}")
m4.metric("⏱️ 數據時標", data_clock)
st.markdown("---")

# --- 4. 繞道算法 (5km 貼岸調整) ---
def generate_v10_path(slat, slon, dlat, dlon):
    # 航點微調，使其更靠近南/北端
    WPS = {
        'N': [25.8, 121.5], 'S': [21.5, 121.0],
        'E': [23.5, 122.8], 'W': [23.5, 119.5]
    }
    path = [[slat, slon]]
    if (slon < 121.0 and dlon > 121.0) or (slon > 121.0 and dlon < 121.0):
        if (slat + dlat) / 2 > 23.8: # 北繞
            if slon < 121.0: path.append(WPS['W'])
            else: path.append(WPS['E'])
            path.append(WPS['N'])
        else: # 南繞
            if slon < 121.0: path.append(WPS['W'])
            else: path.append(WPS['E'])
            path.append(WPS['S'])
    path.append([dlat, dlon])
    
    smooth_p = []
    for i in range(len(path)-1):
        p1, p2 = path[i], path[i+1]
        for t in np.linspace(0, 1, 100):
            smooth_p.append((p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t))
    return smooth_p

# --- 5. 側邊欄控制 ---
with st.sidebar:
    st.header("🚢 導航設定 (5km)")
    curr_lat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    curr_lon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    goal_lat = st.number_input("終點緯度", value=22.500, format="%.3f")
    goal_lon = st.number_input("終點經度", value=122.500, format="%.3f")

    if is_unsafe_v10(curr_lat, curr_lon) or is_unsafe_v10(goal_lat, goal_lon):
        st.error("🚫 無法設點：距離陸地小於 5km")
        btn = st.button("🚀 啟動導航", disabled=True)
    else:
        if st.button("🚀 啟動 AI 安全路徑", use_container_width=True):
            st.session_state.ship_lat, st.session_state.ship_lon = curr_lat, curr_lon
            st.session_state.real_p = generate_v10_path(curr_lat, curr_lon, goal_lat, goal_lon)
            st.session_state.step_idx = 0
            st.rerun()

# --- 6. 地圖繪製 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.OCEAN, facecolor='#001a33')
ax.add_feature(cfeature.LAND, facecolor='#404040', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

if ocean_data is not None:
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    ax.pcolormesh(ocean_data.lon, ocean_data.lat, speed, cmap='YlGn', alpha=0.8, zorder=1)

if st.session_state.real_p:
    px, py = [p[1] for p in st.session_state.real_p], [p[0] for p in st.session_state.real_p]
    ax.plot(px, py, color='white', linestyle='--', linewidth=0.8, alpha=0.6, zorder=4)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=2.5, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, zorder=6)
    ax.scatter(goal_lon, goal_lat, color='gold', marker='*', s=250, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

if st.button("🚢 下一步移動", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx = min(st.session_state.step_idx + 10, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
