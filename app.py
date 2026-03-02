import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import pandas as pd

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS V8 航路監控系統", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.017
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.463
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# --- 2. 衛星與數據後端 ---
@st.cache_data(ttl=3600)
def fetch_hycom_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        # 取得最新時間點
        latest_time_idx = -1
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=latest_time_idx).load()
        
        # 格式化時間顯示
        raw_time = ds.time.values[latest_time_idx]
        data_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 簡化顯示，實際應用可解析 HYCOM Time
        return subset, data_time, "ONLINE"
    except:
        return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_hycom_data()

# --- 3. 陸地禁區檢查 ---
def is_on_land(lat, lon):
    return (21.8 <= lat <= 25.4) and (120.0 <= lon <= 122.1)

# --- 4. 擴充儀表板 (保留舊的，新增衛星與流場監控) ---
st.title("🛰️ HELIOS 智慧導航監控中心")

# 第一層：舊有航行數據
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速 (SOG)", "15.5 kn")
c2.metric("⛽ 能源紅利", "21.2%")
dist = f"{len(st.session_state.real_p)*0.1:.1f} nmi" if st.session_state.real_p else "0.0 nmi"
c3.metric("📏 剩餘里程", dist)
c4.metric("🕒 預估時間", "4.2 hrs")

# 第二層：新增的衛星與數據連線監控
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)

# 衛星連線邏輯
on_land = is_on_land(st.session_state.ship_lat, st.session_state.ship_lon)
if on_land:
    m1.error("❌ GNSS: 訊號中斷 (陸地禁入)")
    sat_count = 0
else:
    m1.success("🟢 GNSS: 訊號鎖定 (穩定)")
    sat_count = 12

m2.metric("📡 衛星連線顆數", f"{sat_count} Pcs")

# 即時流場接收狀態
if stream_status == "ONLINE":
    m3.success(f"🌊 流場接收: {stream_status}")
else:
    m3.error(f"🌊 流場接收: {stream_status}")

m4.metric("⏱️ 數據時間戳記", data_clock)
st.markdown("---")

# --- 5. 導航演算法 (弧形避障) ---
def generate_v8_path(slat, slon, dlat, dlon):
    WPS = {
        'NW': [26.5, 119.5], 'N': [26.8, 121.5], 'NE': [26.5, 123.5],
        'SW': [20.8, 119.5], 'S': [20.5, 121.0], 'SE': [20.8, 123.0]
    }
    route = [[slat, slon]]
    if (slon < 121.0 and dlon > 121.0) or (slon > 121.0 and dlon < 121.0):
        if (slat + dlat) / 2 > 23.8:
            route.extend([WPS['NW'], WPS['N'], WPS['NE']])
        else:
            route.extend([WPS['SW'], WPS['S'], WPS['SE']])
    route.append([dlat, dlon])
    
    final_path = []
    for i in range(len(route)-1):
        p1, p2 = route[i], route[i+1]
        for t in np.linspace(0, 1, 100):
            final_path.append((p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t))
    return final_path

# --- 6. 側邊欄控制 ---
with st.sidebar:
    st.header("⚙️ 設定控制")
    mode = st.radio("📍 起始定位模式", ["立即定位 (板橋)", "手動輸入"])
    if mode == "立即定位 (板橋)":
        curr_lat, curr_lon = 25.017, 121.463
    else:
        curr_lat = st.number_input("起始緯度", value=st.session_state.ship_lat, format="%.3f")
        curr_lon = st.number_input("起始經度", value=st.session_state.ship_lon, format="%.3f")

    d_lat = st.number_input("終點緯度", value=22.500, format="%.3f")
    d_lon = st.number_input("終點經度", value=122.500, format="%.3f")

    if is_on_land(curr_lat, curr_lon) or is_on_land(d_lat, d_lon):
        st.error("警告：座標位於灰色陸地區域！")
        allow_nav = False
    else:
        allow_nav = True

    if st.button("🚀 啟動 AI 安全導航", disabled=not allow_nav, use_container_width=True):
        st.session_state.ship_lat, st.session_state.ship_lon = curr_lat, curr_lon
        st.session_state.real_p = generate_v8_path(curr_lat, curr_lon, d_lat, d_lon)
        st.session_state.step_idx = 0
        st.rerun()

# --- 7. 地圖繪製 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.OCEAN, facecolor='#001a33')
ax.add_feature(cfeature.LAND, facecolor='#404040', zorder=2) # 灰色台灣
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

if ocean_data is not None:
    u = ocean_data.water_u.values
    v = ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    ax.pcolormesh(ocean_data.lon, ocean_data.lat, speed, cmap='YlGn', alpha=0.9, zorder=1)

if st.session_state.real_p:
    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.6, zorder=4)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, edgecolors='white', zorder=6)
    ax.scatter(d_lon, d_lat, color='gold', marker='*', s=300, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx = min(st.session_state.step_idx + 10, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
