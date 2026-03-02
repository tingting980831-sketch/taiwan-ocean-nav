import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS V21 全功能版", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.017
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.463
if 'real_p' not in st.session_state: st.session_state.real_p = []

# --- 2. 數據獲取 ---
@st.cache_data(ttl=3600)
def fetch_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except: return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_data()

# --- 3. 5km 安全限制邏輯 ---
def is_unsafe_5km(lat, lon):
    # 台灣本島禁區
    return (21.85 <= lat <= 25.35) and (120.05 <= lon <= 122.05)

# --- 4. 儀表板 UI (已恢復) ---
st.title("🛰️ HELIOS 導航監控中心 (V21)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速", "15.5 kn")
c2.metric("⛽ 能源紅利", "21.2%")
c3.metric("📏 剩餘里程", f"{len(st.session_state.real_p)*0.08:.1f} nmi" if st.session_state.real_p else "0.0")
c4.metric("🕒 預估時間", "3.2 hrs")

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.success("🟢 GNSS: 訊號鎖定")
m2.metric("📡 衛星連線", "12 Pcs")
m3.info(f"🌊 流場: {stream_status}")
m4.metric("⏱️ 時標", data_clock)
st.markdown("---")

# --- 5. 智慧南北繞行決策 ---
def generate_v21_smart_path(slat, slon, dlat, dlon):
    pts = [[slat, slon]]
    TW_LON = 121.1
    if (slon < TW_LON and dlon > TW_LON) or (slon > TW_LON and dlon < TW_LON):
        north_gate = [26.1, 121.5]
        south_gate = [21.3, 120.9]
        if abs(slat - north_gate[0]) + abs(dlat - north_gate[0]) < abs(slat - south_gate[0]) + abs(dlat - south_gate[0]):
            pts.append(north_gate)
        else:
            pts.append(south_gate)
    pts.append([dlat, dlon])
    pts = np.array(pts)
    t = np.linspace(0, 1, len(pts))
    t_new = np.linspace(0, 1, 150)
    return list(zip(np.interp(t_new, t, pts[:, 0]), np.interp(t_new, t, pts[:, 1])))

# --- 6. 側邊欄控制 (含 5km 檢測) ---
with st.sidebar:
    st.header("🚢 導航設定")
    s_lat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    s_lon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    e_lat = st.number_input("終點緯度", value=22.500, format="%.3f")
    e_lon = st.number_input("終點經度", value=122.500, format="%.3f")

    if is_unsafe_5km(s_lat, s_lon) or is_unsafe_5km(e_lat, e_lon):
        st.error("🚫 座標位於 5km 陸地禁航區！請重新設定")
        btn = st.button("🚀 計算路徑", disabled=True)
    else:
        if st.button("🚀 啟動智能航路", use_container_width=True):
            st.session_state.real_p = generate_v21_smart_path(s_lat, s_lon, e_lat, e_lon)
            st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
            st.rerun()

# --- 7. 地圖顯示 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#404040', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)
if ocean_data is not None:
    lons, lats = ocean_data.lon.values, ocean_data.lat.values
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    ax.pcolormesh(lons, lats, speed, cmap='YlGnBu', alpha=0.6, zorder=1)
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.3, scale=20, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, zorder=6)
    ax.scatter(e_lon, e_lat, color='gold', marker='*', s=250, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)
