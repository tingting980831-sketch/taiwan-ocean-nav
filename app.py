import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 數據與系統初始化 ---
st.set_page_config(page_title="HELIOS V25 全功能導航", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

@st.cache_data(ttl=3600)
def fetch_hycom_v25():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except: return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_hycom_v25()

# --- 2. 航向計算邏輯 ---
def calculate_bearing(lat1, lon1, lat2, lon2):
    y = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))
    bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return bearing

# --- 3. 頂部儀表板 (新增建議航向) ---
st.title("🛰️ HELIOS V25 專業導航系統")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🚀 航速", "15.8 kn")
c2.metric("⛽ 能源紅利", "24.5%")

# 建議航向邏輯
if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
    curr = st.session_state.real_p[st.session_state.step_idx]
    nxt = st.session_state.real_p[st.session_state.step_idx + 1]
    brg = calculate_bearing(curr[0], curr[1], nxt[0], nxt[1])
    c3.metric("🧭 建議航向", f"{brg:.1f}°")
else:
    c3.metric("🧭 建議航向", "---")

c4.metric("📏 狀態", "安全監控中")
c5.metric("🕒 預估時間", "3.5 hrs")

st.markdown("---")

# --- 4. 消失的衛星與流場資訊 (找回來了！) ---
m1, m2, m3, m4 = st.columns(4)
# 5公里禁航區檢查
is_danger = (21.85 <= st.session_state.ship_lat <= 25.35) and (120.05 <= st.session_state.ship_lon <= 122.05)

if is_danger:
    m1.error("❌ GNSS: 訊號中斷 (禁區)")
    m2.metric("📡 衛星連線", "0 Pcs")
else:
    m1.success("🟢 GNSS: 訊號鎖定")
    m2.metric("📡 衛星連線", "12 Pcs")

m3.info(f"🌊 流場數據: {stream_status}")
m4.metric("⏱️ 數據時標", data_clock)
st.markdown("---")

# --- 5. 側邊欄：導航設定 ---
with st.sidebar:
    st.header("🚢 導航設定")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    dlat = st.number_input("終點緯度", value=22.500, format="%.3f")
    dlon = st.number_input("終點經度", value=120.000, format="%.3f")

    if st.button("🚀 啟動智能航路"):
        # 避障路徑生成 (強化南端護欄，不切陸地，不亂繞)
        pts = [[slat, slon]]
        if (slon < 121.2 and dlon > 121.2) or (slon > 121.2 and dlon < 121.2):
            if (slat + dlat) / 2 > 23.8: # 繞北
                pts.append([26.0, 121.5])
                pts.append([25.5, 122.5])
            else: # 繞南
                pts.append([21.8, 120.4])
                pts.append([20.9, 120.8]) # 往南推更多防止切到
                pts.append([21.5, 121.5])
        pts.append([dlat, dlon])
        pts = np.array(pts)
        t_new = np.linspace(0, 1, 150)
        st.session_state.real_p = list(zip(np.interp(t_new, np.linspace(0, 1, len(pts)), pts[:, 0]),
                                          np.interp(t_new, np.linspace(0, 1, len(pts)), pts[:, 1])))
        st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
        st.session_state.step_idx = 0
        st.rerun()

# --- 6. 地圖繪製 (顏色還原至原始版本) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2) # 原始灰色
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.5, zorder=3) # 原始螢光藍

if ocean_data is not None:
    lons, lats = ocean_data.lon.values, ocean_data.lat.values
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    # 還原流場顏色為 YlGn (黃綠色)
    ax.pcolormesh(lons, lats, speed, cmap='YlGn', alpha=0.6, zorder=1)
    # 箭頭
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.4, scale=20, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='#FF00FF', linewidth=3, zorder=6)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, edgecolors='white', zorder=7)
    ax.scatter(dlon, dlat, color='gold', marker='*', s=250, zorder=8)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

# --- 7. 下一階段航行按鈕 ---
if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx = min(st.session_state.step_idx + 10, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
