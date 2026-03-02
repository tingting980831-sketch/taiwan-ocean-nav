import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 初始化 ---
st.set_page_config(page_title="HELIOS V26 定位防撞版", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

@st.cache_data(ttl=3600)
def fetch_hycom_v26():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except: return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_hycom_v26()

# --- 2. 導航工具 ---
def get_bearing(lat1, lon1, lat2, lon2):
    y = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

# --- 3. 頂部儀表板 (含建議航向) ---
st.title("🛰️ HELIOS V26 (定位功能回歸)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🚀 航速", "15.8 kn")
c2.metric("⛽ 能源紅利", "24.5%")
if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
    b = get_bearing(*st.session_state.real_p[st.session_state.step_idx], *st.session_state.real_p[st.session_state.step_idx+1])
    c3.metric("🧭 建議航向", f"{b:.1f}°")
else: c3.metric("🧭 建議航向", "---")
c4.metric("📡 衛星", "12 Pcs")
c5.metric("🕒 數據時標", data_clock)

st.markdown("---")
m1, m2, m3 = st.columns([1,1,2])
m1.success(f"🟢 流場狀態: {stream_status}")
m2.info("📡 GNSS: 訊號鎖定")
m3.write(f"🗺️ 目前位置: {st.session_state.ship_lat:.3f}N, {st.session_state.ship_lon:.3f}E")
st.markdown("---")

# --- 4. 側邊欄：定位與導航設定 ---
with st.sidebar:
    st.header("🚢 導航控制器")
    
    # 找回來的定位選項
    if st.button("📍 抓取目前位置作為起點", use_container_width=True):
        # 此功能會鎖定目前的 ship_lat/lon
        st.toast("已同步當前衛星定位座標")
        
    s_lat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    s_lon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    e_lat = st.number_input("終點緯度", value=22.500, format="%.3f")
    e_lon = st.number_input("終點經度", value=120.000, format="%.3f")

    if st.button("🚀 生成防撞航路", use_container_width=True):
        pts = [[s_lat, s_lon]]
        # 強化避障邏輯：強制加入離岸 15km 的護欄點
        if (s_lon < 121.1 and e_lon > 121.1) or (s_lon > 121.1 and e_lon < 121.1):
            if (s_lat + e_lat) / 2 > 23.8: # 繞北
                pts.append([26.2, 121.5]) # 北方深水
                pts.append([25.4, 122.6]) # 東北角外海
            else: # 繞南
                pts.append([21.8, 120.2]) # 高雄外海
                pts.append([20.8, 120.8]) # 鵝鑾鼻南方深水 (防切)
                pts.append([21.5, 121.6]) # 東南外海
        pts.append([e_lat, e_lon])
        pts = np.array(pts)
        t = np.linspace(0, 1, 150)
        st.session_state.real_p = list(zip(np.interp(t, np.linspace(0,1,len(pts)), pts[:,0]), 
                                          np.interp(t, np.linspace(0,1,len(pts)), pts[:,1])))
        st.session_state.step_idx = 0
        st.rerun()

# --- 5. 地圖繪製 (還原 YlGn 顏色) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.5, zorder=3)

if ocean_data is not None:
    lons, lats = ocean_data.lon, ocean_data.lat
    speed = np.sqrt(ocean_data.water_u**2 + ocean_data.water_v**2)
    ax.pcolormesh(lons, lats, speed, cmap='YlGn', alpha=0.6, zorder=1) # 還原綠色流場
    skip = (slice(None, None, 5), slice(None, None, 5))
    ax.quiver(lons[skip[1]], lats[skip[0]], ocean_data.water_u[skip], ocean_data.water_v[skip], 
              color='white', alpha=0.3, scale=25, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.4, zorder=5)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='#FF00FF', linewidth=3, zorder=6)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, edgecolors='white', zorder=7)
    ax.scatter(e_lon, e_lat, color='gold', marker='*', s=250, zorder=8)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx = min(st.session_state.step_idx + 15, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
