import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS V15 實時流向監控", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.017
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.463
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# --- 2. 數據獲取 (確保縮排正確) ---
@st.cache_data(ttl=3600)
def fetch_hycom_v15():
    try:
        # 使用即時數據網址
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        # 抓取台灣周邊最新時標
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except Exception as e:
        return None, "N/A", f"OFFLINE: {str(e)[:20]}"

ocean_data, data_clock, stream_status = fetch_hycom_v15()

# --- 3. 儀表板 (保留所有監控) ---
st.title("🛰️ HELIOS V15 實時流向與衛星監控")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速", "15.5 kn")
c2.metric("⛽ 能源紅利", "21.2%")
dist_str = f"{len(st.session_state.real_p)*0.1:.1f} nmi" if st.session_state.real_p else "0.0 nmi"
c3.metric("📏 剩餘里程", dist_str)
c4.metric("🕒 預估時間", "4.2 hrs")

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.success("🟢 GNSS: 訊號鎖定")
m2.metric("📡 衛星連線", "12 Pcs")
m3.info(f"🌊 流場接收: {stream_status}")
m4.metric("⏱️ 數據時標", data_clock)
st.markdown("---")

# --- 4. 側邊欄控制 ---
with st.sidebar:
    st.header("🚢 導航設定")
    s_lat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    s_lon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    d_lat = st.number_input("終點緯度", value=22.500, format="%.3f")
    d_lon = st.number_input("終點經度", value=122.500, format="%.3f")

    if st.button("🚀 計算路徑", use_container_width=True):
        # 簡單的兩點連線 (為了示範流場，暫不加入複雜避障)
        st.session_state.real_p = [(s_lat + (d_lat-s_lat)*t, s_lon + (d_lon-s_lon)*t) for t in np.linspace(0, 1, 100)]
        st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
        st.rerun()

# --- 5. 地圖繪製 (加入箭頭) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.OCEAN, facecolor='#001a33')
ax.add_feature(cfeature.LAND, facecolor='#404040', zorder=2) # 灰色台灣
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

if ocean_data is not None:
    lon, lat = ocean_data.lon.values, ocean_data.lat.values
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    
    # 1. 繪製流速熱圖 (底色)
    im = ax.pcolormesh(lon, lat, speed, cmap='YlGn', alpha=0.7, zorder=1)
    
    # 2. 繪製流向箭頭 (Quiver)
    # 每隔 3 個網格點畫一個箭頭，避免太擠
    skip = (slice(None, None, 3), slice(None, None, 3))
    ax.quiver(lon[skip[1]], lat[skip[0]], u[skip], v[skip], 
              color='white', alpha=0.6, scale=15, width=0.002, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='magenta', linewidth=2, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, zorder=6)
    ax.scatter(d_lon, d_lat, color='gold', marker='*', s=200, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)
