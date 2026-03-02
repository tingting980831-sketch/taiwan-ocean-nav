import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from scipy.interpolate import make_interp_spline

# --- 1. 初始化設定 ---
st.set_page_config(page_title="HELIOS 終極航安系統", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.017
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.463
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# --- 2. 獲取即時數據 ---
@st.cache_data(ttl=3600)
def fetch_final_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_final_data()

# --- 3. 5公里安全距離檢查 ---
def is_unsafe_5km(lat, lon):
    # 台灣本島 5km 緩衝矩形
    return (21.85 <= lat <= 25.35) and (120.05 <= lon <= 122.05)

# --- 4. 頂部儀表板 ---
st.title("🛰️ HELIOS 專業導航監控 (V16 終極整合版)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速", "15.5 kn")
c2.metric("⛽ 能源紅利", "21.2%")
dist_val = f"{len(st.session_state.real_p)*0.05:.1f} nmi" if st.session_state.real_p else "0.0 nmi"
c3.metric("📏 剩餘里程", dist_val)
c4.metric("🕒 預估時間", "3.8 hrs")

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
if is_unsafe_5km(st.session_state.ship_lat, st.session_state.ship_lon):
    m1.error("❌ GNSS: 訊號中斷 (靠近陸地)")
    m2.metric("📡 衛星連線", "0 Pcs")
else:
    m1.success("🟢 GNSS: 訊號鎖定")
    m2.metric("📡 衛星連線", "12 Pcs")
m3.info(f"🌊 流場接收: {stream_status}")
m4.metric("⏱️ 數據時標", data_clock)
st.markdown("---")

# --- 5. 避障與路徑平滑算法 ---
def generate_final_path(slat, slon, dlat, dlon):
    # 繞道閘口 (確保在海上)
    WPS = {
        'N_OFF': [26.0, 121.5], 'S_OFF': [21.2, 121.0],
        'E_OFF': [23.5, 123.0], 'W_OFF': [23.5, 119.2]
    }
    pts = [[slat, slon]]
    # 跨越東西岸邏輯
    if (slon < 121.0 and dlon > 121.0) or (slon > 121.0 and dlon < 121.0):
        if (slat + dlat) / 2 > 23.8: # 繞北
            if slon < 121.0: pts.append(WPS['W_OFF'])
            else: pts.append(WPS['E_OFF'])
            pts.append(WPS['N_OFF'])
        else: # 繞南
            if slon < 121.0: pts.append(WPS['W_OFF'])
            else: pts.append(WPS['E_OFF'])
            pts.append(WPS['S_OFF'])
    pts.append([dlat, dlon])
    pts = np.array(pts)
    
    # 平滑化處理 (Spline)
    if len(pts) > 2:
        t = np.linspace(0, 1, len(pts))
        t_new = np.linspace(0, 1, 200)
        x_smooth = make_interp_spline(t, pts[:, 1], k=min(2, len(pts)-1))(t_new)
        y_smooth = make_interp_spline(t, pts[:, 0], k=min(2, len(pts)-1))(t_new)
        return list(zip(y_smooth, x_smooth))
    else:
        return [(pts[0,0] + (pts[1,0]-pts[0,0])*t, pts[0,1] + (pts[1,1]-pts[0,1])*t) for t in np.linspace(0, 1, 200)]

# --- 6. 側邊欄與導航控制 ---
with st.sidebar:
    st.header("🚢 導航控制器")
    mode = st.radio("定位模式", ["📍 立即定位 (板橋)", "⌨️ 手動輸入"])
    if mode == "📍 立即定位 (板橋)":
        s_lat, s_lon = 25.017, 121.463
    else:
        s_lat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
        s_lon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    
    e_lat = st.number_input("終點緯度", value=22.500, format="%.3f")
    e_lon = st.number_input("終點經度", value=122.500, format="%.3f")

    if is_unsafe_5km(s_lat, s_lon) or is_unsafe_5km(e_lat, e_lon):
        st.error("🚫 座標位於 5km 陸地禁航區")
        btn = st.button("🚀 啟動導航", disabled=True)
    else:
        if st.button("🚀 啟動 AI 平滑導航", use_container_width=True):
            st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
            st.session_state.real_p = generate_final_path(s_lat, s_lon, e_lat, e_lon)
            st.session_state.step_idx = 0
            st.rerun()

# --- 7. 地圖繪製 (灰色台灣 + 流向箭頭) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.OCEAN, facecolor='#001a33')
ax.add_feature(cfeature.LAND, facecolor='#404040', zorder=2) # 台灣灰色
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

if ocean_data is not None:
    lons, lats = ocean_data.lon.values, ocean_data.lat.values
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    
    # 繪製底圖
    ax.pcolormesh(lons, lats, speed, cmap='YlGn', alpha=0.7, zorder=1)
    
    # 核心新增：流向箭頭 (Quiver)
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], 
              color='white', alpha=0.5, scale=18, width=0.002, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=6)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, edgecolors='white', zorder=7)
    ax.scatter(e_lon, e_lat, color='gold', marker='*', s=250, zorder=8)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx = min(st.session_state.step_idx + 10, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
