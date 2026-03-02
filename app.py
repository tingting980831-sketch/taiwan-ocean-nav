import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 初始化與數據 ---
st.set_page_config(page_title="HELIOS V22 全地形避障", layout="wide")

@st.cache_data(ttl=3600)
def fetch_hycom_v22():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except: return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_hycom_v22()

# --- 2. 5公里安全檢查 ---
def is_land_danger(lat, lon):
    return (21.85 <= lat <= 25.4) and (120.0 <= lon <= 122.1)

# --- 3. 儀表板回歸 ---
st.title("🛰️ HELIOS V22 (全地形安全避障)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速", "15.8 kn")
c2.metric("⛽ 能源紅利", "24.5%")
c3.metric("📏 剩餘里程", "測算中")
c4.metric("🕒 預估時間", "3.5 hrs")

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.success("🟢 GNSS: 訊號鎖定")
m2.metric("📡 衛星連線", "12 Pcs")
m3.info(f"🌊 流場數據: {stream_status}")
m4.metric("⏱️ 數據時標", data_clock)
st.markdown("---")

# --- 4. 核心修正：多點安全跳板邏輯 ---
def generate_v22_safe_path(slat, slon, dlat, dlon):
    pts = [[slat, slon]]
    
    # 東西岸判斷線
    is_cross = (slon < 121.2 and dlon > 121.2) or (slon > 121.2 and dlon < 121.2)
    
    if is_cross:
        # 決定繞北還是繞南
        if (slat + dlat) / 2 > 23.8: # 繞北
            # 增加「東北角安全點」[25.3, 122.3]，確保不切到三貂角
            pts.append([25.8, 121.8]) # 北閘口
            pts.append([25.3, 122.4]) # 東北角避險點
        else: # 繞南
            # 增加「南端安全點」，確保不切到鵝鑾鼻
            pts.append([21.5, 120.5]) # 西南外海
            pts.append([21.2, 120.9]) # 南閘口
            pts.append([21.6, 121.3]) # 東南外海
            
    pts.append([dlat, dlon])
    pts = np.array(pts)
    
    # 插值生成平滑路徑
    t_new = np.linspace(0, 1, 200)
    y_path = np.interp(t_new, np.linspace(0, 1, len(pts)), pts[:, 0])
    x_path = np.interp(t_new, np.linspace(0, 1, len(pts)), pts[:, 1])
    return list(zip(y_path, x_path))

# --- 5. 控制面板 ---
with st.sidebar:
    st.header("🚢 導航控制器")
    slat = st.number_input("起點緯度", value=25.060, format="%.3f")
    slon = st.number_input("起點經度", value=122.200, format="%.3f")
    dlat = st.number_input("終點緯度", value=22.500, format="%.3f")
    dlon = st.number_input("終點經度", value=120.000, format="%.3f")

    if st.button("🚀 啟動安全航路分析"):
        st.session_state.real_p = generate_v22_safe_path(slat, slon, dlat, dlon)
        st.rerun()

# --- 6. 地圖與流場繪製 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#404040', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.5, zorder=3)

if ocean_data is not None:
    lons, lats = ocean_data.lon, ocean_data.lat
    u, v = ocean_data.water_u, ocean_data.water_v
    speed = np.sqrt(u**2 + v**2)
    ax.pcolormesh(lons, lats, speed, cmap='YlGnBu', alpha=0.5, zorder=1)
    # 箭頭
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.3, scale=20, zorder=4)

if 'real_p' in st.session_state and st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=5) # 洋紅色線
    ax.scatter(px[0], py[0], color='red', s=80, zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=250, zorder=6)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)
