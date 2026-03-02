import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 數據與系統初始化 ---
st.set_page_config(page_title="HELIOS V20 南北平衡版", layout="wide")

@st.cache_data(ttl=3600)
def fetch_hycom_v20():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except: return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_hycom_v20()

# --- 2. 核心邏輯：聰明的南北繞行判斷 ---
def generate_v20_smart_path(slat, slon, dlat, dlon):
    pts = [[slat, slon]]
    TW_LON = 121.1
    
    # 檢查是否需要跨越台灣東西岸
    is_cross = (slon < TW_LON and dlon > TW_LON) or (slon > TW_LON and dlon < TW_LON)
    
    if is_cross:
        # 計算北路徑點 (基隆北方) 與 南路徑點 (鵝鑾鼻南方)
        # 北路徑參考點：取起終點最高緯度再往北加一點安全距離
        north_gate = [max(slat, dlat, 25.4) + 0.2, 121.6]
        # 南路徑參考點：取起終點最低緯度再往南減一點安全距離
        south_gate = [min(slat, dlat, 21.8) - 0.2, 120.9]
        
        # 簡單里程評估邏輯：哪邊比較近就走哪邊
        dist_north = abs(slat - north_gate[0]) + abs(dlat - north_gate[0])
        dist_south = abs(slat - south_gate[0]) + abs(dlat - south_gate[0])
        
        if dist_north < dist_south:
            st.sidebar.success("⚓ 系統判定：往北繞行距離較短")
            pts.append(north_gate)
        else:
            st.sidebar.success("⚓ 系統判定：往南繞行距離較短")
            pts.append(south_gate)
            
    pts.append([dlat, dlon])
    pts = np.array(pts)
    
    # 使用線性插值確保不亂繞圈圈
    t = np.linspace(0, 1, len(pts))
    t_new = np.linspace(0, 1, 150)
    x_path = np.interp(t_new, t, pts[:, 1])
    y_path = np.interp(t_new, t, pts[:, 0])
    
    return list(zip(y_path, x_path))

# --- 3. UI 介面 ---
with st.sidebar:
    st.header("🚢 導航決策系統")
    s_lat = st.number_input("起點緯度", value=25.1, format="%.3f") # 預設淡水附近
    s_lon = st.number_input("起點經度", value=121.3, format="%.3f")
    e_lat = st.number_input("終點緯度", value=24.0, format="%.3f") # 預設花蓮附近
    e_lon = st.number_input("終點經度", value=122.0, format="%.3f")

    if st.button("🚀 啟動最短航路分析"):
        st.session_state.real_p = generate_v20_smart_path(s_lat, s_lon, e_lat, e_lon)
        st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
        st.rerun()

# --- 4. 地圖繪製 (包含流向箭頭) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', zorder=3)

if ocean_data is not None:
    lons, lats = ocean_data.lon.values, ocean_data.lat.values
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    ax.pcolormesh(lons, lats, speed, cmap='YlGnBu', alpha=0.5, zorder=1)
    # 流向箭頭
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.3, scale=20, zorder=4)

if 'real_p' in st.session_state and st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#00FF00', linewidth=2.5, label='Optimized Path', zorder=5) # 綠色代表優化路徑
    ax.scatter(px[0], py[0], color='red', s=60, zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=200, zorder=6)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)
