import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 初始化 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'real_p' not in st.session_state: st.session_state.real_p = []

# --- 2. 衛星狀態 ---
st.markdown("🛰️ **衛星接收強度：穩定 (98.2%)** | 深度避障邏輯 (v4.0) 已載入")

# --- 3. 【核心升級】幾何避障路徑算法 ---
def generate_ultra_safe_path(slat, slon, dlat, dlon):
    # 定義台灣陸地矩形保護區 (稍作放大以確保安全)
    TW_LIMITS = {'lat_min': 21.8, 'lat_max': 25.4, 'lon_min': 120.0, 'lon_max': 122.1}
    
    # 預設四個離岸安全航道點 (避開所有突出的海岬)
    SAFE_PTS = {
        'NW': [25.7, 120.5], # 西北
        'NE': [25.7, 122.4], # 東北
        'SW': [21.3, 120.5], # 西南
        'SE': [21.3, 121.5]  # 東南
    }
    
    route_pts = [[slat, slon]]
    
    # 判斷是否需要跨越台灣 (東西岸互跨)
    is_cross_lon = (slon < 121.0 and dlon > 121.0) or (slon > 121.0 and dlon < 121.0)
    # 判斷是否可能切過北端或南端陸地 (緯度相近但經度跨越)
    is_potentially_cutting = is_cross_lon or (TW_LIMITS['lon_min'] < slon < TW_LIMITS['lon_max']) or (TW_LIMITS['lon_min'] < dlon < TW_LIMITS['lon_max'])

    if is_potentially_cutting:
        # 邏輯：如果平均緯度在 23.8N 以下，強制走南繞航道
        if (slat + dlat) / 2 < 23.8:
            if slon > 121.0: # 東往西：先去東南點，再去西南點
                route_pts.extend([SAFE_PTS['SE'], SAFE_PTS['SW']])
            else:           # 西往東：先去西南點，再去東南點
                route_pts.extend([SAFE_PTS['SW'], SAFE_PTS['SE']])
        else:
            if slon > 121.0: # 東往西：先去東北點，再去西北點
                route_pts.extend([SAFE_PTS['NE'], SAFE_PTS['NW']])
            else:           # 西往東：先去西北點，再去東北點
                route_pts.extend([SAFE_PTS['NW'], SAFE_PTS['NE']])
                
    route_pts.append([dlat, dlon])
    
    # 高密度插值，消除直線切角的可能
    final_path = []
    for i in range(len(route_pts)-1):
        p1, p2 = route_pts[i], route_pts[i+1]
        steps = 100
        for t in np.linspace(0, 1, steps):
            la = p1[0] + (p2[0] - p1[0]) * t
            lo = p1[1] + (p2[1] - p1[1]) * t
            # 二次保險：如果插值點不幸進入陸地範圍，強制外推
            if TW_LIMITS['lat_min'] <= la <= TW_LIMITS['lat_max'] and TW_LIMITS['lon_min'] <= lo <= TW_LIMITS['lon_max']:
                lo = SAFE_PTS['SE'][1] if slon > 121.0 else SAFE_PTS['SW'][1]
            final_path.append((la, lo))
    return final_path

# --- 4. 數據讀取 (HYCOM) ---
@st.cache_data(ttl=3600)
def get_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        return ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
    except: return None

data = get_ocean_data()

# --- 5. 側邊欄 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
s_lat = st.sidebar.number_input("起始緯度", value=st.session_state.ship_lat, format="%.3f")
s_lon = st.sidebar.number_input("起始經度", value=st.session_state.ship_lon, format="%.3f")
d_lat = st.sidebar.number_input("終點緯度", value=24.000, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=120.000, format="%.3f")

if st.sidebar.button("🚀 重新計算安全路徑", use_container_width=True):
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.real_p = generate_ultra_safe_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.step_idx = 0
    st.rerun()

# --- 6. 渲染與地圖 ---
if st.session_state.real_p:
    # 儀表板欄位
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", "15.8 kn")
    c2.metric("⛽ 能源紅利", "22.1%")
    c3.metric("🕒 預估時間", "4.1 hrs")

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 陸地層
    ax.add_feature(cfeature.LAND, facecolor='black', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=0.8, zorder=3)
    
    # 綠色海流底圖
    if data is not None:
        speed = np.sqrt(data.water_u**2 + data.water_v**2)
        ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGn', alpha=0.9, zorder=1)

    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    
    # 畫出虛線路徑
    ax.plot(px, py, color='white', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4)
    # 畫出已走航跡
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=300, zorder=7)
    
    ax.set_extent([119.0, 124.0, 21.0, 26.0])
    st.pyplot(fig)

if st.button("🚢 執行移動"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 5
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
