import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 初始化設定 ---
st.set_page_config(page_title="HELIOS 導航系統 V5", layout="wide")

# 預設位置 (板橋)
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.017
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.463
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# --- 2. 固定儀表板 (放在最上面，確保不消失) ---
st.title("🛰️ HELIOS 智慧避障導航系統")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速 (SOG)", "16.2 kn")
c2.metric("⛽ 能源紅利", "24.8%")
dist_val = f"{len(st.session_state.real_p)*0.1:.1f} nmi" if st.session_state.real_p else "0.0 nmi"
c3.metric("📏 總航程", dist_val)
c4.metric("🕒 預估時間", "4.5 hrs")

# --- 3. 避障核心 V5 (多點弧形繞道) ---
def generate_v5_path(slat, slon, dlat, dlon):
    # 定義更遠、更安全的安全點 (確保連線不切到陸地)
    SAFE_NW = [26.2, 120.0]
    SAFE_N  = [26.2, 121.5]
    SAFE_NE = [26.2, 123.0]
    SAFE_SW = [21.0, 120.0]
    SAFE_S  = [21.0, 121.0]
    SAFE_SE = [21.0, 122.0]
    
    route_pts = [[slat, slon]]
    
    # 判斷是否跨越東西岸 (以經度 121.0 為界)
    is_cross = (slon > 121.1 and dlon < 120.9) or (slon < 120.9 and dlon > 121.1)
    
    if is_cross:
        # 繞北路徑 (三點轉向，形成圓弧)
        if (slat + dlat) / 2 > 23.5:
            if slon > 121.1: route_pts.extend([SAFE_NE, SAFE_N, SAFE_NW])
            else: route_pts.extend([SAFE_NW, SAFE_N, SAFE_NE])
        # 繞南路徑 (三點轉向)
        else:
            if slon > 121.1: route_pts.extend([SAFE_SE, SAFE_S, SAFE_SW])
            else: route_pts.extend([SAFE_SW, SAFE_S, SAFE_SE])
            
    route_pts.append([dlat, dlon])
    
    # 超高密度插值 (每段 100 點)
    final_p = []
    for i in range(len(route_pts)-1):
        p1, p2 = route_pts[i], route_pts[i+1]
        for t in np.linspace(0, 1, 100):
            final_p.append((p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t))
    return final_p

# --- 4. 數據讀取 (HYCOM) ---
@st.cache_data(ttl=3600)
def get_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        return ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
    except: return None

data = get_ocean_data()

# --- 5. 側邊欄控制 ---
st.sidebar.header("🧭 導航控制中心")
mode = st.sidebar.radio("起始點選擇", ["📍 立即定位 (GPS 模擬)", "⌨️ 自行輸入座標"])

if mode == "📍 立即定位 (GPS 模擬)":
    # 固定為板橋座標
    curr_lat, curr_lon = 25.017, 121.463
    st.sidebar.info(f"📍 GPS 定位成功: {curr_lat}, {curr_lon}")
else:
    curr_lat = st.sidebar.number_input("起始緯度", value=st.session_state.ship_lat, format="%.3f")
    curr_lon = st.sidebar.number_input("起始經度", value=st.session_state.ship_lon, format="%.3f")

d_lat = st.sidebar.number_input("終點緯度", value=22.500, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=122.500, format="%.3f")

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    st.session_state.ship_lat, st.session_state.ship_lon = curr_lat, curr_lon
    st.session_state.real_p = generate_v5_path(curr_lat, curr_lon, d_lat, d_lon)
    st.session_state.step_idx = 0
    st.rerun()

# --- 6. 地圖顯示 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# 底圖風格：灰色台灣 + 亮色海岸線
ax.add_feature(cfeature.OCEAN, facecolor='#001a33')
ax.add_feature(cfeature.LAND, facecolor='#555555', zorder=2) 
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

# 經典綠色海流底圖
if data is not None:
    speed = np.sqrt(data.water_u**2 + data.water_v**2)
    ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGn', alpha=0.9, zorder=1)

if st.session_state.real_p:
    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    
    # 預計路徑 (白虛線)
    ax.plot(px, py, color='white', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4)
    # 實際路徑 (紅實線)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    
    # 標記起終點
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=6)
    ax.scatter(d_lon, d_lat, color='gold', marker='*', s=350, edgecolors='black', zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

# --- 7. 下一步按鈕 ---
if st.button("🚢 下一步移動", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx = min(st.session_state.step_idx + 10, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
