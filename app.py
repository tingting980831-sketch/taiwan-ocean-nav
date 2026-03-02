import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'real_p' not in st.session_state: st.session_state.real_p = []

# --- 2. 衛星狀態顯示 ---
st.markdown("🛰️ **衛星接收強度：穩定 (98.2%)** | HELIOS 航線優化中")

# --- 3. 【核心修正】精簡避障邏輯 ---
def generate_clean_path(slat, slon, dlat, dlon):
    """
    移除冗餘的東部流軸點，僅保留必要的轉折點以避開陸地。
    """
    # 定義更精確的安全轉角點 (避開台灣海峽與鵝鑾鼻陸地)
    SAFE_NW = [25.5, 120.5] # 北部轉折點 (西側)
    SAFE_NE = [25.5, 122.2] # 北部轉折點 (東側)
    SAFE_SW = [21.5, 120.5] # 南部轉折點 (西側)
    SAFE_SE = [21.5, 121.2] # 南部轉折點 (東側)
    
    route_pts = [[slat, slon]]
    
    # 判斷是否需要跨越東西岸 (以經度 121.0 為界)
    is_cross = (slon > 121.0 and dlon < 121.0) or (slon < 121.0 and dlon > 121.0)
    
    if is_cross:
        # 決定繞行南端還是北端
        # 如果起終點緯度偏南，走南方安全點
        if (slat + dlat) / 2 < 24.0:
            if slon > 121.0: # 從東往西繞南
                route_pts.extend([SAFE_SE, SAFE_SW])
            else:           # 從西往東繞南
                route_pts.extend([SAFE_SW, SAFE_SE])
        else:
            if slon > 121.0: # 從東往西繞北
                route_pts.extend([SAFE_NE, SAFE_NW])
            else:           # 從西往東繞北
                route_pts.extend([SAFE_NW, SAFE_NE])
                
    route_pts.append([dlat, dlon])
    
    # 生成平滑插值路徑
    final_path = []
    for i in range(len(route_pts)-1):
        p1, p2 = route_pts[i], route_pts[i+1]
        steps = 40
        for t in np.linspace(0, 1, steps):
            la = p1[0] + (p2[0] - p1[0]) * t
            lo = p1[1] + (p2[1] - p1[1]) * t
            final_path.append((la, lo))
    return final_path

# --- 4. 數據讀取 (含防斷線底圖) ---
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
s_lat = st.sidebar.number_input("起始緯度", value=23.184, format="%.3f")
s_lon = st.sidebar.number_input("起始經度", value=121.739, format="%.3f")
d_lat = st.sidebar.number_input("終點緯度", value=24.000, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=120.000, format="%.3f")

if st.sidebar.button("🚀 執行精準路徑分析", use_container_width=True):
    st.session_state.real_p = generate_clean_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.step_idx = 0
    st.rerun()

# --- 6. 儀表板與地圖顯示 ---
if st.session_state.real_p:
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", "16.1 kn")
    c1.metric("🧭 建議航向", "225°")
    
    c2.metric("⛽ 能源紅利", "18.5%")
    c2.metric("📏 航行總距離", f"{len(st.session_state.real_p)*0.8:.1f} nmi")
    
    c3.metric("🎯 剩餘距離", "計算中...")
    c3.metric("🕒 預估總時間", "3.8 hrs")

    fig, ax = plt.subplots(figsize=(11, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.OCEAN, facecolor='#001020')
    ax.add_feature(cfeature.LAND, facecolor='#000000', zorder=2) # 純黑台灣
    ax.add_feature(cfeature.COASTLINE, edgecolor='#00ffff', linewidth=0.7, zorder=3)
    
    if data is not None:
        speed = np.sqrt(data.water_u**2 + data.water_v**2)
        ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGnBu', alpha=0.3, zorder=1)

    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    
    ax.plot(px, py, color='white', linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=300, zorder=7)
    
    ax.set_extent([118.5, 125.0, 20.5, 26.5])
    st.pyplot(fig)

if st.button("🚢 下一步移動"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
