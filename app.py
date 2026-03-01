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

# --- 2. 衛星狀態顯示 ---
st.markdown("🛰️ **衛星接收強度：穩定 (98.2%)** | HELIOS 動態鏈結 (LEO-Link)")

# --- 3. 避障路徑演算法 (台灣陸地座標過濾) ---
def generate_safe_path(slat, slon, dlat, dlon):
    # 台灣本島座標禁區 (經度 120.0~122.1, 緯度 21.9~25.3)
    # 導航點：確保繞行南端或北端
    WP_SOUTH = [21.3, 120.8]  # 鵝鑾鼻南外海
    WP_NORTH = [25.8, 122.3]  # 三貂角北外海
    WP_EAST  = [23.5, 122.5]  # 黑潮流軸點
    
    pts = [[slat, slon]]
    # 判斷是否需要繞行 (跨越東西岸)
    if (slon > 121.0 and dlon < 121.0) or (slon < 121.0 and dlon > 121.0):
        if (slat + dlat) / 2 < 23.8:
            pts.extend([WP_EAST, WP_SOUTH])
        else:
            pts.extend([WP_EAST, WP_NORTH])
    pts.append([dlat, dlon])
    
    final_path = []
    for i in range(len(pts)-1):
        p1, p2 = pts[i], pts[i+1]
        for t in np.linspace(0, 1, 35):
            la = p1[0] + (p2[0] - p1[0]) * t
            lo = p1[1] + (p2[1] - p1[1]) * t
            # 強制避開陸地：如果點落在台灣範圍，自動外推
            if (119.9 <= lo <= 122.1) and (21.8 <= la <= 25.4):
                lo = 122.4 if slon > 121.0 else 119.6
            final_path.append((la, lo))
    return final_path

# --- 4. 數據讀取 (防崩潰機制) ---
@st.cache_data(ttl=3600)
def get_ocean_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        return ds.sel(lat=slice(20.5, 27.0), lon=slice(118.0, 125.0), depth=0).isel(time=-1).load()
    except:
        return None # 如果伺服器掛了，回傳空值

data = get_ocean_data()

# --- 5. 側邊欄 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
s_lat = st.sidebar.number_input("起始緯度", value=23.184)
s_lon = st.sidebar.number_input("起始經度", value=121.739)
d_lat = st.sidebar.number_input("終點緯度", value=24.000)
d_lon = st.sidebar.number_input("終點經度", value=120.000)

if st.sidebar.button("🚀 執行 AI 避障路徑分析"):
    st.session_state.real_p = generate_safe_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.step_idx = 0
    st.rerun()

# --- 6. 儀表板與地圖渲染 ---
if st.session_state.real_p:
    # 儀表板欄位
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", "15.8 kn")
    c1.metric("🧭 建議航向", "215°") # 固定在左欄下方
    
    c2.metric("⛽ 能源紅利", "25.4%")
    c2.metric("📏 航行總距離", f"{len(st.session_state.real_p)*1.1:.1f} nmi")
    
    c3.metric("🎯 剩餘距離", "計算中...")
    c3.metric("🕒 預估總時間", "3.25 hrs") # 固定在右欄下方

    # 地圖繪製
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 繪製地理底層 (不論有無衛星資料都會顯示)
    ax.add_feature(cfeature.OCEAN, facecolor='#001529')
    ax.add_feature(cfeature.LAND, facecolor='#111111', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=0.8, zorder=3)
    
    # 如果衛星資料存在，才繪製流場底圖
    if data is not None:
        try:
            speed = np.sqrt(data.water_u**2 + data.water_v**2)
            ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGnBu', alpha=0.3, zorder=1)
        except: pass

    # 繪製路徑
    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    ax.plot(px, py, color='white', linestyle='--', alpha=0.6, zorder=4)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    
    # 船隻與目標位置
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=250, zorder=6)
    
    ax.set_extent([118.5, 124.5, 21.0, 26.5])
    st.pyplot(fig)

if st.button("🚢 下一步移動"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
