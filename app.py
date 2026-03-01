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

# --- 2. 側邊欄 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
loc_mode = st.sidebar.radio("起始點選擇", ["立即定位 (GPS 模擬)", "自行輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    s_lat, s_lon = 23.184, 121.739
    st.sidebar.success(f"📍 GPS 已鎖定: {s_lat}, {s_lon}")
else:
    s_lat = st.sidebar.number_input("起始緯度", value=23.184, format="%.3f")
    s_lon = st.sidebar.number_input("起始經度", value=121.739, format="%.3f")

d_lat = st.sidebar.number_input("終點緯度", value=24.000, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=120.000, format="%.3f")

# --- 3. 路徑生成演算法 (關鍵修正) ---
def generate_safe_path(slat, slon, dlat, dlon):
    # 定義轉折點 (例如：台灣南端繞行點)
    # 如果起點在東部，終點在西部，必須繞過南部 (約 21.8, 120.8) 或北部
    points = [[slat, slon]]
    
    # 簡單避障邏輯：如果兩地被台灣阻隔 (經度跨越 121.0)
    if (slon > 121.5 and dlon < 120.5) or (slon < 120.5 and dlon > 121.5):
        # 判斷往南繞還是往北繞較近
        if (slat + dlat) / 2 < 23.5:
            points.append([21.8, 120.8]) # 鵝鑾鼻外海
        else:
            points.append([25.5, 122.0]) # 三貂角外海
            
    points.append([dlat, dlon])
    
    # 使用線性插值產生高密度航點 (確保路徑不消失)
    final_path = []
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        num_sub_steps = 25
        lats = np.linspace(p1[0], p2[0], num_sub_steps)
        lons = np.linspace(p1[1], p2[1], num_sub_steps)
        for la, lo in zip(lats, lons):
            final_path.append((la, lo))
            
    return final_path

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    st.session_state.real_p = generate_safe_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.step_idx = 0
    st.rerun()

# --- 4. 數據讀取與繪圖 ---
@st.cache_data(ttl=3600)
def get_ocean_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        return ds.sel(lat=slice(21.0, 26.5), lon=slice(118.5, 124.5), depth=0).isel(time=-1).load()
    except: return None

data = get_ocean_data()
if data is not None and st.session_state.real_p:
    # 儀表板計算
    curr_pt = data.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u, v = float(curr_pt.water_u), float(curr_pt.water_v)
    sog = 15.0 + (u * 1.94)
    head = (np.degrees(np.arctan2(v, u)) + 360) % 360
    
    st.subheader("📊 HELIOS 智慧導航儀表板")
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
    c1.metric("🧭 建議航向", f"{head:.0f}°")
    
    # 顯示總距離預估 (根據 path 長度)
    total_dist = len(st.session_state.real_p) * 2.5 # 粗略估算
    c2.metric("📏 航行總距離", f"{total_dist:.1f} nmi")
    c3.metric("🕒 預估總時間", f"{total_dist/sog:.2f} hrs")

    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor='#222', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', zorder=3)
    
    # 繪製完整路徑 (虛線)
    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.7, zorder=4)
    
    # 繪製已航行路徑 (紅線)
    idx = st.session_state.step_idx
    ax.plot(px[:idx+1], py[:idx+1], color='red', linewidth=3, zorder=5)
    
    # 標點
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=200, zorder=6)
    
    ax.set_extent([119, 124, 21, 26.5])
    st.pyplot(fig)

if st.button("🚢 下一步"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
