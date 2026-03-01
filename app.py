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

# --- 2. 側邊欄控制 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
s_lat = st.sidebar.number_input("起始緯度", value=23.184, format="%.3f")
s_lon = st.sidebar.number_input("起始經度", value=121.739, format="%.3f")
d_lat = st.sidebar.number_input("終點緯度", value=24.000, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=120.000, format="%.3f")

# --- 3. 【核心修正】避障路徑生成演算法 ---
def generate_safe_path(slat, slon, dlat, dlon):
    # 定義安全轉折點 (Waypoints)
    WP_SOUTH = [21.8, 120.8]  # 鵝鑾鼻南方海域
    WP_NORTH = [25.5, 122.0]  # 三貂角東北方海域
    WP_EAST  = [23.5, 122.2]  # 東部黑潮流軸區
    
    route_points = [[slat, slon]]
    
    # 判斷是否需要繞過台灣 (跨越經度 121.0)
    needs_bypass = (slon > 121.0 and dlon < 121.0) or (slon < 121.0 and dlon > 121.0)
    
    if needs_bypass:
        # 判斷往南繞還是往北繞較近
        if (slat + dlat) / 2 < 23.8:
            # 往南繞：先到東部流軸 -> 繞過南端 -> 抵達西部
            route_points.append(WP_EAST)
            route_points.append(WP_SOUTH)
        else:
            # 往北繞：先到東部流軸 -> 繞過北端 -> 抵達西部
            route_points.append(WP_EAST)
            route_points.append(WP_NORTH)
            
    route_points.append([dlat, dlon])
    
    # 高密度線性插值，確保路徑不消失且平滑
    final_path = []
    for i in range(len(route_points)-1):
        p1, p2 = route_points[i], route_points[i+1]
        steps = 40
        for la, lo in zip(np.linspace(p1[0], p2[0], steps), np.linspace(p1[1], p2[1], steps)):
            final_path.append((la, lo))
    return final_path

if st.sidebar.button("🚀 執行 AI 安全路徑分析", use_container_width=True):
    st.session_state.real_p = generate_safe_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.step_idx = 0
    st.rerun()

# --- 4. 數據與衛星狀態 ---
st.markdown("🛰️ **衛星接收強度：穩定 (98.2%)** | HELIOS 動態鏈結 (LEO-Link)")

@st.cache_data(ttl=3600)
def get_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        return ds.sel(lat=slice(20.5, 26.5), lon=slice(118.0, 125.0), depth=0).isel(time=-1).load()
    except: return None

data = get_ocean_data()

# --- 5. 儀表板區域 ---
if st.session_state.real_p:
    u, v = 0.5, 0.3
    if data is not None:
        curr = data.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
        u, v = float(curr.water_u), float(curr.water_v)
    
    sog = 15.0 + (u * 1.94)
    head = (np.degrees(np.arctan2(v, u)) + 360) % 360
    dist_total = len(st.session_state.real_p) * 1.2
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
    c1.metric("🧭 建議航向", f"{head:.0f}°") # 固定左下
    
    c2.metric("⛽ 能源紅利", "12.5%")
    c2.metric("📏 航行總距離", f"{dist_total:.1f} nmi")
    
    c3.metric("🎯 剩餘距離", f"{max(0.0, dist_total - st.session_state.step_idx*1.2):.1f} nmi")
    c3.metric("🕒 預估總時間", f"{dist_total/sog:.2f} hrs") # 固定右下

    # --- 6. 地圖繪圖 (保證底圖存在) ---
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor='#1a1a1a', zorder=2) # 黑色台灣陸地
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', zorder=3)
    
    if data is not None:
        speed = np.sqrt(data.water_u**2 + data.water_v**2)
        ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGnBu', alpha=0.3)

    # 繪製路徑
    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    ax.plot(px, py, color='white', linestyle='--', alpha=0.5, zorder=4) # 規劃虛線
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=250, zorder=6) # 終點星號
    
    ax.set_extent([118.5, 124.5, 21.0, 26.5])
    st.pyplot(fig)

if st.button("🚢 更新航行數據 (下一步)"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
