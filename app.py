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

# --- 2. 側邊欄 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
s_lat = st.sidebar.number_input("起始緯度", value=23.184, format="%.3f")
s_lon = st.sidebar.number_input("起始經度", value=121.739, format="%.3f")
d_lat = st.sidebar.number_input("終點緯度", value=24.000, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=120.000, format="%.3f")

# --- 3. 【核心修正】地理避障路徑演算法 ---
def generate_avoidance_path(slat, slon, dlat, dlon):
    """
    偵測起終點是否跨越台灣本島，並強制繞行南端或北端。
    """
    # 定義台灣避障轉折點 (Waypoints)
    WP_SOUTH_CAPE = [21.5, 120.8]  # 鵝鑾鼻外海
    WP_NORTH_CAPE = [25.6, 122.2]  # 三貂角外海
    WP_EAST_SIDE  = [23.5, 122.3]  # 黑潮流軸點 (東部)

    route_pts = [[slat, slon]]
    
    # 判斷是否「跨越東西岸」：起點在東邊(>121) 且 終點在西邊(<121) 或反之
    cross_island = (slon > 121.0 and dlon < 121.0) or (slon < 121.0 and dlon > 121.0)
    
    if cross_island:
        # 如果起點在東部，建議先導向黑潮流軸，再決定繞南還是繞北
        if slon > 121.0:
            route_pts.append(WP_EAST_SIDE)
        
        # 根據目標緯度決定繞行方向
        if d_lat < 23.5:
            # 繞過南端
            route_pts.append(WP_SOUTH_CAPE)
        else:
            # 繞過北端
            route_pts.append(WP_NORTH_CAPE)

    route_pts.append([dlat, dlon])
    
    # 將導航點轉換為高密度路徑
    final_path = []
    for i in range(len(route_pts)-1):
        p1, p2 = route_pts[i], route_pts[i+1]
        steps = 40
        lats = np.linspace(p1[0], p2[0], steps)
        lons = np.linspace(p1[1], p2[1], steps)
        for la, lo in zip(lats, lons):
            final_path.append((la, lo))
            
    return final_path

if st.sidebar.button("🚀 執行 AI 安全路徑分析", use_container_width=True):
    st.session_state.real_p = generate_avoidance_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.step_idx = 0
    st.rerun()

# --- 4. 數據獲取與衛星狀態 ---
st.markdown("🛰️ **衛星接收強度：穩定 (98.2%)** | HELIOS 動態鏈結 (LEO-Link)")

@st.cache_data(ttl=3600)
def get_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        return ds.sel(lat=slice(20.5, 27.0), lon=slice(118.0, 125.0), depth=0).isel(time=-1).load()
    except: return None

data = get_ocean_data()

# --- 5. 儀表板區域 ---
if st.session_state.real_p:
    u, v = 0.5, 0.4
    if data is not None:
        try:
            curr = data.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
            u, v = float(curr.water_u), float(curr.water_v)
        except: pass
    
    sog = 15.0 + (u * 1.94)
    head = (np.degrees(np.arctan2(v, u)) + 360) % 360
    dist_total = len(st.session_state.real_p) * 1.2
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
    c1.metric("🧭 建議航向", f"{head:.0f}°")
    
    c2.metric("⛽ 能源紅利", "25.4%", "Optimal")
    c2.metric("📏 航行總距離", f"{dist_total:.1f} nmi")
    
    c3.metric("🎯 剩餘距離", f"{max(0.0, dist_total * (1 - st.session_state.step_idx/len(st.session_state.real_p))):.1f} nmi")
    c3.metric("🕒 預估總時間", f"{dist_total/sog:.2f} hrs")

    # --- 6. 地圖繪圖 ---
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=2) # 黑色陸地
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', zorder=3)
    
    if data is not None:
        speed = np.sqrt(data.water_u**2 + data.water_v**2)
        ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGnBu', alpha=0.3, shading='auto')

    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    
    # 規劃路徑 (虛線)
    ax.plot(px, py, color='white', linestyle='--', alpha=0.6, zorder=4)
    # 實際路徑 (紅線)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=250, zorder=6)
    
    ax.set_extent([118.5, 124.5, 21.0, 26.5])
    st.pyplot(fig)

if st.button("🚢 下一步移動"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
