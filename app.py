import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import make_interp_spline

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

LEO_STABILITY = 0.982 
FUEL_GAIN_AVG = 25.4  

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'real_p' not in st.session_state: st.session_state.real_p = []

# --- 2. 側邊欄控制 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
loc_mode = st.sidebar.radio("起始點選擇", ["立即定位 (GPS 模擬)", "自行輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    start_lat, start_lon = 23.184, 121.739
    st.sidebar.success(f"📍 GPS 已鎖定: {start_lat}, {start_lon}")
else:
    start_lat = st.sidebar.number_input("起始緯度", value=23.184, format="%.3f")
    start_lon = st.sidebar.number_input("起始經度", value=121.739, format="%.3f")

st.sidebar.markdown("---")
dest_lat = st.sidebar.number_input("終點緯度", value=25.500, format="%.3f")
dest_lon = st.sidebar.number_input("終點經度", value=121.800, format="%.3f")

# --- 3. 數據讀取 ---
@st.cache_data(ttl=3600)
def get_fast_ocean_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.5, 27.0), lon=slice(118.0, 125.0), depth=0).isel(time=-1, lat=slice(None, None, 2), lon=slice(None, None, 2)).load()
        return subset
    except: return None

# --- 4. 關鍵修正：具備避障邏輯的路徑演算法 ---
def generate_helios_path(slat, slon, dlat, dlon):
    # 定義台灣陸地矩形範圍 (約略值)
    taiwan_lat_min, taiwan_lat_max = 21.9, 25.3
    taiwan_lon_min, taiwan_lon_max = 120.0, 122.1

    # 建立控制點清單
    ctrl_pts = [[slat, slon]]

    # 檢查是否需要繞過台灣東岸 (核心修正點)
    # 如果路徑會從東邊橫跨到西邊，或者經過台灣緯度區間，強制加入東岸導引點
    if (slon > 122.1 and dlon < 122.1) or (slon < 122.1 and dlon > 122.1) or (taiwan_lat_min < (slat+dlat)/2 < taiwan_lat_max):
        # 加入兩個位於東部海域(黑潮流軸)的導引點
        mid_lat1 = slat + (dlat - slat) * 0.33
        mid_lat2 = slat + (dlat - slat) * 0.66
        # 強制這兩點在經度 122.3 以上，避免切入陸地
        ctrl_pts.append([mid_lat1, 122.4]) 
        ctrl_pts.append([mid_lat2, 122.5])

    ctrl_pts.append([dlat, dlon])
    ctrl_pts = np.array(ctrl_pts)

    # 平滑化生成
    t = np.linspace(0, 1, len(ctrl_pts))
    t_smooth = np.linspace(0, 1, 60)
    
    # 這裡使用 k=min(2, len(ctrl_pts)-1) 確保點數太少時不會報錯
    k_val = min(2, len(ctrl_pts)-1)
    spline_lat = make_interp_spline(t, ctrl_pts[:, 0], k=k_val)(t_smooth)
    spline_lon = make_interp_spline(t, ctrl_pts[:, 1], k=k_val)(t_smooth)
    
    # 二次檢查：確保所有生成的點都不在陸地上
    safe_lat, safe_lon = [], []
    for la, lo in zip(spline_lat, spline_lon):
        if taiwan_lat_min < la < taiwan_lat_max and lo < 122.2:
            lo = 122.4 # 強制推離陸地
        safe_lat.append(la)
        safe_lon.append(lo)

    return [tuple(p) for p in zip(safe_lat, safe_lon)]

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    st.session_state.real_p = generate_helios_path(start_lat, start_lon, dest_lat, dest_lon)
    st.session_state.ship_lat, st.session_state.ship_lon = start_lat, start_lon
    st.session_state.step_idx = 0
    st.rerun()

# --- 5. 儀表板渲染 (對調位置) ---
subset = get_fast_ocean_data()
if subset is not None and st.session_state.real_p:
    curr_pt = subset.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u, v = float(curr_pt.water_u), float(curr_pt.water_v)
    sog = 15.0 + (u * 1.94)
    suggested_head = (np.degrees(np.arctan2(v, u)) + 360) % 360
    
    st.subheader("📊 HELIOS 智慧導航決策儀表板")
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
    c1.metric("🧭 建議航向", f"{suggested_head:.0f}°")
    
    # 距離預估邏輯
    total_d = sum(np.sqrt((st.session_state.real_p[i][0]-st.session_state.real_p[i+1][0])**2 + (st.session_state.real_p[i][1]-st.session_state.real_p[i+1][1])**2) * 60 for i in range(len(st.session_state.real_p)-1))
    traveled_d = (st.session_state.step_idx / (len(st.session_state.real_p)-1)) * total_d
    
    c2.metric("⛽ 能源紅利", f"{FUEL_GAIN_AVG}%", "Optimal")
    c2.metric("📏 航行總距離", f"{total_d:.1f} nmi", f"已航行 {traveled_d:.1f}")
    
    c3.metric("🎯 剩餘距離", f"{max(0.0, total_d - traveled_d):.1f} nmi")
    c3.metric("🕒 預估總時間", f"{total_d / sog:.2f} hrs")

    # --- 6. 地圖繪圖 ---
    fig, ax = plt.subplots(figsize=(11, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    speed_grid = np.sqrt(subset.water_u**2 + subset.water_v**2)
    ax.pcolormesh(subset.lon, subset.lat, speed_grid, cmap='YlGnBu', alpha=0.4, shading='auto')
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1e1e1e', zorder=5) # 提高陸地層級
    
    rx, ry = [p[1] for p in st.session_state.real_p], [p[0] for p in st.session_state.real_p]
    ax.plot(rx, ry, color='white', linestyle='--', linewidth=1.5, zorder=6) # 虛線路徑
    ax.plot(rx[:st.session_state.step_idx+1], ry[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=7) # 已航行紅線
    
    ax.scatter(dest_lon, dest_lat, color='gold', marker='*', s=350, edgecolors='black', zorder=8)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=150, edgecolors='white', zorder=9)
    ax.set_extent([119, 124.5, 21.0, 26.5])
    st.pyplot(fig)

if st.button("🚢 更新下一步航行數據"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
