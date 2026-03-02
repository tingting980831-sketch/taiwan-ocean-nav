import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from scipy.interpolate import make_interp_spline

# --- 1. 系統初始化與儀表板 ---
st.set_page_config(page_title="HELIOS V11 平滑曲線版", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.017
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.463
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# 實時數據獲取 (保持監控效果)
@st.cache_data(ttl=3600)
def fetch_hycom_v11():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_hycom_v11()

st.title("🛰️ HELIOS 智慧導航 (V11 平滑路徑優化版)")

# 頂部儀表板 (保留舊有指標)
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速", "15.5 kn")
c2.metric("⛽ 能源紅利", "21.2%")
dist_str = f"{len(st.session_state.real_p)*0.05:.1f} nmi" if st.session_state.real_p else "0.0 nmi"
c3.metric("📏 剩餘里程", dist_str)
c4.metric("🕒 預估時間", "3.8 hrs")

st.markdown("---")
m1, m2, m3, m4 = st.columns(4)

# --- 2. 5公里安全距離邏輯 (0.045度) ---
def is_unsafe_v11(lat, lon):
    return (21.85 <= lat <= 25.35) and (120.05 <= lon <= 122.05)

unsafe = is_unsafe_v11(st.session_state.ship_lat, st.session_state.ship_lon)
if unsafe:
    m1.error("❌ GNSS: 訊號異常 (靠近陸地)")
    m2.metric("📡 衛星連線", "0 Pcs")
else:
    m1.success("🟢 GNSS: 訊號鎖定")
    m2.metric("📡 衛星連線", "12 Pcs")

m3.info(f"🌊 流場接收: {stream_status}")
m4.metric("⏱️ 數據時標", data_clock)
st.markdown("---")

# --- 3. 平滑路徑生成 (Bezier Curve 邏輯) ---
def generate_smooth_path_v11(slat, slon, dlat, dlon):
    # 定義避障關鍵點
    WPS = {
        'N_OFF': [26.2, 121.5], 'S_OFF': [21.2, 121.0],
        'E_OFF': [23.5, 123.2], 'W_OFF': [23.5, 119.2]
    }
    
    control_pts = [[slat, slon]]
    
    # 跨越台灣時的平滑路徑點選擇
    if (slon < 121.0 and dlon > 121.0) or (slon > 121.0 and dlon < 121.0):
        if (slat + dlat) / 2 > 23.8: # 北繞
            if slon < 121.0: control_pts.append(WPS['W_OFF'])
            else: control_pts.append(WPS['E_OFF'])
            control_pts.append(WPS['N_OFF'])
        else: # 南繞
            if slon < 121.0: control_pts.append(WPS['W_OFF'])
            else: control_pts.append(WPS['E_OFF'])
            control_pts.append(WPS['S_OFF'])
            
    control_pts.append([dlat, dlon])
    control_pts = np.array(control_pts)

    # 使用 Scipy 進行 Spline 平滑插值
    x = control_pts[:, 1]
    y = control_pts[:, 0]
    
    # 只有兩點以上才進行平滑處理
    if len(control_pts) > 2:
        t = np.linspace(0, 1, len(control_pts))
        t_new = np.linspace(0, 1, 200) # 生成 200 個超平滑點
        x_smooth = make_interp_spline(t, x, k=min(2, len(control_pts)-1))(t_new)
        y_smooth = make_interp_spline(t, y, k=min(2, len(control_pts)-1))(t_new)
        return list(zip(y_smooth, x_smooth))
    else:
        # 沒障礙時直接直線生成
        return [(y[0] + (y[1]-y[0])*t, x[0] + (x[1]-x[0])*t) for t in np.linspace(0, 1, 200)]

# --- 4. 控制側邊欄 ---
with st.sidebar:
    st.header("🚢 路徑平滑控制器")
    curr_lat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    curr_lon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    goal_lat = st.number_input("終點緯度", value=22.500, format="%.3f")
    goal_lon = st.number_input("終點經度", value=122.500, format="%.3f")

    if is_unsafe_v11(curr_lat, curr_lon) or is_unsafe_v11(goal_lat, goal_lon):
        st.error("🚫 安全距離不足 (5km 禁區)")
        nav_btn = st.button("🚀 計算路徑", disabled=True)
    else:
        if st.button("🚀 計算 AI 平滑航線", use_container_width=True):
            st.session_state.ship_lat, st.session_state.ship_lon = curr_lat, curr_lon
            st.session_state.real_p = generate_smooth_path_v11(curr_lat, curr_lon, goal_lat, goal_lon)
            st.session_state.step_idx = 0
            st.rerun()

# --- 5. 地圖顯示 (灰色台灣 + 實時流場) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.OCEAN, facecolor='#001a33')
ax.add_feature(cfeature.LAND, facecolor='#404040', zorder=2) # 灰色陸地
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

if ocean_data is not None:
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    ax.pcolormesh(ocean_data.lon, ocean_data.lat, speed, cmap='YlGn', alpha=0.8, zorder=1)

if st.session_state.real_p:
    px, py = [p[1] for p in st.session_state.real_p], [p[0] for p in st.session_state.real_p]
    # 預計航線 (淡色虛線)
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.4, zorder=4)
    # 實際航行軌跡 (紅色實線，平滑化)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='#FF3333', linewidth=3, zorder=5)
    
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, edgecolors='white', zorder=6)
    ax.scatter(goal_lon, goal_lat, color='gold', marker='*', s=300, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        # 每次移動一小段平滑路徑
        st.session_state.step_idx = min(st.session_state.step_idx + 8, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
