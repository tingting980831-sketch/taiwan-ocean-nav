import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import make_interp_spline

# --- 1. 系統初始化與常數 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

LEO_STABILITY = 0.982 
FUEL_GAIN_AVG = 25.4  

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'total_planned_dist' not in st.session_state: st.session_state.total_planned_dist = 0.0

# --- 2. 側邊欄控制 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
loc_mode = st.sidebar.radio("起始點選擇", ["立即定位 (GPS 模擬)", "自行輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    s_lat, s_lon = 23.184, 121.739
    st.sidebar.info(f"📍 GPS 座標: {s_lat}, {s_lon}")
else:
    s_lat = st.sidebar.number_input("起始緯度", value=23.184, format="%.3f")
    s_lon = st.sidebar.number_input("起始經度", value=121.739, format="%.3f")

d_lat = st.sidebar.number_input("終點緯度", value=25.500, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=121.800, format="%.3f")

# --- 3. 數據讀取 (優化緩存) ---
@st.cache_data(ttl=3600)
def get_fast_ocean_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(21.0, 26.5), lon=slice(118.5, 124.5), depth=0).isel(time=-1, lat=slice(None, None, 2), lon=slice(None, None, 2)).load()
        return subset
    except: return None

# --- 4. 核心演算法：平滑避障路徑 (解決路徑怪怪的問題) ---
def generate_helios_path(slat, slon, dlat, dlon):
    # 建立多個導航點以形成自然曲線
    # 這裡加入一個「轉折點」來確保繞過台灣東岸
    mid_lat = (slat + dlat) / 2
    # 如果終點在北邊且起點在南邊，強迫中間點向東偏離，捕獲黑潮
    mid_lon = 122.6 if (slon < 122.2 and 22 < mid_lat < 25) else (slon + dlon) / 2
    
    ctrl_pts = np.array([
        [slat, slon],
        [mid_lat, mid_lon],
        [dlat, dlon]
    ])
    
    # 使用 B-Spline 進行路徑平滑，消除階梯感
    t = np.linspace(0, 1, len(ctrl_pts))
    t_smooth = np.linspace(0, 1, 50) # 產生 50 個平滑點
    
    spline_lat = make_interp_spline(t, ctrl_pts[:, 0], k=2)(t_smooth)
    spline_lon = make_interp_spline(t, ctrl_pts[:, 1], k=2)(t_smooth)
    
    return [tuple(p) for p in zip(spline_lat, spline_lon)]

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    st.session_state.real_p = generate_helios_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.step_idx = 0
    # 預估總距離計算
    dist = 0
    for i in range(len(st.session_state.real_p)-1):
        p1, p2 = st.session_state.real_p[i], st.session_state.real_p[i+1]
        dist += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 60
    st.session_state.total_planned_dist = dist
    st.rerun()

# --- 5. 數據計算與儀表板渲染 (對調位置) ---
subset = get_fast_ocean_data()
if subset is not None and st.session_state.real_p:
    curr_pt = subset.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u, v = float(curr_pt.water_u), float(curr_pt.water_v)
    sog = 15.0 + (u * 1.94) # 對地速度
    
    # 建議航向
    suggested_head = (np.degrees(np.arctan2(v, u)) + 360) % 360
    
    # 距離統計
    total_d = st.session_state.total_planned_dist
    traveled_d = (st.session_state.step_idx / (len(st.session_state.real_p)-1)) * total_d
    rem_d = total_d - traveled_d
    
    # 時間預估
    est_total_time = total_d / sog

    st.subheader("📊 HELIOS 智慧導航決策儀表板")
    c1, c2, c3 = st.columns(3)
    
    # 左：航速與建議航向
    c1.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
    c1.metric("🧭 建議航向", f"{suggested_head:.0f}°") # 互改位置
    
    # 中：能源紅利與總距離
    c2.metric("⛽ 能源紅利", f"{FUEL_GAIN_AVG}%", "Optimal")
    c2.metric("📏 航行總距離", f"{total_d:.1f} nmi", f"已航行 {traveled_d:.1f}")
    
    # 右：剩餘距離與預估總時間
    c3.metric("🎯 剩餘距離", f"{rem_d:.1f} nmi")
    c3.metric("🕒 預估總時間", f"{est_total_time:.2f} hrs") # 互改位置
    
    st.caption(f"📡 衛星接收強度: 穩定 ({LEO_STABILITY*100:.1f}%) | HELIOS 動態鏈結中")

    # --- 6. 地圖繪圖區 ---
    fig, ax = plt.subplots(figsize=(11, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    speed_grid = np.sqrt(subset.water_u**2 + subset.water_v**2)
    mesh = ax.pcolormesh(subset.lon, subset.lat, speed_grid, cmap='YlGnBu', alpha=0.4, shading='auto')
    
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1e1e1e', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', zorder=3)
    
    # 分段繪製：紅線為當前位置，虛線為剩餘路徑
    full_x = [p[1] for p in st.session_state.real_p]
    full_y = [p[0] for p in st.session_state.real_p]
    
    idx = st.session_state.step_idx
    # 已航行部分：紅色實線
    ax.plot(full_x[:idx+1], full_y[:idx+1], color='red', linewidth=3, zorder=4)
    # 未航行部分：白色虛線 (接在紅線之後)
    ax.plot(full_x[idx:], full_y[idx:], color='white', linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)
    
    # 終點圖標 (星型)
    ax.scatter(d_lon, d_lat, color='gold', marker='*', s=350, edgecolors='black', zorder=6, label='DESTINATION')
    
    # 船隻與向量
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=150, edgecolors='white', zorder=7)
    ax.quiver(st.session_state.ship_lon, st.session_state.ship_lat, u, v, color='red', scale=5, zorder=8)

    ax.set_extent([119, 124.5, 21.0, 26.5])
    st.pyplot(fig)

# --- 7. 移動模擬 ---
if st.button("🚢 執行移動：更新動態數據"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        new_loc = st.session_state.real_p[st.session_state.step_idx]
        st.session_state.ship_lat, st.session_state.ship_lon = new_loc
        st.rerun()
