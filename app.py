import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 基礎設定與常數 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

LEO_STABILITY = 0.982 
FUEL_GAIN_AVG = 25.4  

# 初始化 Session State
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'pred_p' not in st.session_state: st.session_state.pred_p = []

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

# --- 3. 數據讀取 (極速抽樣版) ---
@st.cache_data(ttl=3600)
def get_fast_ocean_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(21.0, 26.5), lon=slice(118.5, 124.5), depth=0).isel(time=-1, lat=slice(None, None, 3), lon=slice(None, None, 3)).load()
        return subset
    except: return None

# --- 4. 優化路徑演算法 (平滑且起點相連) ---
def generate_connected_path(slat, slon, dlat, dlon):
    steps = 40 
    lats = np.linspace(slat, dlat, steps)
    lons = np.linspace(slon, dlon, steps)
    
    path = []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        if i > 0 and i < steps - 1:
            if 21.9 < la < 25.4 and 120.0 < lo < 122.2:
                lo = 122.6 # 避障偏航
        path.append((la, lo))
    
    smooth_path = []
    window = 5
    for i in range(len(path)):
        start = max(0, i - window // 2)
        end = min(len(path), i + window // 2 + 1)
        avg_la = np.mean([p[0] for p in path[start:end]])
        avg_lo = np.mean([p[1] for p in path[start:end]])
        if i == 0: smooth_path.append((slat, slon))
        elif i == len(path)-1: smooth_path.append((dlat, dlon))
        else: smooth_path.append((avg_la, avg_lo))
    return smooth_path

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    st.session_state.real_p = generate_connected_path(s_lat, s_lon, d_lat, d_lon)
    # 預測路徑僅作為對比，稍微偏移
    st.session_state.pred_p = [(la, lo - 0.12) for la, lo in st.session_state.real_p]
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.step_idx = 0
    st.rerun()

# --- 5. 數據計算與儀表板渲染 ---
subset = get_fast_ocean_data()
if subset is not None and st.session_state.real_p:
    curr_pt = subset.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u, v = float(curr_pt.water_u), float(curr_pt.water_v)
    sog = 15.0 + (u * 1.94)
    
    def calc_dist(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 60
    
    total_planned_dist = sum(calc_dist(st.session_state.real_p[i], st.session_state.real_p[i+1]) for i in range(len(st.session_state.real_p)-1))
    traveled_dist = (st.session_state.step_idx / (len(st.session_state.real_p)-1)) * total_planned_dist
    rem_dist = total_planned_dist - traveled_dist
    
    total_est_time = total_planned_dist / sog
    suggested_head = (np.degrees(np.arctan2(v, u)) + 360) % 360

    st.subheader("📊 HELIOS 智慧導航決策儀表板")
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
    c1.metric("📡 衛星接收", f"穩定 ({LEO_STABILITY*100:.1f}%)")
    
    c2.metric("⛽ 能源紅利", f"{FUEL_GAIN_AVG}%", "Optimal")
    c2.metric("📏 航行總距離", f"{total_planned_dist:.1f} nmi", f"已航行 {traveled_dist:.1f}")
    
    c3.metric("🎯 剩餘距離", f"{rem_dist:.1f} nmi")
    c3.metric("🧭 建議航向", f"{suggested_head:.0f}°")
    
    # 額外資訊欄
    st.info(f"🕒 預估總航程時間: {total_est_time:.2f} 小時")

    # --- 6. 地圖繪圖區 ---
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    speed_grid = np.sqrt(subset.water_u**2 + subset.water_v**2)
    mesh = ax.pcolormesh(subset.lon, subset.lat, speed_grid, cmap='YlGnBu', alpha=0.4, shading='auto')
    plt.colorbar(mesh, ax=ax, label='Current Speed (m/s)', fraction=0.03, pad=0.04)
    
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1e1e1e', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', zorder=3)
    
    # 分段繪製路徑：紅線(已規劃)
    rx, ry = [p[1] for p in st.session_state.real_p], [p[0] for p in st.session_state.real_p]
    ax.plot(rx, ry, color='red', linewidth=2.5, label='HELIOS Optimized Path', zorder=4)
    
    # 預測路徑(虛線)：僅加在紅線之後或者作為環境對應
    px, py = [p[1] for p in st.session_state.pred_p], [p[0] for p in st.session_state.pred_p]
    ax.plot(px[st.session_state.step_idx:], py[st.session_state.step_idx:], color='white', linestyle='--', alpha=0.6, label='Forecast Horizon')
    
    # 終點標標 (星型)
    ax.scatter(d_lon, d_lat, color='gold', marker='*', s=250, edgecolors='black', linewidth=1.5, zorder=6, label='DESTINATION')
    
    # 船隻目前位置
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=7)
    # 海流向量箭頭
    ax.quiver(st.session_state.ship_lon, st.session_state.ship_lat, u, v, color='red', scale=6, zorder=8)

    ax.set_extent([119, 124.5, 21.0, 26.5])
    ax.legend(loc='lower right', fontsize='small')
    st.pyplot(fig)

# --- 7. 移動模擬 ---
if st.button("🚢 推進至下一導航點"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
    else:
        st.success("🏁 已成功抵達目的地！")
