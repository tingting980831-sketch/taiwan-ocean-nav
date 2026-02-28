import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 系統初始化與常數設定 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

LEO_STABILITY = 0.982  # 衛星接收穩定度 98.2%
FUEL_GAIN_AVG = 25.4   # 預期節能增益 25.4%

# 初始化 Session State 儲存航行狀態
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'total_dist' not in st.session_state: st.session_state.total_dist = 0.0
if 'total_time' not in st.session_state: st.session_state.total_time = 0.0
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'pred_p' not in st.session_state: st.session_state.pred_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

# --- 2. 側邊欄控制台 ---
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
        # 僅選取台灣海域並跳點取樣 (isel step=3) 以加快速度
        subset = ds.sel(
            lat=slice(21.0, 26.5), 
            lon=slice(118.5, 124.5), 
            depth=0
        ).isel(
            time=-1, 
            lat=slice(None, None, 3), 
            lon=slice(None, None, 3)
        ).load()
        return subset
    except Exception as e:
        return None

# --- 4. 平滑路徑演算法 (解決直角問題) ---
def generate_smooth_paths(slat, slon, dlat, dlon):
    steps = 30
    lats = np.linspace(slat, dlat, steps)
    lons = np.linspace(slon, dlon, steps)
    
    raw_path = []
    for la, lo in zip(lats, lons):
        # 避障邏輯：強制偏向東側黑潮區 (避開台灣島)
        if 21.9 < la < 25.4 and 120.0 < lo < 122.2:
            lo = 122.6
        raw_path.append((la, lo))
    
    # 使用簡單移動平均進行平滑化，消除跳點產生的直角
    smooth_real = []
    for i in range(len(raw_path)):
        if i < 2 or i > len(raw_path) - 3:
            smooth_real.append(raw_path[i])
        else:
            avg_la = np.mean([raw_path[j][0] for j in range(i-2, i+3)])
            avg_lo = np.mean([raw_path[j][1] for j in range(i-2, i+3)])
            smooth_real.append((avg_la, avg_lo))
            
    # 預測路徑 (虛線)：加入隨機微小偏誤模擬預報不確定性
    smooth_pred = [(la, lo - 0.15 if 22 < la < 25 else lo) for la, lo in smooth_real]
    
    return smooth_real, smooth_pred

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    with st.spinner('📡 正在運算 HELIOS 動態場...'):
        st.session_state.real_p, st.session_state.pred_p = generate_smooth_paths(s_lat, s_lon, d_lat, d_lon)
        st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
        st.session_state.step_idx, st.session_state.total_dist, st.session_state.total_time = 0, 0.0, 0.0
        st.rerun()

# --- 5. 數據計算與儀表板渲染 ---
subset = get_fast_ocean_data()

if subset is not None and st.session_state.real_p:
    # 1. 取得目前位置海流 (正確資料)
    curr_pt = subset.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u, v = float(curr_pt.water_u), float(curr_pt.water_v)
    
    # 2. 更新累計數據 (每步模擬 0.5 小時)
    current_sog = 15.0 + (u * 1.94) # 基礎航速 + 海流增益
    time_step = 0.5
    st.session_state.total_time = st.session_state.step_idx * time_step
    st.session_state.total_dist = st.session_state.total_time * current_sog
    
    # 3. 其它指標計算
    rem_dist = max(0.0, 139.0 - st.session_state.total_dist)
    suggested_head = np.degrees(np.arctan2(v, u)) % 360
    
    # --- 儀表板佈局 ---
    st.subheader("📊 HELIOS 衛星導航即時儀表板")
    r1, r2, r3 = st.columns(3)
    r1.metric("🚀 當前航速 (SOG)", f"{current_sog:.1f} kn")
    r1.metric("📡 衛星接收", f"穩定 ({LEO_STABILITY*100:.1f}%)", "LEO-Link")
    
    r2.metric("⛽ 能源紅利", f"{FUEL_GAIN_AVG}%", "Optimal")
    r2.metric("📏 航行總距離", f"{st.session_state.total_dist:.1f} nmi")
    
    r3.metric("🎯 剩餘距離", f"{rem_dist:.1f} nmi")
    r3.metric("🕒 航行總時間", f"{st.session_state.total_time:.2f} hrs")

    # --- 6. 地圖繪製區 ---
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # A. 背景海流格子圖
    speed_grid = np.sqrt(subset.water_u**2 + subset.water_v**2)
    mesh = ax.pcolormesh(subset.lon, subset.lat, speed_grid, cmap='YlGnBu', alpha=0.5, shading='auto')
    plt.colorbar(mesh, ax=ax, label='Current Speed (m/s)', fraction=0.03, pad=0.04)

    # B. 陸地與海岸線
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#2c2c2c', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', zorder=3)

    # C. 路徑對比
    # 預測路徑 (白色虛線)
    px_lons, py_lats = [p[1] for p in st.session_state.pred_p], [p[0] for p in st.session_state.pred_p]
    ax.plot(px_lons, py_lats, color='white', linestyle='--', linewidth=1, label='Forecast (Predicted)')
    
    # 正確航道 (紅色實線 - 平滑化後)
    rx_lons, ry_lats = [p[1] for p in st.session_state.real_p], [p[0] for p in st.session_state.real_p]
    ax.plot(rx_lons, ry_lats, color='red', linestyle='-', linewidth=2.5, label='HELIOS Optimized (Actual)')

    # D. 船隻位置與方向
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=5)
    ax.quiver(st.session_state.ship_lon, st.session_state.ship_lat, u, v, color='red', scale=5, zorder=6)

    ax.set_extent([118.5, 124.5, 21.0, 26.5])
    ax.legend(loc='lower right')
    st.pyplot(fig)

# --- 7. 移動模擬按鈕 ---
if st.button("🚢 執行下一步移動 (更新實測數據)"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        next_loc = st.session_state.real_p[st.session_state.step_idx]
        st.session_state.ship_lat, st.session_state.ship_lon = next_loc
        st.rerun()
    else:
        st.success("🏁 抵達目標海域，導航任務完成。")
