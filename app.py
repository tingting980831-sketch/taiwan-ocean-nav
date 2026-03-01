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

# --- 2. 側邊欄：輸入控制 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
loc_mode = st.sidebar.radio("起始點選擇", ["立即定位 (GPS 模擬)", "自行輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    s_lat, s_lon = 23.184, 121.739
    st.sidebar.success(f"📍 GPS 已鎖定: {s_lat}, {s_lon}")
else:
    s_lat = st.sidebar.number_input("起始緯度", value=23.184, format="%.3f")
    s_lon = st.sidebar.number_input("起始經度", value=121.739, format="%.3f")

st.sidebar.markdown("---")
d_lat = st.sidebar.number_input("終點緯度", value=25.500, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=121.800, format="%.3f")

# --- 3. 路徑生成：穩定流場導引與避障 ---
def generate_helios_path(slat, slon, dlat, dlon):
    # 建立路徑控制點，避免橫切台灣
    points = [[slat, slon]]
    
    # 避障偵測：如果起點在東部但終點在西部，或者路徑太靠近陸地
    # 強制加入東岸流軸導航點 (122.2E 是黑潮流軸)
    if slon > 121.0 or dlon > 121.0:
        mid_lat = (slat + dlat) / 2
        points.append([mid_lat, 122.2])
    
    points.append([dlat, dlon])
    
    final_path = []
    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        steps = 40
        lats = np.linspace(p1[0], p2[0], steps)
        lons = np.linspace(p1[1], p2[1], steps)
        for la, lo in zip(lats, lons):
            final_path.append((la, lo))
    return final_path

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    st.session_state.real_p = generate_helios_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.step_idx = 0
    st.rerun()

# --- 4. 數據獲取 (解決底圖消失問題) ---
@st.cache_data(ttl=3600)
def get_ocean_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.5, 27.0), lon=slice(118.0, 125.0), depth=0).isel(time=-1).load()
        return subset
    except: return None

# --- 5. 渲染儀表板與衛星狀態 ---
data = get_ocean_data()

# 衛星狀態列 (獨立於圖表上方)
st.markdown("🛰️ **衛星接收強度：穩定 (98.2%)** | HELIOS 動態鏈結中")

if st.session_state.real_p:
    # 流速插值
    u, v = 0.6, 0.4
    if data is not None:
        try:
            curr_pt = data.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
            u, v = float(curr_pt.water_u), float(curr_pt.water_v)
        except: pass
    
    sog = 15.0 + (u * 1.94)
    head = (np.degrees(np.arctan2(v, u)) + 360) % 360
    
    # 距離計算
    total_pts = len(st.session_state.real_p)
    dist_total = total_pts * 1.2
    dist_rem = max(0.0, dist_total * (1 - st.session_state.step_idx / total_pts))
    est_time = dist_total / sog

    # 儀表板區域 (依照要求對調)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
        st.metric("🧭 建議航向", f"{head:.0f}°") # 左側下方
    with c2:
        st.metric("⛽ 能源紅利", "25.4%", "Optimal")
        st.metric("📏 航行總距離", f"{dist_total:.1f} nmi")
    with c3:
        st.metric("🎯 剩餘距離", f"{dist_rem:.1f} nmi")
        st.metric("🕒 預估總時間", f"{est_time:.2f} hrs") # 右側下方

    # --- 6. 地圖繪製 (修復配色與層級) ---
    fig, ax = plt.subplots(figsize=(11, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    
    if data is not None:
        speed = np.sqrt(data.water_u**2 + data.water_v**2)
        mesh = ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGnBu', alpha=0.4, shading='auto')
        plt.colorbar(mesh, ax=ax, label='Current Speed (m/s)', fraction=0.03, pad=0.04)

    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1e1e1e', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', zorder=3)
    
    # 路徑繪製
    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    idx = st.session_state.step_idx
    
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.8, zorder=4) # 全程規劃
    ax.plot(px[:idx+1], py[:idx+1], color='red', linewidth=3, zorder=5) # 實際航跡
    
    # 標記
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=350, edgecolors='black', zorder=7)
    
    # 向量箭頭
    ax.quiver(st.session_state.ship_lon, st.session_state.ship_lat, u, v, color='red', scale=5, zorder=8)

    ax.set_extent([119, 124.5, 21.0, 26.5])
    st.pyplot(fig)

if st.button("🚢 更新航行數據 (下一步)"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
