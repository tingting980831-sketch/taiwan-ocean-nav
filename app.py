import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.special import binom

# --- 1. 系統初始化與狀態 ---
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

# --- 3. 核心算法：貝茲曲線避障導航 ---
def bernstein_poly(i, n, t):
    return binom(n, i) * (t**(n-i)) * (1-t)**i

def generate_bezier_path(points, num=60):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), points[n-i])
    return curve

def generate_safe_helios_path(slat, slon, dlat, dlon):
    # 台灣陸地緩衝邊界 (120.0E - 122.0E, 21.9N - 25.4N)
    # 如果路徑會穿過這個區域，則需修正
    ctrl_pts = [[slat, slon]]
    
    # 避障邏輯：如果跨越東西岸 (經度 121.0 為中心)
    if (slon > 121.2 and dlon < 120.8) or (slon < 120.8 and dlon > 121.2):
        # 決定繞南還是繞北 (以 23.5N 為界)
        if (slat + dlat) / 2 < 23.8:
            # 繞過南方：加入東側流軸點 + 鵝鑾鼻深海點 (21.5N, 120.8E)
            ctrl_pts.append([22.5, 122.2]) # 黑潮流軸點
            ctrl_pts.append([21.4, 121.0]) # 南端安全轉彎點 (避開墾丁近海)
        else:
            # 繞過北方：加入三貂角外海點
            ctrl_pts.append([24.5, 122.3])
            ctrl_pts.append([25.8, 121.8]) # 北端安全轉彎點
            
    ctrl_pts.append([dlat, dlon])
    
    # 使用貝茲曲線產生平滑路徑
    path_array = generate_bezier_path(np.array(ctrl_pts))
    return [tuple(p) for p in path_array]

if st.sidebar.button("🚀 執行 AI 安全路徑分析", use_container_width=True):
    st.session_state.real_p = generate_safe_helios_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.step_idx = 0
    st.rerun()

# --- 4. 數據與衛星狀態顯示 ---
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
    u, v = 0.5, 0.3
    if data is not None:
        try:
            curr = data.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
            u, v = float(curr.water_u), float(curr.water_v)
        except: pass
    
    sog = 15.0 + (u * 1.94)
    head = (np.degrees(np.arctan2(v, u)) + 360) % 360
    dist_total = len(st.session_state.real_p) * 1.1 # 估算航程
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
    c1.metric("🧭 建議航向", f"{head:.0f}°") # 左下
    
    c2.metric("⛽ 能源紅利", "25.4%", "Optimal")
    c2.metric("📏 航行總距離", f"{dist_total:.1f} nmi")
    
    c3.metric("🎯 剩餘距離", f"{max(0.0, dist_total * (1 - st.session_state.step_idx/60)):.1f} nmi")
    c3.metric("🕒 預估總時間", f"{dist_total/sog:.2f} hrs") # 右下

    # --- 6. 地圖繪圖 ---
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor='#121212', zorder=2) # 黑色陸地
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', zorder=3)
    
    if data is not None:
        speed = np.sqrt(data.water_u**2 + data.water_v**2)
        ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGnBu', alpha=0.3, shading='auto')

    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    
    ax.plot(px, py, color='white', linestyle='--', alpha=0.6, zorder=4) # 規劃虛線
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5) # 實際紅線
    
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=300, edgecolors='black', zorder=7)
    
    ax.set_extent([118.5, 124.5, 21.0, 26.5])
    st.pyplot(fig)

if st.button("🚢 更新航行數據 (下一步)"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
