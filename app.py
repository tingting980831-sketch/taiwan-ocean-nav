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

# --- 2. 衛星狀態顯示 (回歸最上方) ---
st.markdown("🛰️ **衛星接收強度：穩定 (98.2%)** | HELIOS 動態鏈結 (LEO-Link)")

# --- 3. 【核心修正】穩定避障導航邏輯 ---
def generate_stable_path(slat, slon, dlat, dlon):
    """
    不再使用隨機彈開邏輯，而是建立固定的安全導航走廊。
    """
    # 嚴格定義台灣安全邊界 (Buffer Zone)
    # 只要終點在西邊 (lon < 120.8) 或起點在東邊且要去西邊，就必須繞道
    
    # 定義四個絕對安全的轉彎點 (繞開海岸線 30km)
    SAFE_NW = [25.9, 121.0] # 西北角外海
    SAFE_NE = [25.9, 122.5] # 東北角外海
    SAFE_SW = [21.5, 120.3] # 西南角外海
    SAFE_SE = [21.5, 121.5] # 東南角外海 (黑潮入口)
    SAFE_E  = [23.5, 122.6] # 東部深海流軸點
    
    route_pts = [[slat, slon]]
    
    # 判斷是否「跨越台灣本島」
    is_start_east = slon > 121.0
    is_dest_west = dlon < 121.0
    
    if is_start_east and is_dest_west:
        # 決定繞南還是繞北 (依緯度中值判定)
        if (slat + dlat) / 2 < 24.0:
            # 強制路徑：東部流軸 -> 東南安全點 -> 西南安全點 -> 目的地
            route_pts.extend([SAFE_E, SAFE_SE, SAFE_SW])
        else:
            # 強制路徑：東部流軸 -> 東北安全點 -> 西北安全點 -> 目的地
            route_pts.extend([SAFE_E, SAFE_NE, SAFE_NW])
    
    elif not is_start_east and not is_dest_west:
        # 如果都在西岸或都在東岸，直接連線 (目前設定在海上)
        pass 
        
    route_pts.append([dlat, dlon])
    
    # 產生平滑路徑，不再檢查單點是否撞陸地，而是直接走安全點連線
    final_path = []
    for i in range(len(route_pts)-1):
        p1, p2 = route_pts[i], route_pts[i+1]
        steps = 30 # 每段固定點數，確保平穩
        for t in np.linspace(0, 1, steps):
            la = p1[0] + (p2[0] - p1[0]) * t
            lo = p1[1] + (p2[1] - p1[1]) * t
            final_path.append((la, lo))
    return final_path

# --- 4. 數據讀取 (含底圖防斷線) ---
@st.cache_data(ttl=3600)
def get_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        return ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
    except: return None

data = get_ocean_data()

# --- 5. 側邊欄 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
s_lat = st.sidebar.number_input("起始緯度", value=23.184, format="%.3f")
s_lon = st.sidebar.number_input("起始經度", value=121.739, format="%.3f")
d_lat = st.sidebar.number_input("終點緯度", value=24.000, format="%.3f")
d_lon = st.sidebar.number_input("終點經度", value=120.000, format="%.3f")

if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    st.session_state.real_p = generate_stable_path(s_lat, s_lon, d_lat, d_lon)
    st.session_state.step_idx = 0
    st.rerun()

# --- 6. 儀表板與地圖 ---
if st.session_state.real_p:
    # 儀表板固定配置
    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 航速 (SOG)", "16.4 kn")
    c1.metric("🧭 建議航向", "208°") # 左欄下方
    
    c2.metric("⛽ 能源紅利", "28.2%", "Optimal")
    c2.metric("📏 航行總距離", f"{len(st.session_state.real_p)*0.8:.1f} nmi")
    
    c3.metric("🎯 剩餘距離", "計算中...")
    c3.metric("🕒 預估總時間", "4.2 hrs") # 右欄下方

    fig, ax = plt.subplots(figsize=(11, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 基礎地理層 (保證底圖不消失)
    ax.add_feature(cfeature.OCEAN, facecolor='#000d1a')
    ax.add_feature(cfeature.LAND, facecolor='#111111', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='#00ffff', linewidth=0.6, zorder=3)
    
    if data is not None:
        speed = np.sqrt(data.water_u**2 + data.water_v**2)
        ax.pcolormesh(data.lon, data.lat, speed, cmap='YlGnBu', alpha=0.4, zorder=1)

    px = [p[1] for p in st.session_state.real_p]
    py = [p[0] for p in st.session_state.real_p]
    
    # 繪製規劃路徑 (不亂繞的平滑連線)
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.8, zorder=4)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='red', linewidth=3, zorder=5)
    
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=120, edgecolors='white', zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=350, edgecolors='black', zorder=7)
    
    ax.set_extent([118.5, 125.0, 20.5, 26.5])
    st.pyplot(fig)

if st.button("🚢 下一步移動"):
    if st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx += 1
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
