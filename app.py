import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 系統初始化與狀態管理 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

# 初始化 Session State
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'history_path' not in st.session_state: st.session_state.history_path = [] # 儲存走過的紅線
if 'planned_path' not in st.session_state: st.session_state.planned_path = [] # 儲存預測的虛線

# --- 2. 側邊欄控制台 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")

# 起始點選擇
loc_mode = st.sidebar.radio("起始定位模式", ["立即定位 (GPS 模擬)", "手動輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    # 模擬一個固定的 GPS 起點
    start_lat, start_lon = 23.184, 121.739
    st.sidebar.success(f"📍 GPS 定位成功: {start_lat}, {start_lon}")
else:
    start_lat = st.sidebar.number_input("輸入起始緯度", value=23.184, format="%.3f")
    start_lon = st.sidebar.number_input("輸入起始經度", value=121.739, format="%.3f")

# 終點設定
dest_lat = st.sidebar.number_input("目標緯度", value=25.500, format="%.3f")
dest_lon = st.sidebar.number_input("目標經度", value=121.800, format="%.3f")

# --- 3. 路徑規劃與避障演算法 ---
def plan_full_route(s_lat, s_lon, d_lat, d_lon):
    """生成完整路徑並避開台灣陸地"""
    steps = 20
    lats = np.linspace(s_lat, d_lat, steps)
    lons = np.linspace(s_lon, d_lon, steps)
    path = []
    for lat, lon in zip(lats, lons):
        # 避障邏輯：如果是台灣陸地範圍，強制向東繞行到黑潮區
        if 120.0 < lon < 122.2 and 21.9 < lat < 25.3:
            lon = 122.6 
        path.append((lat, lon))
    return path

# 按下分析按鈕
if st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True):
    # 重設船隻位置到起點
    st.session_state.ship_lat = start_lat
    st.session_state.ship_lon = start_lon
    # 生成完整預測路徑（虛線）
    st.session_state.planned_path = plan_full_route(start_lat, start_lon, dest_lat, dest_lon)
    # 重設歷史路徑（紅線）
    st.session_state.history_path = [(start_lat, start_lon)]
    st.sidebar.balloons()

# --- 4. 核心數據獲取 (HYCOM) ---
@st.cache_data(ttl=3600)
def fetch_ocean_data():
    DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    return xr.open_dataset(DATA_URL, decode_times=False)

try:
    ds = fetch_ocean_data()
    # 抓取當前位置的流場
    curr_ds = ds.isel(time=-1, depth=0).interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u_val = float(curr_ds.water_u)
    v_val = float(curr_ds.water_v)
    
    # 儀表板計算
    sog = 15.0 + (u_val * 1.94) # 節
    fuel_efficiency = 25.4 if u_val > 0.4 else 12.0
except:
    u_val, v_val, sog, fuel_efficiency = 0.1, 0.1, 15.0, 0.0

# --- 5. 介面呈現：儀表板 ---
st.subheader("📊 HELIOS 即時導航監控儀表板")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 當前對地航速 (SOG)", f"{sog:.1f} kn")
c2.metric("⛽ 能源紅利增益", f"{fuel_efficiency}%")
c3.metric("📍 船隻位置", f"{st.session_state.ship_lon:.2f}E, {st.session_state.ship_lat:.2f}N")
c4.metric("📡 通訊延遲", "42 ms (LEO)")

# --- 6. 地圖繪製 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1a1a1a', zorder=1)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', linewidth=0.8, zorder=2)
ax.set_extent([118, 125, 20, 27]) # 聚焦台灣海域

# A. 繪製預估路徑 (藍色虛線 - 代表預測未來)
if st.session_state.planned_path:
    p_lats = [p[0] for p in st.session_state.planned_path]
    p_lons = [p[1] for p in st.session_state.planned_path]
    ax.plot(p_lons, p_lats, color='cyan', linestyle='--', linewidth=1.5, alpha=0.6, label='Predicted Route (HELIOS AI)')

# B. 繪製實際路徑 (紅色實線 - 代表已知真實路徑)
if st.session_state.history_path:
    h_lats = [p[0] for p in st.session_state.history_path]
    h_lons = [p[1] for p in st.session_state.history_path]
    ax.plot(h_lons, h_lats, color='red', linestyle='-', linewidth=2.5, label='Actual Verified Path', zorder=4)

# C. 繪製當前海流向量 (紅色實線箭頭)
ax.quiver(st.session_state.ship_lon, st.session_state.ship_lat, u_val, v_val, 
          color='red', scale=5, width=0.01, label='Real-time Current Vector', zorder=5)

# D. 船隻位置標記
ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='white', s=80, edgecolors='red', zorder=6)
ax.scatter(dest_lon, dest_lat, color='yellow', marker='*', s=200, label='Target', zorder=6)

ax.legend(loc='lower right', facecolor='#333333', labelcolor='white')
st.pyplot(fig)

# --- 7. 移動模擬控制 ---
if st.button("🚢 執行下一步移動 (模擬實測推進)"):
    if st.session_state.planned_path:
        # 尋找目前在預先規劃路徑中的下一個點
        # 這裡簡單模擬：把 planned_path 的第一個點移到 history_path
        if len(st.session_state.planned_path) > 1:
            next_step = st.session_state.planned_path.pop(0)
            st.session_state.ship_lat, st.session_state.ship_lon = next_step
            st.session_state.history_path.append(next_step)
            st.rerun()
        else:
            st.success("🏁 已抵達目標海域，任務完成。")
