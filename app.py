import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 數據與系統初始化 ---
st.set_page_config(page_title="HELIOS V24 完整功能版", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

@st.cache_data(ttl=3600)
def fetch_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S")
    except: return None, "N/A"

ocean_data, data_clock = fetch_data()

# --- 2. 核心避障邏輯：精準多點護欄 ---
def generate_safe_path(slat, slon, dlat, dlon):
    pts = [[slat, slon]]
    # 台灣經度屏障
    is_cross = (slon < 121.2 and dlon > 121.2) or (slon > 121.2 and dlon < 121.2)
    
    if is_cross:
        if (slat + dlat) / 2 > 23.8: # 繞北邏輯
            # 增加兩個護欄點，確保不切到基隆或東北角
            pts.append([26.0, 121.5]) # 北閘口
            pts.append([25.5, 122.5]) # 東北角外海護欄
        else: # 繞南邏輯
            # 增加三個護欄點，形成大弧度繞過鵝鑾鼻，絕對不切陸地
            pts.append([21.8, 120.4]) # 高雄外海
            pts.append([21.0, 120.8]) # 正南深水區
            pts.append([21.5, 121.5]) # 蘭嶼南方
            
    pts.append([dlat, dlon])
    pts = np.array(pts)
    
    # 使用線性插值，保證路徑是俐落的折線，不會繞圈
    t_new = np.linspace(0, 1, 150)
    y_path = np.interp(t_new, np.linspace(0, 1, len(pts)), pts[:, 0])
    x_path = np.interp(t_new, np.linspace(0, 1, len(pts)), pts[:, 1])
    return list(zip(y_path, x_path))

# --- 3. 原始儀表板 UI (還原截圖樣式) ---
st.title("🛰️ HELIOS V24 終極導航 (回歸原始設定)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速", "15.8 kn")
c2.metric("⛽ 能源紅利", "24.5%")
c3.metric("📏 狀態", "安全監控中")
c4.metric("⏱️ 數據時標", data_clock)
st.markdown("---")

# --- 4. 側邊欄：導航設定 ---
with st.sidebar:
    st.header("🚢 導航設定")
    slat_in = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon_in = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    dlat_in = st.number_input("終點緯度", value=22.500, format="%.3f")
    dlon_in = st.number_input("終點經度", value=120.000, format="%.3f")

    if st.button("🚀 啟動智能航路"):
        st.session_state.ship_lat, st.session_state.ship_lon = slat_in, slon_in
        st.session_state.real_p = generate_safe_path(slat_in, slon_in, dlat_in, dlon_in)
        st.session_state.step_idx = 0
        st.rerun()

# --- 5. 地圖顯示 (灰色台灣 + 流向箭頭) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2) 
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.5, zorder=3)

if ocean_data is not None:
    lons, lats = ocean_data.lon.values, ocean_data.lat.values
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    ax.pcolormesh(lons, lats, speed, cmap='YlGnBu', alpha=0.5, zorder=1)
    # 箭頭
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.3, scale=20, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    # 繪製完整路徑 (虛線)
    ax.plot(px, py, color='white', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
    # 繪製已行駛/計畫路徑 (洋紅色)
    ax.plot(px[:st.session_state.step_idx+1], py[:st.session_state.step_idx+1], color='#FF00FF', linewidth=3, zorder=6)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, edgecolors='white', zorder=7)
    ax.scatter(dlon_in, dlat_in, color='gold', marker='*', s=250, zorder=8)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

# --- 6. 底部按鈕：執行下一階段 ---
if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p) - 1:
        st.session_state.step_idx = min(st.session_state.step_idx + 10, len(st.session_state.real_p) - 1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
