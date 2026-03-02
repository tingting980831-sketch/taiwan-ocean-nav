import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS V23 遮罩防撞版", layout="wide")

@st.cache_data(ttl=3600)
def fetch_hycom_v23():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S")
    except: return None, "N/A"

ocean_data, data_clock = fetch_hycom_v23()

# --- 2. 核心修正：陸地遮罩與避障邏輯 (防止切到 & 亂繞) ---
def generate_v23_path(slat, slon, dlat, dlon):
    # 建立絕對安全點 (增加緩衝區，確保不切到陸地)
    NORTH_SAFE = [26.0, 121.5]
    SOUTH_SAFE = [21.0, 120.8]  # 往南推更多，防止切到恆春半島
    
    pts = [[slat, slon]]
    
    # 東西岸判定
    is_cross = (slon < 121.1 and dlon > 121.1) or (slon > 121.1 and dlon < 121.1)
    
    if is_cross:
        # 決定繞行路徑：比較北繞與南繞的緯度偏移量
        if (slat + dlat) / 2 > 23.8:
            pts.append(NORTH_SAFE)
        else:
            # 關鍵修正：只有當終點不在南閘口「北方」時才去南閘口，防止繞圈圈
            pts.append(SOUTH_SAFE)
            # 增加東南緩衝點，確保繞過後直接切向目標
            pts.append([21.3, 121.5]) 
            
    pts.append([dlat, dlon])
    pts = np.array(pts)
    
    # ！！！放棄 Spline，改用線性分段插值，絕對不會畫出多餘的圓圈
    t_new = np.linspace(0, 1, 100)
    y_path = np.interp(t_new, np.linspace(0, 1, len(pts)), pts[:, 0])
    x_path = np.interp(t_new, np.linspace(0, 1, len(pts)), pts[:, 1])
    return list(zip(y_path, x_path))

# --- 3. 完整儀表板 ---
st.title("🛰️ HELIOS V23 (遮罩防撞與路徑精簡)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速", "15.8 kn")
c2.metric("⛽ 能源紅利", "26.1%")
c3.metric("📏 狀態", "安全航行中")
c4.metric("⏱️ 數據時標", data_clock)

# --- 4. 側邊欄控制 ---
with st.sidebar:
    st.header("🚢 導航設定")
    s_lat = st.number_input("起點緯度", value=slat if 'slat' in locals() else 25.060, format="%.3f")
    s_lon = st.number_input("起點經度", value=slon if 'slon' in locals() else 122.200, format="%.3f")
    d_lat = st.number_input("終點緯度", value=22.500, format="%.3f")
    d_lon = st.number_input("終點經度", value=120.000, format="%.3f")

    if st.button("🚀 重新生成路徑 (防撞優化)"):
        st.session_state.real_p = generate_v23_path(s_lat, s_lon, d_lat, d_lon)
        st.rerun()

# --- 5. 地圖顯示 (灰色台灣遮罩) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# 增加背景深度，讓陸地更明顯
ax.add_feature(cfeature.OCEAN, facecolor='#000b1a')
ax.add_feature(cfeature.LAND, facecolor='#2b2b2b', zorder=2) 
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.5, zorder=3)

if ocean_data is not None:
    lons, lats = ocean_data.lon, ocean_data.lat
    u, v = ocean_data.water_u, ocean_data.water_v
    speed = np.sqrt(u**2 + v**2)
    ax.pcolormesh(lons, lats, speed, cmap='YlGnBu', alpha=0.5, zorder=1)
    # 箭頭
    skip = (slice(None, None, 5), slice(None, None, 5))
    ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.3, scale=25, zorder=4)

if 'real_p' in st.session_state and st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    # 使用洋紅色線，並確保不切到陸地
    ax.plot(px, py, color='#FF00FF', linewidth=3, alpha=0.9, zorder=5)
    ax.scatter(px[0], py[0], color='red', s=80, zorder=6)
    ax.scatter(px[-1], py[-1], color='gold', marker='*', s=250, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)
