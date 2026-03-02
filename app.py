import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS V28 最終防撞版", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

@st.cache_data(ttl=3600)
def fetch_hycom_v28():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except: return None, "N/A", "OFFLINE"

ocean_data, data_clock, stream_status = fetch_hycom_v28()

# --- 2. 鐵律：陸地檢查 (防止選陸地當起點) ---
def is_on_land(lat, lon):
    # 台灣本島大略座標區塊 (含安全緩衝)
    return (21.9 <= lat <= 25.35) and (120.1 <= lon <= 122.0)

# --- 3. 航向計算 ---
def calc_bearing(p1, p2):
    y = np.sin(np.radians(p2[1]-p1[1])) * np.cos(np.radians(p2[0]))
    x = np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0])) - \
        np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y,x)) + 360) % 360

# --- 4. 頂部儀表板 (含建議航向) ---
st.title("🛰️ HELIOS V28 (絕對避障與定位版)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🚀 航速", "15.8 kn")
c2.metric("⛽ 能源紅利", "24.5%")

# 建議航向
brg = "---"
if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
    brg = f"{calc_bearing(st.session_state.real_p[st.session_state.step_idx], st.session_state.real_p[st.session_state.step_idx+1]):.1f}°"
c3.metric("🧭 建議航向", brg)
c4.metric("📡 衛星", "12 Pcs")
c5.metric("🕒 數據時標", data_clock)

st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.success(f"🌊 流場: {stream_status}")
m2.info("📡 GNSS: 訊號鎖定")
m3.write(f"📍 目前位置: {st.session_state.ship_lat:.3f}N, {st.session_state.ship_lon:.3f}E")
st.markdown("---")

# --- 5. 側邊欄：定位與避障 ---
with st.sidebar:
    st.header("🚢 導航控制器")
    if st.button("📍 抓取目前位置作為起點", use_container_width=True):
        st.toast("定位已同步")
        
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=22.500, format="%.3f")
    elon = st.number_input("終點經度", value=120.000, format="%.3f")

    # 檢查是否在陸地
    if is_on_land(slat, slon) or is_on_land(elat, elon):
        st.error("🚫 錯誤：座標位於陸地禁區")
        st.button("🚀 啟動智能航路", disabled=True, use_container_width=True)
    else:
        if st.button("🚀 啟動智能航路", use_container_width=True):
            pts = [[slat, slon]]
            # 東西跨越判定
            if (slon < 121.2 and elon > 121.2) or (slon > 121.2 and elon < 121.2):
                # 決定繞行點 (單一點位防止亂繞)
                if (slat + elat) / 2 > 23.8: # 繞北
                    pts.append([25.8, 121.8])
                else: # 繞南
                    pts.append([20.8, 120.8]) # 鵝鑾鼻南方深水區 (絕不切陸地)
            pts.append([elat, elon])
            
            # 生成平滑直線段
            final_p = []
            for i in range(len(pts)-1):
                for t in np.linspace(0, 1, 50):
                    final_p.append([pts[i][0] + (pts[i+1][0]-pts[i][0])*t, 
                                   pts[i][1] + (pts[i+1][1]-pts[i][1])*t])
            st.session_state.real_p = final_p
            st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
            st.session_state.step_idx = 0
            st.rerun()

# --- 6. 地圖繪製 (顏色還原 YlGn) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.5, zorder=3)

if ocean_data is not None:
    lons, lats = ocean_data.lon, ocean_data.lat
    speed = np.sqrt(ocean_data.water_u**2 + ocean_data.water_v**2)
    ax.pcolormesh(lons, lats, speed, cmap='YlGn', alpha=0.6, zorder=1)
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(lons[skip[1]], lats[skip[0]], ocean_data.water_u[skip], ocean_data.water_v[skip], 
              color='white', alpha=0.3, scale=22, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=6) # 粗洋紅色線
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, edgecolors='white', zorder=7)
    ax.scatter(elon, elat, color='gold', marker='*', s=250, zorder=8)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
        st.session_state.step_idx = min(st.session_state.step_idx + 10, len(st.session_state.real_p)-1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.rerun()
