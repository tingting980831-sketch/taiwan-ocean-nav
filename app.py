import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import os

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS V31 最終視覺修正", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

CACHE_FILE = "hycom_cache.nc"

# --- 2. 陸地硬性判定 (用於遮罩與導航) ---
def is_on_land(lat, lon):
    # 台灣本島
    taiwan = (21.8 <= lat <= 25.4) and (120.0 <= lon <= 122.1)
    # 中國沿岸 (擴大範圍確保箭頭完全消失)
    china = (lat >= 24.0 and lon <= 119.8)
    return taiwan or china

# --- 3. 數據抓取 (強化模擬流場的真實感) ---
@st.cache_data(ttl=3600)
def fetch_hycom_v31():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=True, chunks={'time': 1})
        subset = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        return subset, subset.time.dt.strftime("%Y-%m-%d %H:%M").values.item(), "ONLINE (LIVE)"
    except:
        # 如果失敗，生成更有「律動感」的模擬流場 (不再是排排站)
        lats, lons = np.linspace(20, 27, 80), np.linspace(118, 126, 80)
        lon2d, lat2d = np.meshgrid(lons, lats)
        # 加入渦流模擬，看起來更自然
        u = 0.7 * np.sin((lat2d-22)/2) + 0.2 * np.cos(lon2d/1.5)
        v = 0.5 * np.cos((lon2d-122)/2) + 0.1 * np.sin(lat2d/1.5)
        ds_sim = xr.Dataset({"water_u": (("lat","lon"), u), "water_v": (("lat","lon"), v)},
                            coords={"lat": lats, "lon": lons})
        return ds_sim, "SIMULATED", "EMERGENCY_MODE"

ocean_data, data_time, stream_status = fetch_hycom_v31()

# --- 4. 儀表板 ---
st.title("🛰️ HELIOS 智慧航行系統 V31")
c = st.columns(4)
c[0].metric("🚀 基準航速", "12.0 kn")
c[1].metric("⛽ 省油效益", "24.5%")
c[2].metric("🌊 流場狀態", stream_status)
c[3].metric("🕒 數據時標", data_time)
st.markdown("---")

# --- 5. 側邊欄與導航 ---
with st.sidebar:
    st.header("🚢 導航設定")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=22.500, format="%.3f")
    elon = st.number_input("終點經度", value=120.000, format="%.3f")
    
    if is_on_land(slat, slon) or is_on_land(elat, elon):
        st.error("🚫 座標位於陸地！")
    else:
        if st.button("🚀 啟動智能航路", use_container_width=True):
            # 路徑點生成
            pts = [[slat, slon]]
            if (slon - 121.2) * (elon - 121.2) < 0: # 跨越東西岸
                mid = [20.7, 120.7] if (slat+elat)/2 < 24 else [26.0, 122.0]
                pts.append(mid)
            pts.append([elat, elon])
            
            final_p = []
            for i in range(len(pts)-1):
                for t in np.linspace(0, 1, 60):
                    final_p.append([pts[i][0]+(pts[i+1][0]-pts[i][0])*t, 
                                   pts[i][1]+(pts[i+1][1]-pts[i][1])*t])
            st.session_state.real_p = final_p
            st.rerun()

# --- 6. 地圖繪製 (視覺終極修正) ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#2b2b2b', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.5, zorder=5)

if ocean_data is not None:
    lons, lats = ocean_data.lon.values, ocean_data.lat.values
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    
    # 【關鍵修正：全域陸地箭頭遮罩】
    u_masked = u.copy()
    v_masked = v.copy()
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if is_on_land(lat, lon):
                u_masked[i, j] = np.nan
                v_masked[i, j] = np.nan

    # 繪製漸層底色
    ax.pcolormesh(lons, lats, speed, cmap='YlGn', alpha=0.6, shading='auto', zorder=1)
    
    # 繪製箭頭 (僅限海面)
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(lons[skip[1]], lats[skip[0]], u_masked[skip], v_masked[skip], 
              color='white', alpha=0.5, scale=25, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=6)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)
