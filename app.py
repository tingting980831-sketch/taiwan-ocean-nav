import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# ===============================
# 1. 初始化 (還原你原始的變數名)
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.500
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.000
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

CACHE_FILE = "hycom_cache.nc"

# ===============================
# 2. 數據抓取 (物理遮罩邏輯：徹底解決陸地箭頭)
# ===============================
@st.cache_data(ttl=3600)
def fetch_hycom_v34():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=True, chunks={'time': 1})
        subset = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        
        u, v = subset.water_u.values, subset.water_v.values
        speed = np.sqrt(u**2 + v**2)
        # 關鍵遮罩：速度為0、太大或NaN的點(陸地)全部設為無效，quiver就不會畫出來
        mask = (speed == 0) | (speed > 5) | np.isnan(speed)
        u[mask] = np.nan
        v[mask] = np.nan
        
        subset.water_u.values, subset.water_v.values = u, v
        return subset, subset.time.dt.strftime("%Y-%m-%d %H:%M").values.item(), "ONLINE (LIVE)"
    except:
        # 模擬模式：也要手動抹除陸地箭頭
        lats, lons = np.linspace(20,27,80), np.linspace(118,126,80)
        lon2d, lat2d = np.meshgrid(lons, lats)
        u, v = 0.6 * np.ones_like(lon2d), 0.8 * np.ones_like(lon2d)
        # 手動定義陸地範圍遮罩
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if (21.8 <= lat <= 25.4 and 120.0 <= lon <= 122.1) or (lat >= 24.3 and lon <= 119.6):
                    u[i, j], v[i, j] = np.nan, np.nan
        
        ds_sim = xr.Dataset({"water_u": (("lat","lon"), u), "water_v": (("lat","lon"), v)},
                            coords={"lat": lats, "lon": lons})
        return ds_sim, "SIMULATED", "BACKUP_EMERGENCY"

ocean_data, ocean_time, stream_status = fetch_hycom_v34()

# ===============================
# 3. 原始儀表板 (完全不改動你的格式)
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")

r1 = st.columns(4)
r1[0].metric("🚀 基準航速", "12 kn")
r1[1].metric("⛽ 省油效益", "24.5%")
r1[2].metric("📡 衛星連線", "12 Pcs")
r1[3].metric("🌊 流場狀態", stream_status)

r2 = st.columns(4)
r2[0].metric("🧭 建議航向", "---")
r2[1].metric("📏 剩餘路程", "---")
r2[2].metric("🕒 預計抵達", "---")
r2[3].metric("🕒 數據時標", ocean_time)
st.markdown("---")

# ===============================
# 4. 側邊欄 (修復：現在可以輸入座標了)
# ===============================
with st.sidebar:
    st.header("🚢 導航控制中心")
    
    # 這裡就是讓你輸入座標的地方
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    if st.button("🚀 啟動智能航路", use_container_width=True):
        # 更新 session state
        st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
        st.session_state.dest_lat, st.session_state.dest_lon = elat, elon
        
        # 簡單生成直線路徑 (你可以之後再換回避障邏輯)
        st.session_state.real_p = [[slat, slon], [elat, elon]]
        st.session_state.step_idx = 0
        st.rerun()

# ===============================
# 5. 地圖繪製 (視覺修正：平滑流場 + 無陸地箭頭)
# ===============================
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#2b2b2b', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

if ocean_data is not None:
    lons, lats = ocean_data.lon.values, ocean_data.lat.values
    u, v = ocean_data.water_u.values, ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    
    # 用平滑色塊取代格子感
    ax.pcolormesh(lons, lats, speed, cmap='YlGn', alpha=0.7, shading='gouraud', zorder=1)
    
    # 繪製箭頭 (skip=5 讓畫面乾淨，width=0.002 變細)
    skip = (slice(None, None, 5), slice(None, None, 5))
    ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], 
              color='white', alpha=0.4, scale=30, width=0.002, zorder=4)

# 畫出航線與點
if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, zorder=7)
    ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color='gold', marker='*', s=200, zorder=8)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)
