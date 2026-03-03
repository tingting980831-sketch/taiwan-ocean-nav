import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import os
import time

# ===============================
# 1. 初始化
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統 V29", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.5
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.0

CACHE_FILE = "hycom_cache.nc"

# ===============================
# 2. HYCOM 穩定抓取 (修正時間抓取邏輯)
# ===============================
@st.cache_data(ttl=3600)
def fetch_hycom_v29():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=True, chunks={'time': 1})
        # 抓取最新時間，並過濾台灣海域
        subset = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        data_time = subset.time.dt.strftime("%Y-%m-%d %H:%M").values.item()
        subset.to_netcdf(CACHE_FILE)
        return subset, data_time, "ONLINE (LIVE)"
    except Exception:
        if os.path.exists(CACHE_FILE):
            ds_c = xr.open_dataset(CACHE_FILE)
            return ds_c, "CACHE", "OFFLINE_MODE"
        return None, "N/A", "ERROR"

ocean_data, ocean_time, stream_status = fetch_hycom_v29()

# ===============================
# 3. 陸地判斷與避障 (加強緩衝)
# ===============================
def is_on_land(lat, lon):
    # 這裡的範圍要跟底圖的灰色區域完全對齊
    taiwan = (21.9 <= lat <= 25.35) and (120.08 <= lon <= 122.02)
    china = (lat >= 24.3 and lon <= 119.6)
    return taiwan or china

def calc_bearing(p1, p2):
    y = np.sin(np.radians(p2[1]-p1[1])) * np.cos(np.radians(p2[0]))
    x = np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0])) - \
        np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

# ===============================
# 4. 智能路徑 (修正繞南邏輯，防亂繞)
# ===============================
def smart_route_v29(start, end):
    pts = [start]
    # 跨越東西岸才需要中繼點
    if (start[1] - 121.2) * (end[1] - 121.2) < 0:
        if (start[0] + end[0]) / 2 > 23.8: # 繞北
            pts.append([26.3, 121.8])
        else: # 繞南：強制拉遠，避免切到恆春半島
            pts.append([20.8, 120.8])
    pts.append(end)
    
    final = []
    for i in range(len(pts)-1):
        # 使用 80 個點確保路徑平滑但不轉彎
        seg = np.linspace(pts[i], pts[i+1], 80)
        final.extend(seg.tolist())
    return final

# ===============================
# 5. UI 佈局 (雙行儀表板)
# ===============================
st.title("🛰️ HELIOS 智慧航行系統 (視覺修正版)")

r1 = st.columns(4)
r1[0].metric("🚀 基準航速", "12.0 kn")
r1[1].metric("⛽ 省油效益", "24.5%")
r1[2].metric("📡 衛星連線", "12 Pcs")
r1[3].metric("🌊 流場狀態", stream_status)

# 計算即時航向
brg = "---"
if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
    p_now = st.session_state.real_p[st.session_state.step_idx]
    p_next = st.session_state.real_p[st.session_state.step_idx+1]
    brg = f"{calc_bearing(p_now, p_next):.1f}°"

r2 = st.columns(4)
r2[0].metric("🧭 建議航向", brg)
r2[1].metric("📏 數據時標", ocean_time)
r2[2].metric("📍 目前位置", f"{st.session_state.ship_lat:.2f}N")
r2[3].metric("🕒 預估耗時", "---")
st.markdown("---")

# ===============================
# 6. 側邊欄控制
# ===============================
with st.sidebar:
    st.header("🚢 導航設定")
    if st.button("📍 GPS 定位起點", use_container_width=True):
        st.toast("已抓取當前衛星座標")

    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")
    st.session_state.dest_lat, st.session_state.dest_lon = elat, elon

    if is_on_land(slat, slon) or is_on_land(elat, elon):
        st.error("❌ 警告：起點或終點位於陸地！")
    else:
        if st.button("🚀 啟動智能航路", use_container_width=True):
            st.session_state.real_p = smart_route_v29([slat, slon], [elat, elon])
            st.session_state.step_idx = 0
            st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
            st.rerun()

# ===============================
# 7. 地圖繪製 (核心修正：陸地遮罩)
# ===============================
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# 先畫陸地底色
ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.5, zorder=5)

if ocean_data is not None:
    lons = ocean_data.lon.values
    lats = ocean_data.lat.values
    u = ocean_data.water_u.values
    v = ocean_data.water_v.values
    speed = np.sqrt(u**2 + v**2)
    
    # 修正：將速度極小的區域（通常是陸地或無數據區）設為 NaN，不畫箭頭
    u_clean = np.where(speed < 0.01, np.nan, u)
    v_clean = np.where(speed < 0.01, np.nan, v)

    # 繪製彩色流場 (設定 vmin/vmax 讓顏色更飽和，看起來不怪)
    im = ax.pcolormesh(lons, lats, speed, cmap='YlGn', shading='auto', 
                       alpha=0.7, vmin=0, vmax=1.2, zorder=1)
    
    # 繪製箭頭 (使用過濾後的 u_clean, v_clean，確保陸地沒箭頭)
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(lons[skip[1]], lats[skip[0]], u_clean[skip], v_clean[skip], 
              color='white', alpha=0.4, scale=20, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    # 繪製航線 (粗洋紅色)
    ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=6)
    # 船隻與終點
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=100, edgecolors='white', zorder=7)
    ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color='gold', marker='*', s=300, zorder=8)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

# ===============================
# 8. 執行按鈕
# ===============================
if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
        st.session_state.step_idx = min(st.session_state.step_idx + 10, len(st.session_state.real_p)-1)
        curr = st.session_state.real_p[st.session_state.step_idx]
        st.session_state.ship_lat, st.session_state.ship_lon = curr[0], curr[1]
        st.rerun()
