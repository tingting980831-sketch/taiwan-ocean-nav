import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ===============================
# 1. 衛星與系統設定 (400km, 3面4顆)
# ===============================
SAT_CONFIG = {"total_sats": 12, "altitude_km": 400}

st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧導航控制台")

# ===============================
# 2. 側邊欄：輸入起點與終點
# ===============================
with st.sidebar:
    st.header("📍 航點座標輸入")
    st.info("請輸入經緯度，系統將自動定位至最近海域。")
    
    # 起點輸入
    st.subheader("🏁 出發點")
    s_lon = st.number_input("起點經度 (119-123)", value=120.30, format="%.2f")
    s_lat = st.number_input("起點緯度 (20-26)", value=22.60, format="%.2f")
    
    # 終點輸入
    st.subheader("🏁 目的地")
    e_lon = st.number_input("終點經度 (119-123)", value=122.00, format="%.2f")
    e_lat = st.number_input("終點緯度 (20-26)", value=24.50, format="%.2f")
    
    run_nav = st.button("🚀 啟動衛星導航計算", use_container_width=True)

# ===============================
# 3. 讀取 HYCOM 資料 (你的原始邏輯)
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        
        lat_slice, lon_slice = slice(20, 26), slice(119, 123)
        u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        
        return u_data['lon'].values, u_data['lat'].values, u_data.values, v_data.values, latest_time
    except Exception as e:
        st.error(f"資料讀取失敗: {e}")
        return None, None, None, None, None

lons, lats, u, v, obs_time = load_hycom_data()

# ===============================
# 4. 繪圖與結果顯示
# ===============================
if lons is not None:
    # 計算流速強度
    speed = np.sqrt(u**2 + v**2)

    # 建立 Matplotlib 圖表
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([119, 123, 20, 26], crs=ccrs.PlateCarree())

    # 視覺化設定 (灰色陸地)
    ax.add_feature(cfeature.LAND, facecolor='gray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', zorder=3)
    
    # 背景海流
    im = ax.pcolormesh(lons, lats, speed, cmap='YlGnBu', alpha=0.7, shading='auto', zorder=1)
    plt.colorbar(im, label='流速 (m/s)', shrink=0.6)

    # 繪製海流箭頭
    ax.quiver(lons[::2], lats[::2], u[::2, ::2], v[::2, ::2], color='black', scale=10, zorder=4)

    # 在地圖上標註「你輸入的」座標
    ax.scatter(s_lon, s_lat, color='lime', s=150, marker='o', label='起點', edgecolors='black', zorder=10)
    ax.scatter(e_lon, e_lat, color='yellow', s=250, marker='*', label='終點', edgecolors='black', zorder=10)

    # 如果按下按鈕，畫一條直線模擬導航 (之後可以換成 A* 曲線)
    if run_nav:
        ax.plot([s_lon, e_lon], [s_lat, e_lat], color='red', linestyle='--', linewidth=2, label='預計航線', zorder=5)
        st.success(f"✅ 已鎖定目標航線：從 ({s_lat}, {s_lon}) 前往 ({e_lat}, {e_lon})")

    ax.legend(loc='lower right')
    plt.title(f"HELIOS V6 即時監控 - {obs_time}")

    # --- 顯示圖片 ---
    st.pyplot(fig)

    # 數據儀表板
    col1, col2, col3 = st.columns(3)
    col1.metric("🛰️ 衛星高度", f"{SAT_CONFIG['altitude_km']} km")
    col2.metric("📏 直線距離", f"{np.sqrt((s_lat-e_lat)**2 + (s_lon-e_lon)**2)*111:.1f} km")
    col3.metric("📅 觀測時間", obs_time.strftime("%m/%d %H:%M"))
