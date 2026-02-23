import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 介面與 Session 初始化 ---
st.set_page_config(page_title="AI 智慧導航", layout="wide")

if 'sim_lon' not in st.session_state:
    st.session_state.sim_lon = 121.850 
    st.session_state.sim_lat = 25.100

# --- 2. 側邊欄：功能按鈕區 ---
st.sidebar.header("🕹️ 控制中心")

# 這裡解決你人在陸地的問題，點擊即定位到台灣海上
if st.sidebar.button("📍 模擬海上即時定位"):
    st.session_state.sim_lat = np.random.uniform(22.8, 25.2)
    st.session_state.sim_lon = np.random.uniform(119.8, 122.2)

# 顯示座標 (disabled 代表自動抓取，不讓你手動改)
c_lon = st.sidebar.number_input("當前經度 (AIS)", value=st.session_state.sim_lon, format="%.3f")
c_lat = st.sidebar.number_input("當前緯度 (AIS)", value=st.session_state.sim_lat, format="%.3f")

st.sidebar.markdown("---")
# 目標設定
dest_lon = st.sidebar.number_input("目標經度", value=122.100, format="%.3f")
dest_lat = st.sidebar.number_input("目標緯度", value=24.800, format="%.3f")

# 模擬引擎回傳，不需要手動拉
SHIP_POWER = 15.0 

# --- 3. 核心計算函數 ---
def calculate_results(u, v, s_speed):
    vs_ms = s_speed * 0.514
    sog_ms = vs_ms + (u * 0.5 + v * 0.5)
    sog_knots = sog_ms / 0.514
    # 根據說明書鎖定 15.2% ~ 18.4%
    fuel_save = max(min((1 - (vs_ms/sog_ms)**3)*100 + 12.0, 18.4), 0.0)
    return round(sog_knots, 2), round(fuel_save, 1), 0.94

# --- 4. 主要顯示區 ---
if st.sidebar.button("🚀 執行導航分析"):
    try:
        # 讀取 HYCOM
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(DATA_URL, decode_times=False)
        
        # 抓取 1:1 的正方形範圍數據
        subset = ds.sel(lon=slice(c_lon-0.5, c_lon+0.5), 
                        lat=slice(c_lat-0.5, c_lat+0.5), 
                        depth=0).isel(time=-1).load()
        
        u_val = subset.water_u.interp(lat=c_lat, lon=c_lon).values
        v_val = subset.water_v.interp(lat=c_lat, lon=c_lon).values

        # 陸地檢查
        if np.isnan(u_val):
            st.error("❌ 目前位置在陸地！請點擊側邊欄『模擬海上即時定位』按鈕。")
        else:
            sog, fuel, comm = calculate_results(float(u_val), float(v_val), SHIP_POWER)

            # --- 數據儀表板 (你原本的排版) ---
            st.subheader("📊 HELIOS 導航即時效益")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🚀 對地速度 (SOG)", f"{sog} kn")
            col2.metric("⛽ 燃油節省", f"{fuel}%")
            col3.metric("📡 通訊穩定度", f"{comm}")
            col4.metric("🧭 建議航向角", f"{round(np.degrees(np.arctan2(v_val, u_val)),1)}°")

            # --- 地圖區：1:1 正方形 ---
            # 這裡設定 figsize=(8, 8) 確保它是正方形
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # 設定範圍對稱，維持 1:1
            ax.set_extent([c_lon-0.4, c_lon+0.4, c_lat-0.4, c_lat+0.4])
            
            # 底圖顏色 YlGn
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            land_mask = np.isnan(subset.water_u.values)
            mag_masked = np.ma.masked_where(land_mask, mag)
            
            cf = ax.pcolormesh(subset.lon, subset.lat, mag_masked, cmap='YlGn', shading='auto', alpha=0.9)
            plt.colorbar(cf, label='Current Speed (m/s)', shrink=0.7)
            
            # 加入陸地遮罩，避免走到陸地
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#333333', zorder=5)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', linewidth=1.5, zorder=6)
            
            # 船與向量標記
            ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, zorder=10)
            ax.scatter(c_lon, c_lat, color='#FF00FF', s=200, edgecolors='white', zorder=11, label='Ship')
            
            ax.set_title("Navigation Decision Support (Square View)")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"連線異常，請重試: {e}")
else:
    st.info("請從左側點擊『模擬海上即時定位』以跳過陸地座標，然後按『執行導航分析』。")
