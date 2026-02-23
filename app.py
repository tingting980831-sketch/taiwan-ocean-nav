import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 介面與 Session 初始化 ---
st.set_page_config(page_title="AI 智慧導航", layout="wide")

if 'sim_lon' not in st.session_state:
    st.session_state.sim_lon = 119.710 # 根據你圖中的座標
    st.session_state.sim_lat = 22.909

# --- 2. 側邊欄：控制中心 ---
st.sidebar.header("🕹️ 控制中心")

if st.sidebar.button("📍 模擬海上即時定位"):
    st.session_state.sim_lat = np.random.uniform(22.8, 25.2)
    st.session_state.sim_lon = np.random.uniform(119.8, 122.2)

c_lon = st.sidebar.number_input("當前經度 (AIS)", value=st.session_state.sim_lon, format="%.3f")
c_lat = st.sidebar.number_input("當前緯度 (AIS)", value=st.session_state.sim_lat, format="%.3f")

st.sidebar.markdown("---")
dest_lon = st.sidebar.number_input("目標經度", value=119.000, format="%.3f")
dest_lat = st.sidebar.number_input("目標緯度", value=24.800, format="%.3f")

SHIP_POWER = 15.0 

# --- 3. 計算函數 ---
def calculate_results(u, v, s_speed):
    vs_ms = s_speed * 0.514
    sog_ms = vs_ms + (u * 0.5 + v * 0.5)
    sog_knots = sog_ms / 0.514
    fuel_save = max(min((1 - (vs_ms/sog_ms)**3)*100 + 12.0, 18.4), 0.0)
    return round(sog_knots, 2), round(fuel_save, 1), 0.94

# --- 4. 主要顯示區 ---
if st.sidebar.button("🚀 執行導航分析"):
    try:
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(DATA_URL, decode_times=False)
        
        # 抓取正方形範圍 (度數差相同)
        subset = ds.sel(lon=slice(c_lon-0.5, c_lon+0.5), 
                        lat=slice(c_lat-0.5, c_lat+0.5), 
                        depth=0).isel(time=-1).load()
        
        u_val = subset.water_u.interp(lat=c_lat, lon=c_lon).values
        v_val = subset.water_v.interp(lat=c_lat, lon=c_lon).values

        if np.isnan(u_val):
            st.error("❌ 目前位置在陸地！")
        else:
            sog, fuel, comm = calculate_results(float(u_val), float(v_val), SHIP_POWER)

            # 數據儀表板
            st.subheader("📊 HELIOS 導航即時效益")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🚀 對地速度 (SOG)", f"{sog} kn")
            col2.metric("⛽ 燃油節省", f"{fuel}%")
            col3.metric("📡 通訊穩定度", f"{comm}")
            col4.metric("🧭 建議航向角", f"{round(np.degrees(np.arctan2(v_val, u_val)),1)}°")

            # --- 關鍵修正區：強制 1:1 正方形 ---
            # figsize 設定成 8x8 (正方形畫布)
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # 設定座標軸比例相等 (這是最重要的一行)
            ax.set_aspect('equal', adjustable='box') 
            
            # 設定顯示範圍 (經度差 = 緯度差)
            ax.set_extent([c_lon-0.4, c_lon+0.4, c_lat-0.4, c_lat+0.4])
            
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            land_mask = np.isnan(subset.water_u.values)
            mag_masked = np.ma.masked_where(land_mask, mag)
            
            cf = ax.pcolormesh(subset.lon, subset.lat, mag_masked, cmap='YlGn', shading='auto', alpha=0.9)
            plt.colorbar(cf, label='Current Speed (m/s)', shrink=0.8, pad=0.05)
            
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#333333', zorder=5)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', linewidth=1.5, zorder=6)
            
            ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, zorder=10)
            ax.scatter(c_lon, c_lat, color='#FF00FF', s=200, edgecolors='white', zorder=11)
            
            ax.set_title("Navigation Decision Support (True Square View)")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"連線異常: {e}")
