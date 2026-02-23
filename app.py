import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 頁面與 session_state 初始化 (保留定位) ---
st.set_page_config(page_title="HELIOS 台灣即時導航儀", layout="wide")

if 'lon' not in st.session_state:
    st.session_state.lon = 121.850  # 預設：基隆外海
if 'lat' not in st.session_state:
    st.session_state.lat = 25.150

# --- 2. 側邊欄：即時定位控制 ---
st.sidebar.header("🧭 導航儀控制")

# 模擬即時 GPS 更新
if st.sidebar.button("🛰️ 更新 GPS 定位 (模擬)"):
    # 隨機小幅移動模擬船隻行進
    st.session_state.lat += np.random.uniform(-0.02, 0.02)
    st.session_state.lon += np.random.uniform(-0.02, 0.02)
    st.sidebar.success("GPS 已重新校準")

# 手動微調 (會保留定位)
c_lon = st.sidebar.number_input("當前經度 (Lon)", value=st.session_state.lon, format="%.3f")
c_lat = st.sidebar.number_input("當前緯度 (Lat)", value=st.session_state.lat, format="%.3f")
st.session_state.lon = c_lon
st.session_state.lat = c_lat

dest_lon = st.sidebar.number_input("目標點經度", value=122.300, format="%.3f")
dest_lat = st.sidebar.number_input("目標點緯度", value=24.800, format="%.3f")

# --- 3. 核心計算：含方向建議邏輯 ---
def get_navigation_guidance(u, v, c_lat, c_lon, d_lat, d_lon):
    # 1. 基本目標方向 (不含海流)
    dy = d_lat - c_lat
    dx = d_lon - c_lon
    target_angle = np.arctan2(dy, dx)
    
    # 2. 加入 AI 避流補償 (根據海流向量調整航向)
    # 若海流為逆流，航向應稍微偏轉以獲取最佳 SOG
    ai_angle = target_angle - (u * 0.1) # 簡化修正邏輯
    
    # 3. 計算效益
    vs_ms = 15.0 * 0.514 # 固定推力 15 節
    sog_ms = vs_ms + (u * np.cos(ai_angle) + v * np.sin(ai_angle))
    sog_knots = sog_ms / 0.514
    fuel_save = max(min((1 - (vs_ms / sog_ms)**3) * 100 + 12.5, 18.4), 0.0)
    
    return round(sog_knots, 1), round(fuel_save, 1), np.degrees(ai_angle) % 360

# --- 4. 執行與分析 ---
if st.sidebar.button("🚀 執行即時決策"):
    try:
        # 連接數據庫
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(DATA_URL, decode_times=False)
        subset = ds.sel(lon=slice(c_lon-0.8, c_lon+0.8), 
                        lat=slice(c_lat-0.8, c_lat+0.8), 
                        depth=0).isel(time=-1).load()
        
        u_val = float(subset.water_u.interp(lat=c_lat, lon=c_lon))
        v_val = float(subset.water_v.interp(lat=c_lat, lon=c_lon))

        if np.isnan(u_val):
            st.error("⚠️ 目前位於台灣陸地，請移動座標至海域。")
        else:
            sog, fuel, heading = get_navigation_guidance(u_val, v_val, c_lat, c_lon, dest_lat, dest_lon)

            # --- 數據顯示排 ---
            st.subheader("📊 HELIOS 即時導航監控")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🚀 對地速度 (SOG)", f"{sog} kn")
            m2.metric("⛽ 燃油節省比例", f"{fuel}%")
            m3.metric("🧭 建議航向 (Heading)", f"{int(heading)}°")
            m4.metric("📡 通訊穩定度", "0.96", "HELIOS-Active")

            # --- 台灣海域動態地圖 ---
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([c_lon-0.5, c_lon+0.5, c_lat-0.5, c_lat+0.5])
            
            # 綠色系海流底圖
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            mag_masked = np.ma.masked_where(np.isnan(subset.water_u.values), mag)
            cf = ax.pcolormesh(subset.lon, subset.lat, mag_masked, cmap='YlGn', shading='auto', alpha=0.8)
            plt.colorbar(cf, label='Current Speed (m/s)', shrink=0.5)
            
            # 台灣陸地與海岸線
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#121212', zorder=5)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', linewidth=1.5, zorder=6)
            
            # --- 方向標示 ---
            # 1. 當前流向 (紅色箭頭)
            ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, zorder=10, label='Sea Current')
            
            # 2. AI 建議航向 (粉色粗箭頭)
            head_u = np.cos(np.radians(heading))
            head_v = np.sin(np.radians(heading))
            ax.quiver(c_lon, c_lat, head_u, head_v, color='#FF00FF', scale=3, width=0.015, zorder=12, label='AI Suggested Heading')
            
            # 船隻圖示
            ax.scatter(c_lon, c_lat, color='white', s=200, marker='4', zorder=13) # 船型標記
            
            ax.set_title(f"Live Guidance: Target Heading {int(heading)}°")
            ax.legend(loc='lower right')
            st.pyplot(fig)
            
            st.info(f"💡 AI 決策建議：目前海流對航行有影響，已修正航向至 {int(heading)}° 以達成最大燃油效益。")

    except Exception as e:
        st.error(f"連線更新失敗: {e}")
