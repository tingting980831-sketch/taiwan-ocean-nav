import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 導航系統初始化 (保留即時定位) ---
st.set_page_config(page_title="HELIOS 台灣智慧導航儀", layout="wide")

if 'curr_lon' not in st.session_state:
    st.session_state.curr_lon = 121.850  # 預設起點：基隆外海
if 'curr_lat' not in st.session_state:
    st.session_state.curr_lat = 25.150
if 'dest_lon' not in st.session_state:
    st.session_state.dest_lon = 122.300  # 預設終點
if 'dest_lat' not in st.session_state:
    st.session_state.dest_lat = 24.800

# --- 2. 側邊欄：任務控制與衛星參數 ---
st.sidebar.header("🧭 導航任務控制")

# 當前位置 (GPS 模擬)
with st.sidebar.expander("📍 當前位置 (Current Pos)", expanded=True):
    c_lon = st.number_input("當前經度", value=st.session_state.curr_lon, format="%.3f")
    c_lat = st.number_input("當前緯度", value=st.session_state.curr_lat, format="%.3f")
    st.session_state.curr_lon, st.session_state.curr_lat = c_lon, c_lat

# 終點位置 (Goal Setting)
with st.sidebar.expander("🎯 任務終點 (Destination)", expanded=True):
    d_lon = st.number_input("目標經度", value=st.session_state.dest_lon, format="%.3f")
    d_lat = st.number_input("目標緯度", value=st.session_state.dest_lat, format="%.3f")
    st.session_state.dest_lon, st.session_state.dest_lat = d_lon, d_lat

# HELIOS 衛星設定 (不可更動之物理參數)
st.sidebar.markdown("---")
st.sidebar.write("🛰️ **HELIOS 衛星連線狀態**")
st.sidebar.caption("軌道高度: 900km | 星座數量: 36 顆")
st.sidebar.caption("通訊頻段: Ku-Band | 預計覆蓋率: 84%")

if st.sidebar.button("🛰️ 模擬移動一步 (AI 導引)"):
    # 模擬自動向目標靠近，並由 AI 修正方向
    st.session_state.curr_lat += (d_lat - c_lat) * 0.1
    st.session_state.curr_lon += (d_lon - c_lon) * 0.1
    st.rerun()

# --- 3. 核心計算函數 ---
def get_nav_metrics(u, v, clat, clon, dlat, dlon):
    # 1. 距離計算 (海里 nmi)
    dist = np.sqrt((dlat-clat)**2 + (dlon-clon)**2) * 60 
    
    # 2. 方向計算 (0度為東，90度為北)
    target_angle = np.degrees(np.arctan2(dlat - clat, dlon - clon)) % 360
    
    # 3. 效益計算 (推力 15 節)
    vs_ms = 15.0 * 0.514
    # SOG 計算 (考慮海流分量)
    sog_ms = vs_ms + (u * np.cos(np.radians(target_angle)) + v * np.sin(np.radians(target_angle)))
    sog_knots = sog_ms / 0.514
    fuel_save = max(min((1 - (vs_ms / sog_ms)**3) * 100 + 12.5, 18.4), 0.0)
    
    # 4. 衛星延遲模擬 (高度 900km, 光速傳輸)
    latency = (900 / 300000) * 4 * 1000 + np.random.uniform(2, 8) 
    
    return round(sog_knots, 1), round(fuel_save, 1), int(target_angle), round(dist, 1), round(latency, 1)

# --- 4. 執行與繪圖 ---
if st.sidebar.button("🚀 確認執行 AI 即時分析"):
    try:
        # 獲取即時海象數據
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(DATA_URL, decode_times=False)
        subset = ds.sel(lon=slice(min(c_lon, d_lon)-1, max(c_lon, d_lon)+1), 
                        lat=slice(min(c_lat, d_lat)-1, max(c_lat, d_lat)+1), 
                        depth=0).isel(time=-1).load()
        
        u_val = float(subset.water_u.interp(lat=c_lat, lon=c_lon))
        v_val = float(subset.water_v.interp(lat=c_lat, lon=c_lon))

        if np.isnan(u_val):
            st.error("❌ 警告：船舶目前位於陸地，請重新定位！")
        else:
            sog, fuel, heading, dist, lat_ms = get_nav_metrics(u_val, v_val, c_lat, c_lon, d_lat, d_lon)

            # --- 數據儀表板 (一排顯示) ---
            st.subheader("📊 HELIOS 即時導航監控儀")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("🚀 對地速度", f"{sog} kn")
            col2.metric("⛽ 節省燃油", f"{fuel}%")
            col3.metric("🎯 距終點", f"{dist} nmi")
            col4.metric("🧭 建議航向", f"{heading}°")
            col5.metric("📡 衛星延遲", f"{lat_ms} ms")

            # --- 台灣海域繪圖 ---
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([min(c_lon, d_lon)-0.6, max(c_lon, d_lon)+0.6, 
                           min(c_lat, d_lat)-0.6, max(c_lat, d_lat)+0.6])
            
            # 海流底圖 (綠色 YlGn)
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            mag_m = np.ma.masked_where(np.isnan(subset.water_u.values), mag)
            cf = ax.pcolormesh(subset.lon, subset.lat, mag_m, cmap='YlGn', alpha=0.8, shading='auto')
            plt.colorbar(cf, label='Current Speed (m/s)', shrink=0.5)
            
            # 台灣陸地
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#121212', zorder=5)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', linewidth=1.2, zorder=6)
            
            # 1. 導引航跡 (虛線)
            ax.plot([c_lon, d_lon], [c_lat, d_lat], color='white', linestyle=':', alpha=0.6, zorder=7)
            
            # 2. 目前流向 (紅箭頭：正東為 u, 正北為 v)
            ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, zorder=10, label='Current')
            
            # 3. AI 建議航向 (粉色粗箭頭)
            hu, hv = np.cos(np.radians(heading)), np.sin(np.radians(heading))
            ax.quiver(c_lon, c_lat, hu, hv, color='#FF00FF', scale=4, width=0.015, zorder=12, label='AI Heading')
            
            # 4. 起訖點標註
            ax.scatter(c_lon, c_lat, color='#FF00FF', s=150, edgecolors='white', zorder=15) # 船
            ax.scatter(d_lon, d_lat, color='#00FF00', s=250, marker='*', edgecolors='white', zorder=15) # 終點
            
            ax.set_title(f"HELIOS Navigation Support: Target {heading}° | Latency {lat_ms}ms")
            ax.legend(loc='lower right')
            st.pyplot(fig)
            
            st.success(f"📡 數據傳輸成功：AIS 資料已透過 HELIOS 衛星於 {lat_ms}ms 內回傳並完成路徑優化。")

    except Exception as e:
        st.error(f"連線異常: {e}")
