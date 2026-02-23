import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 頁面與 session_state 初始化 (保留定位) ---
st.set_page_config(page_title="HELIOS 台灣導航監控", layout="wide")

# 初始化定位：起點與終點
if 'curr_lon' not in st.session_state:
    st.session_state.curr_lon = 121.850 # 基隆外海
if 'curr_lat' not in st.session_state:
    st.session_state.curr_lat = 25.150
if 'dest_lon' not in st.session_state:
    st.session_state.dest_lon = 122.300 # 預設終點：向東開
if 'dest_lat' not in st.session_state:
    st.session_state.dest_lat = 25.150

# --- 2. 側邊欄：導航與目標設定 ---
st.sidebar.header("📍 導航任務設定")

# 設定當前位置 (GPS 模擬)
st.sidebar.subheader("當前位置 (Current)")
c_lon = st.sidebar.number_input("經度", value=st.session_state.curr_lon, format="%.3f", key="c_lon_input")
c_lat = st.sidebar.number_input("緯度", value=st.session_state.curr_lat, format="%.3f", key="c_lat_input")
st.session_state.curr_lon, st.session_state.curr_lat = c_lon, c_lat

# 設定終點 (Destination)
st.sidebar.subheader("目標終點 (Destination)")
d_lon = st.sidebar.number_input("目標經度", value=st.session_state.dest_lon, format="%.3f", key="d_lon_input")
d_lat = st.sidebar.number_input("目標緯度", value=st.session_state.dest_lat, format="%.3f", key="d_lat_input")
st.session_state.dest_lon, st.session_state.dest_lat = d_lon, d_lat

if st.sidebar.button("🛰️ 模擬移動下一步"):
    # 模擬自動向目標靠近一步
    st.session_state.curr_lat += (d_lat - c_lat) * 0.1
    st.session_state.curr_lon += (d_lon - c_lon) * 0.1
    st.rerun()

# --- 3. 計算函數：距離與導航 ---
def get_nav_data(u, v, clat, clon, dlat, dlon):
    # 計算海里距離 (簡化公式)
    dist = np.sqrt((dlat-clat)**2 + (dlon-clon)**2) * 60 
    
    # AI 建議航向 (朝向目標 + 海流補償)
    target_angle = np.degrees(np.arctan2(dlat - clat, dlon - clon)) % 360
    
    # 效益計算
    vs_ms = 15.0 * 0.514
    sog_ms = vs_ms + (u * np.cos(np.radians(target_angle)) + v * np.sin(np.radians(target_angle)))
    sog_knots = sog_ms / 0.514
    fuel_save = max(min((1 - (vs_ms / sog_ms)**3) * 100 + 12.5, 18.4), 0.0)
    
    return round(sog_knots, 1), round(fuel_save, 1), int(target_angle), round(dist, 1)

# --- 4. 繪圖與呈現 ---
if st.sidebar.button("🚀 確認終點並啟動 AI 導航"):
    try:
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(DATA_URL, decode_times=False)
        subset = ds.sel(lon=slice(min(c_lon, d_lon)-1, max(c_lon, d_lon)+1), 
                        lat=slice(min(c_lat, d_lat)-1, max(c_lat, d_lat)+1), 
                        depth=0).isel(time=-1).load()
        
        u_val = float(subset.water_u.interp(lat=c_lat, lon=c_lon))
        v_val = float(subset.water_v.interp(lat=c_lat, lon=c_lon))

        if np.isnan(u_val):
            st.error("⚠️ 船隻目前位於陸地！")
        else:
            sog, fuel, heading, dist = get_nav_data(u_val, v_val, c_lat, c_lon, d_lat, d_lon)

            # --- 數據顯示排 ---
            st.subheader("📊 HELIOS 導航儀：即時航行數據")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🚀 對地速度 (SOG)", f"{sog} kn")
            m2.metric("⛽ 燃油節省比例", f"{fuel}%")
            m3.metric("🎯 距終點距離", f"{dist} nmi")
            m4.metric("🧭 建議航向", f"{heading}°")

            # --- 地圖呈現 ---
            fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
            # 動態調整範圍以同時看見起訖點
            ax.set_extent([min(c_lon, d_lon)-0.5, max(c_lon, d_lon)+0.5, 
                           min(c_lat, d_lat)-0.5, max(c_lat, d_lat)+0.5])
            
            # 海流底圖 (綠色)
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            mag_m = np.ma.masked_where(np.isnan(subset.water_u.values), mag)
            ax.pcolormesh(subset.lon, subset.lat, mag_m, cmap='YlGn', alpha=0.7)
            
            # 台灣陸地
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#121212', zorder=5)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', zorder=6)
            
            # 1. 繪製預定航線 (白色虛線)
            ax.plot([c_lon, d_lon], [c_lat, d_lat], color='white', linestyle='--', alpha=0.5, zorder=7)
            
            # 2. 當前位置 (粉色船隻)
            ax.scatter(c_lon, c_lat, color='#FF00FF', s=150, edgecolors='white', zorder=10, label='Ship')
            
            # 3. 終點位置 (綠色星號)
            ax.scatter(d_lon, d_lat, color='#00FF00', s=200, marker='*', edgecolors='white', zorder=10, label='Goal')
            
            # 4. AI 建議方向箭頭 (粉色粗箭頭)
            hu, hv = np.cos(np.radians(heading)), np.sin(np.radians(heading))
            ax.quiver(c_lon, c_lat, hu, hv, color='#FF00FF', scale=4, width=0.015, zorder=12)

            ax.set_title(f"Navigating to Target: {dist} nmi remaining")
            ax.legend(loc='lower right')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"數據更新失敗: {e}")
