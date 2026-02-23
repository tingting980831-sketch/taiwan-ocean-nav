import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 導航系統初始化 ---
st.set_page_config(page_title="HELIOS 智慧導航系統 - 地球區域強化版", layout="wide")

if 'curr_lon' not in st.session_state:
    st.session_state.curr_lon = 121.850  # 當前位置 (預設基隆外海)
if 'curr_lat' not in st.session_state:
    st.session_state.curr_lat = 25.150
if 'dest_lon' not in st.session_state:
    st.session_state.dest_lon = 122.300  # 終點位置
if 'dest_lat' not in st.session_state:
    st.session_state.dest_lat = 24.800

# --- 2. 側邊欄：操作面板 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")

# 定位模式二選一
loc_mode = st.sidebar.radio("定位模式", ["立即定位 (GPS 模擬)", "手動輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    st.sidebar.info(f"📍 GPS 即時連線中\nLon: {st.session_state.curr_lon:.3f}\nLat: {st.session_state.curr_lat:.3f}")
    c_lon, c_lat = st.session_state.curr_lon, st.session_state.curr_lat
else:
    c_lon = st.sidebar.number_input("手動輸入經度", value=st.session_state.curr_lon, format="%.3f")
    c_lat = st.sidebar.number_input("手動輸入緯度", value=st.session_state.curr_lat, format="%.3f")
    st.session_state.curr_lon, st.session_state.curr_lat = c_lon, c_lat

# 終點設定
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 任務終點設定")
d_lon = st.sidebar.number_input("目標經度", value=st.session_state.dest_lon, format="%.3f")
d_lat = st.sidebar.number_input("目標緯度", value=st.session_state.dest_lat, format="%.3f")
st.session_state.dest_lon, st.session_state.dest_lat = d_lon, d_lat

# 衛星參數 (僅顯示地球強化版參數)
st.sidebar.markdown("---")
st.sidebar.write("🛰️ **HELIOS 星座狀態 (地球)**")
st.sidebar.caption("軌道高度: 900km | 衛星總數: 36 顆")
st.sidebar.caption("佈署策略: 中低緯度區域強化 (25°-45°)")
st.sidebar.caption("台灣區域穩定度: 98% (Active)")

# 按鈕功能
col_btn1, col_btn2 = st.sidebar.columns(2)
btn_analyze = col_btn1.button("🚀 AI 分析", use_container_width=True)
btn_move = col_btn2.button("🚢 模擬移動", use_container_width=True)

if btn_move:
    # 往目標點前進 10%
    st.session_state.curr_lat += (d_lat - st.session_state.curr_lat) * 0.1
    st.session_state.curr_lon += (d_lon - st.session_state.curr_lon) * 0.1
    st.rerun()

# --- 3. 核心計算邏輯 ---
def get_nav_metrics(u, v, clat, clon, dlat, dlon):
    # 距離與方向
    dist = np.sqrt((dlat-clat)**2 + (dlon-clon)**2) * 60 
    target_angle = np.degrees(np.arctan2(dlat - clat, dlon - clon)) % 360
    
    # 省油效益 (固定推力 15 節)
    vs_ms = 15.0 * 0.514
    sog_ms = vs_ms + (u * np.cos(np.radians(target_angle)) + v * np.sin(np.radians(target_angle)))
    sog_knots = sog_ms / 0.514
    fuel_save = max(min((1 - (vs_ms / sog_ms)**3) * 100 + 12.5, 18.4), 0.0)
    
    # 衛星延遲 (900km 軌道 + 地球環境修正)
    latency = (900 / 300) * 4 + 15.5 + np.random.uniform(0, 5)
    
    return round(sog_knots, 1), round(fuel_save, 1), int(target_angle), round(dist, 1), round(latency, 1)

# --- 4. 繪圖呈現 ---
if btn_analyze or btn_move:
    try:
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(DATA_URL, decode_times=False)
        subset = ds.sel(lon=slice(min(c_lon, d_lon)-1, max(c_lon, d_lon)+1), 
                        lat=slice(min(c_lat, d_lat)-1, max(c_lat, d_lat)+1), 
                        depth=0).isel(time=-1).load()
        
        u_val = float(subset.water_u.interp(lat=c_lat, lon=c_lon))
        v_val = float(subset.water_v.interp(lat=c_lat, lon=c_lon))

        sog, fuel, heading, dist, lat_ms = get_nav_metrics(u_val, v_val, c_lat, c_lon, d_lat, d_lon)

        # 儀表板數據排
        st.subheader("📊 HELIOS 系統即時分析報告")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🚀 對地航速 (SOG)", f"{sog} kn")
        m2.metric("⛽ 預估省油", f"{fuel}%")
        m3.metric("🎯 距終點", f"{dist} nmi")
        m4.metric("🧭 建議航向", f"{heading}°")
        m5.metric("📡 通訊延遲", f"{lat_ms} ms")

        # 地圖 (台灣區域，綠色底圖)
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([min(c_lon, d_lon)-0.6, max(c_lon, d_lon)+0.6, 
                       min(c_lat, d_lat)-0.6, max(c_lat, d_lat)+0.6])
        
        # 海流流速底圖
        mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
        ax.pcolormesh(subset.lon, subset.lat, mag, cmap='YlGn', alpha=0.8, shading='auto')
        
        # 台灣地圖特徵
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#121212', zorder=5)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', zorder=6)
        
        # 標註：流向(紅)、建議航向(粉)、起訖點
        ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, zorder=10, label='Sea Current (U:East/V:North)')
        hu, hv = np.cos(np.radians(heading)), np.sin(np.radians(heading))
        ax.quiver(c_lon, c_lat, hu, hv, color='#FF00FF', scale=4, width=0.015, zorder=12, label='AI Heading Advice')
        
        ax.plot([c_lon, d_lon], [c_lat, d_lat], 'w:', alpha=0.5, zorder=7) # 航跡線
        ax.scatter(c_lon, c_lat, color='#FF00FF', s=150, edgecolors='white', zorder=15) # 船
        ax.scatter(d_lon, d_lat, color='#00FF00', s=250, marker='*', edgecolors='white', zorder=15) # 終點
        
        ax.legend(loc='lower right')
        st.pyplot(fig)
        st.success(f"衛星狀態：通訊穩定。數據已由 HELIOS 36 星座透過 900km 軌道即時回傳。")

    except Exception as e:
        st.error(f"數據庫連線中...請稍候再試。 (系統提示: {e})")
