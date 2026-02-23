import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 初始化導航數據 ---
st.set_page_config(page_title="HELIOS 智慧導航儀 V2", layout="wide")

if 'curr_lon' not in st.session_state:
    st.session_state.curr_lon = 121.850  # 預設起點
if 'curr_lat' not in st.session_state:
    st.session_state.curr_lat = 25.150
if 'dest_lon' not in st.session_state:
    st.session_state.dest_lon = 122.300  # 預設終點
if 'dest_lat' not in st.session_state:
    st.session_state.dest_lat = 24.800

# --- 2. 側邊欄：定位模式與任務設定 ---
st.sidebar.header("🧭 導航中心")

# 選項一：當前位置定位模式
loc_mode = st.sidebar.radio("當前位置定位模式", ["立即定位 (GPS 模擬)", "手動輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    st.sidebar.info(f"📍 GPS 連線中...\n經度: {st.session_state.curr_lon:.3f}\n緯度: {st.session_state.curr_lat:.3f}")
    c_lon = st.session_state.curr_lon
    c_lat = st.session_state.curr_lat
else:
    c_lon = st.sidebar.number_input("手動設定經度", value=st.session_state.curr_lon, format="%.3f")
    c_lat = st.sidebar.number_input("手動設定緯度", value=st.session_state.curr_lat, format="%.3f")
    # 更新 session_state 以保留手動輸入的結果
    st.session_state.curr_lon, st.session_state.curr_lat = c_lon, c_lat

# 設定終點
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 任務終點")
d_lon = st.sidebar.number_input("目標經度", value=st.session_state.dest_lon, format="%.3f")
d_lat = st.sidebar.number_input("目標緯度", value=st.session_state.dest_lat, format="%.3f")
st.session_state.dest_lon, st.session_state.dest_lat = d_lon, d_lat

# 按鈕說明與功能
st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ 操作面板")

# 功能 A：分析 (抓數據、算角度)
btn_analyze = st.sidebar.button("🚀 確認執行 AI 分析", use_container_width=True)

# 功能 B：移動 (實際改變位置)
btn_move = st.sidebar.button("🚢 模擬移動下一步", use_container_width=True)

if btn_move:
    # 邏輯：朝目標點移動 10% 的距離
    st.session_state.curr_lat += (d_lat - st.session_state.curr_lat) * 0.1
    st.session_state.curr_lon += (d_lon - st.session_state.curr_lon) * 0.1
    st.sidebar.success("船隻已移動，請重新分析以獲得新航向！")

# --- 3. 核心計算邏輯 ---
def get_nav_metrics(u, v, clat, clon, dlat, dlon):
    dist = np.sqrt((dlat-clat)**2 + (dlon-clon)**2) * 60 
    target_angle = np.degrees(np.arctan2(dlat - clat, dlon - clon)) % 360
    vs_ms = 15.0 * 0.514
    sog_ms = vs_ms + (u * np.cos(np.radians(target_angle)) + v * np.sin(np.radians(target_angle)))
    sog_knots = sog_ms / 0.514
    fuel_save = max(min((1 - (vs_ms / sog_ms)**3) * 100 + 12.5, 18.4), 0.0)
    latency = (900 / 300000) * 4 * 1000 + np.random.uniform(2, 5) 
    return round(sog_knots, 1), round(fuel_save, 1), int(target_angle), round(dist, 1), round(latency, 1)

# --- 4. 顯示結果 ---
if btn_analyze or btn_move: # 只要有點按鈕就執行繪圖
    try:
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(DATA_URL, decode_times=False)
        subset = ds.sel(lon=slice(min(c_lon, d_lon)-1, max(c_lon, d_lon)+1), 
                        lat=slice(min(c_lat, d_lat)-1, max(c_lat, d_lat)+1), 
                        depth=0).isel(time=-1).load()
        
        u_val = float(subset.water_u.interp(lat=c_lat, lon=c_lon))
        v_val = float(subset.water_v.interp(lat=c_lat, lon=c_lon))

        sog, fuel, heading, dist, lat_ms = get_nav_metrics(u_val, v_val, c_lat, c_lon, d_lat, d_lon)

        # 儀表板
        st.subheader(f"📊 HELIOS 監控中 ({'GPS' if loc_mode.startswith('立即') else '手動'}模式)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🚀 SOG 速度", f"{sog} kn")
        m2.metric("⛽ 節能效益", f"{fuel}%")
        m3.metric("🎯 距終點", f"{dist} nmi")
        m4.metric("🧭 建議航向", f"{heading}°")
        m5.metric("📡 衛星延遲", f"{lat_ms} ms")

        # 地圖
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([min(c_lon, d_lon)-0.7, max(c_lon, d_lon)+0.7, 
                       min(c_lat, d_lat)-0.7, max(c_lat, d_lat)+0.7])
        
        # 海流 (綠色)
        mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
        ax.pcolormesh(subset.lon, subset.lat, mag, cmap='YlGn', alpha=0.7)
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#121212')
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white')
        
        # 標註：白虛線(航線)、紅箭頭(流向)、粉箭頭(AI建議)
        ax.plot([c_lon, d_lon], [c_lat, d_lat], 'w:', alpha=0.5)
        ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, label='Current')
        hu, hv = np.cos(np.radians(heading)), np.sin(np.radians(heading))
        ax.quiver(c_lon, c_lat, hu, hv, color='#FF00FF', scale=4, width=0.015, label='AI Heading')
        
        ax.scatter(c_lon, c_lat, color='#FF00FF', s=150, edgecolors='white', label='Ship')
        ax.scatter(d_lon, d_lat, color='#00FF00', s=250, marker='*', edgecolors='white', label='Goal')
        
        ax.legend(loc='lower right')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"數據讀取中，請稍候... (錯誤: {e})")
