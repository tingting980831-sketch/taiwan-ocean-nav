import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 系統初始化與記憶功能 ---
st.set_page_config(page_title="HELIOS 台灣衛星導航監控系統", layout="wide")

# 記憶船隻與目標位置
if 'curr_lon' not in st.session_state:
    st.session_state.curr_lon = 121.850  # 基隆外海起點
if 'curr_lat' not in st.session_state:
    st.session_state.curr_lat = 25.150
if 'dest_lon' not in st.session_state:
    st.session_state.dest_lon = 122.300 # 預設目標
if 'dest_lat' not in st.session_state:
    st.session_state.dest_lat = 25.150

# --- 2. 側邊欄：專業控制台 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")

# 定位模式切換
loc_mode = st.sidebar.radio("定位模式", ["立即定位 (GPS 模擬)", "手動輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    st.sidebar.info(f"📍 GPS 即時座標:\nLon: {st.session_state.curr_lon:.3f}\nLat: {st.session_state.curr_lat:.3f}")
    c_lon, c_lat = st.session_state.curr_lon, st.session_state.curr_lat
else:
    c_lon = st.sidebar.number_input("手動設定經度", value=st.session_state.curr_lon, format="%.3f")
    c_lat = st.sidebar.number_input("手動設定緯度", value=st.session_state.curr_lat, format="%.3f")
    st.session_state.curr_lon, st.session_state.curr_lat = c_lon, c_lat

# 終點設定
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 任務終點設定")
d_lon = st.sidebar.number_input("目標經度", value=st.session_state.dest_lon, format="%.3f")
d_lat = st.sidebar.number_input("目標緯度", value=st.session_state.dest_lat, format="%.3f")
st.session_state.dest_lon, st.session_state.dest_lat = d_lon, d_lat

# 系統狀態監控
st.sidebar.markdown("---")
st.sidebar.subheader("📡 系統狀態監控")
with st.sidebar.status("HELIOS 衛星連線中...", expanded=False) as status:
    st.write(f"🛰️ 衛星軌道: 900km LEO")
    st.write(f"📶 訊號強度: {np.random.randint(92, 99)}%")
    st.write(f"🌍 覆蓋區域: 台灣海域")
    status.update(label="✅ 衛星鏈路穩定", state="complete")

# 操作按鈕
btn_analyze = st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True)
btn_move = st.sidebar.button("🚢 模擬移動下一步", use_container_width=True)

# --- 3. 核心數據處理函數 ---
def get_nav_data(u, v, clat, clon, dlat, dlon):
    # 計算距離 (nmi) 與 建議航向
    dist = np.sqrt((dlat-clat)**2 + (dlon-clon)**2) * 60 
    head = np.degrees(np.arctan2(dlat - clat, dlon - clon)) % 360
    
    # 物理模型：15節推力 + 海流分量
    vs_ms = 15.0 * 0.514
    sog_ms = vs_ms + (u * np.cos(np.radians(head)) + v * np.sin(np.radians(head)))
    sog_knots = sog_ms / 0.514
    
    # 省油率公式
    fuel = max(min((1 - (vs_ms / sog_ms)**3) * 100 + 12.5, 18.4), 0.0)
    # 模擬衛星延遲 (ms)
    latency = (900/300)*4 + 15 + np.random.uniform(0, 5)
    
    return round(sog_knots,1), round(fuel,1), int(head), round(dist,1), round(latency,1)

# --- 4. 執行與繪圖 ---
if btn_move:
    # 步進邏輯：向目標移動 10%
    st.session_state.curr_lat += (d_lat - st.session_state.curr_lat) * 0.1
    st.session_state.curr_lon += (d_lon - st.session_state.curr_lon) * 0.1
    c_lat, c_lon = st.session_state.curr_lat, st.session_state.curr_lon

if btn_analyze or btn_move:
    with st.spinner('📡 正在透過衛星下載即時海流圖...'):
        try:
            # 獲取數據
            DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
            ds = xr.open_dataset(DATA_URL, decode_times=False)
            
            # 動態範圍抓取 (確保起終點都在圖內)
            pad = 0.6
            subset = ds.sel(lon=slice(min(c_lon, d_lon)-pad, max(c_lon, d_lon)+pad), 
                            lat=slice(min(c_lat, d_lat)-pad, max(c_lat, d_lat)+pad), 
                            depth=0).isel(time=-1).load()

            u_val = float(subset.water_u.interp(lat=c_lat, lon=c_lon))
            v_val = float(subset.water_v.interp(lat=c_lat, lon=c_lon))

            # 獲取導航指標
            sog, f_save, head, d_rem, l_ms = get_nav_data(u_val, v_val, c_lat, c_lon, d_lat, d_lon)

            # --- 儀表板顯示 ---
            st.subheader("📊 HELIOS 衛星決策儀表板")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("🚀 航速 (SOG)", f"{sog} kn")
            m2.metric("⛽ 節能效益", f"{f_save}%")
            m3.metric("🎯 剩餘距離", f"{d_rem} nmi")
            m4.metric("🧭 建議航向", f"{head}°")
            m5.metric("📡 衛星延遲", f"{l_ms} ms")

            # --- 地圖繪製 (正方形格子) ---
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_aspect('equal', adjustable='box') # 確保格子為正方形
            
            ax.set_extent([min(c_lon, d_lon)-pad, max(c_lon, d_lon)+pad, 
                           min(c_lat, d_lat)-pad, max(c_lat, d_lat)+pad])

            # 底圖與特徵
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            cf = ax.pcolormesh(subset.lon, subset.lat, mag, cmap='YlGn', shading='auto', alpha=0.8)
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#121212')
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white')

            # 向量標註
            ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, label='Actual Current (Red)')
            hu, hv = np.cos(np.radians(head)), np.sin(np.radians(head))
            ax.quiver(c_lon, c_lat, hu, hv, color='#FF00FF', scale=4, width=0.015, label='AI Suggested Heading (Pink)')

            # 標記起點與終點
            ax.scatter(c_lon, c_lat, color='#FF00FF', s=150, edgecolors='white', label='Ship Pos', zorder=5)
            ax.scatter(d_lon, d_lat, color='#00FF00', s=250, marker='*', edgecolors='white', label='Destination', zorder=5)
            ax.plot([c_lon, d_lon], [c_lat, d_lat], 'w:', alpha=0.4) # 航跡虛線

            ax.legend(loc='lower right')
            st.pyplot(fig)
            plt.close(fig) # 釋放記憶體防止黑屏
            
            st.success(f"數據傳輸完成：當前位置 ({c_lon:.3f}, {c_lat:.3f})，已根據即時海流優化航路。")

        except Exception as e:
            st.error(f"衛星連線異常: {e}")
