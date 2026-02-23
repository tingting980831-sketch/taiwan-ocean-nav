import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 導航系統初始化 ---
st.set_page_config(page_title="HELIOS 智慧導航儀 - 穩定版", layout="wide")

# 記憶起點與終點，確保不會跑掉
if 'curr_lon' not in st.session_state:
    st.session_state.curr_lon = 121.850
if 'curr_lat' not in st.session_state:
    st.session_state.curr_lat = 25.150
if 'dest_lon' not in st.session_state:
    st.session_state.dest_lon = 122.300
if 'dest_lat' not in st.session_state:
    st.session_state.dest_lat = 25.150

# --- 2. 側邊欄控制 ---
st.sidebar.header("🧭 導航控制中心")

loc_mode = st.sidebar.radio("定位模式", ["立即定位 (GPS 模擬)", "手動輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    st.sidebar.info(f"📍 GPS 定位中\n經度: {st.session_state.curr_lon:.3f}\n緯度: {st.session_state.curr_lat:.3f}")
    c_lon, c_lat = st.session_state.curr_lon, st.session_state.curr_lat
else:
    # 手動輸入時，直接同步到 session_state
    c_lon = st.sidebar.number_input("手動經度", value=st.session_state.curr_lon, format="%.3f", key="man_lon")
    c_lat = st.sidebar.number_input("手動緯度", value=st.session_state.curr_lat, format="%.3f", key="man_lat")
    st.session_state.curr_lon, st.session_state.curr_lat = c_lon, c_lat

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 目標設定")
d_lon = st.sidebar.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")
d_lat = st.sidebar.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
st.session_state.dest_lon, st.session_state.dest_lat = d_lon, d_lat

# 操作按鈕
st.sidebar.markdown("---")
btn_analyze = st.sidebar.button("🚀 確認執行 AI 分析 (連線衛星)", use_container_width=True)
btn_move = st.sidebar.button("🚢 模擬移動下一步 (執行導航)", use_container_width=True)

# --- 3. 核心邏輯處理 ---
# 如果點擊移動，先更新座標
if btn_move:
    # 往目標點靠近 10%，並直接存回 session_state
    st.session_state.curr_lat += (d_lat - st.session_state.curr_lat) * 0.1
    st.session_state.curr_lon += (d_lon - st.session_state.curr_lon) * 0.1
    c_lat, c_lon = st.session_state.curr_lat, st.session_state.curr_lon

# 只要有按任何一個鈕，就執行繪圖與分析
if btn_analyze or btn_move:
    with st.spinner('📡 正在透過 HELIOS 衛星抓取 HYCOM 即時海象數據...'):
        try:
            DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
            ds = xr.open_dataset(DATA_URL, decode_times=False)
            
            # 動態調整抓取範圍，確保起點和終點都在圖內，且不會跑掉
            pad = 0.5
            subset = ds.sel(lon=slice(min(c_lon, d_lon)-pad, max(c_lon, d_lon)+pad), 
                            lat=slice(min(c_lat, d_lat)-pad, max(c_lat, d_lat)+pad), 
                            depth=0).isel(time=-1).load()
            
            u_val = float(subset.water_u.interp(lat=c_lat, lon=c_lon))
            v_val = float(subset.water_v.interp(lat=c_lat, lon=c_lon))

            # 計算導航數據
            dist_rem = np.sqrt((d_lat-c_lat)**2 + (d_lon-c_lon)**2) * 60 
            head_angle = np.degrees(np.arctan2(d_lat - c_lat, d_lon - c_lon)) % 360
            sog_knots = (15.0 * 0.514 + (u_val * np.cos(np.radians(head_angle)) + v_val * np.sin(np.radians(head_angle)))) / 0.514
            fuel = max(min((1 - (15.0/sog_knots)**3) * 100 + 12.5, 18.4), 0.0)
            l_ms = (900/300)*4 + 15 + np.random.uniform(0, 5)

            # --- 儀表板顯示 ---
            st.subheader("📊 HELIOS 系統儀表板")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🚀 航速 (SOG)", f"{round(sog_knots,1)} kn")
            c2.metric("⛽ 省油率", f"{round(fuel,1)}%")
            c3.metric("🎯 剩餘距離", f"{round(dist_rem,1)} nmi")
            c4.metric("🧭 建議航向", f"{int(head_angle)}°")
            c5.metric("📡 衛星延遲", f"{round(l_ms,1)} ms")

            # --- 地圖繪製 ---
            fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([min(c_lon, d_lon)-0.6, max(c_lon, d_lon)+0.6, 
                           min(c_lat, d_lat)-0.6, max(c_lat, d_lat)+0.6])
            
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            ax.pcolormesh(subset.lon, subset.lat, mag, cmap='YlGn', alpha=0.8)
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#121212')
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white')

            # 標註：流向(紅)、建議航向(粉)
            ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, label='Current')
            hu, hv = np.cos(np.radians(head_angle)), np.sin(np.radians(head_angle))
            ax.quiver(c_lon, c_lat, hu, hv, color='#FF00FF', scale=4, width=0.015, label='AI Advice')
            
            ax.scatter(c_lon, c_lat, color='#FF00FF', s=150, edgecolors='white', label='Ship')
            ax.scatter(d_lon, d_lat, color='#00FF00', s=250, marker='*', edgecolors='white', label='Goal')
            
            ax.legend(loc='lower right')
            st.pyplot(fig)
            
            if btn_move:
                st.success(f"🚢 船隻已移動！當前位置：({round(c_lon,3)}, {round(c_lat,3)})")

        except Exception as e:
            st.error(f"數據加載失敗，可能是因為座標太靠近陸地或網路問題。")
