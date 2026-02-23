import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 系統初始化與記憶功能 ---
st.set_page_config(page_title="HELIOS 台灣衛星導航監控系統", layout="wide")

if 'curr_lon' not in st.session_state:
    st.session_state.curr_lon = 121.850
if 'curr_lat' not in st.session_state:
    st.session_state.curr_lat = 25.150
if 'dest_lon' not in st.session_state:
    st.session_state.dest_lon = 122.300
if 'dest_lat' not in st.session_state:
    st.session_state.dest_lat = 25.150

# --- 2. 側邊欄控制台 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
loc_mode = st.sidebar.radio("定位模式", ["立即定位 (GPS 模擬)", "手動輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    c_lon, c_lat = st.session_state.curr_lon, st.session_state.curr_lat
else:
    c_lon = st.sidebar.number_input("手動設定經度", value=st.session_state.curr_lon, format="%.3f")
    c_lat = st.sidebar.number_input("手動設定緯度", value=st.session_state.curr_lat, format="%.3f")
    st.session_state.curr_lon, st.session_state.curr_lat = c_lon, c_lat

st.sidebar.markdown("---")
d_lon = st.sidebar.number_input("目標經度", value=st.session_state.dest_lon, format="%.3f")
d_lat = st.sidebar.number_input("目標緯度", value=st.session_state.dest_lat, format="%.3f")
st.session_state.dest_lon, st.session_state.dest_lat = d_lon, d_lat

st.sidebar.markdown("---")
with st.sidebar.status("HELIOS 衛星連線中...", expanded=False) as status:
    st.write(f"🛰️ 衛星軌道: 900km LEO (區域強化)")
    st.write(f"📶 訊號強度: {np.random.randint(92, 99)}%")
    status.update(label="✅ 衛星鏈路穩定 (隨傳隨回)", state="complete")

btn_analyze = st.sidebar.button("🚀 確認執行 AI 分析", use_container_width=True)
btn_move = st.sidebar.button("🚢 模擬移動下一步", use_container_width=True)

if btn_move:
    st.session_state.curr_lat += (d_lat - st.session_state.curr_lat) * 0.1
    st.session_state.curr_lon += (d_lon - st.session_state.curr_lon) * 0.1
    c_lat, c_lon = st.session_state.curr_lat, st.session_state.curr_lon

# --- 3. 核心數據處理函數 ---
def get_nav_data(u, v, clat, clon, dlat, dlon):
    dist = np.sqrt((dlat-clat)**2 + (dlon-clon)**2) * 60 
    head = np.degrees(np.arctan2(dlat - clat, dlon - clon)) % 360
    vs_ms = 15.0 * 0.514 
    sog_ms = vs_ms + (u * np.cos(np.radians(head)) + v * np.sin(np.radians(head)))
    sog_knots = sog_ms / 0.514
    fuel = max(min((1 - (vs_ms / sog_ms)**3) * 100 + 15.2, 25.4), 0.0)
    latency = (900/300)*4 + 15 + np.random.uniform(0, 5)
    return round(sog_knots, 1), round(fuel, 1), int(head), round(dist, 1), round(latency, 1)

# --- 4. 執行與繪圖 ---
if btn_analyze or btn_move:
    with st.spinner('📡 正在下載 HELIOS 即時數據...'):
        try:
            DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
            ds = xr.open_dataset(DATA_URL, decode_times=False)
            
            margin = 0.6
            lon_min, lon_max = min(c_lon, d_lon) - margin, max(c_lon, d_lon) + margin
            lat_min, lat_max = min(c_lat, d_lat) - margin, max(c_lat, d_lat) + margin

            # 動態畫布比例
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min
            aspect_ratio = lon_range / lat_range
            base_width = 10
            fig_height = base_width / aspect_ratio
            fig_height = max(min(fig_height, 12), 6)

            subset = ds.sel(lon=slice(lon_min, lon_max), 
                            lat=slice(lat_min, lat_max), 
                            depth=0).isel(time=-1).load()
            
            u_val = float(subset.water_u.interp(lat=c_lat, lon=c_lon))
            v_val = float(subset.water_v.interp(lat=c_lat, lon=c_lon))
            sog, f_save, head, d_rem, l_ms = get_nav_data(u_val, v_val, c_lat, c_lon, d_lat, d_lon)

            # 數據看板
            st.subheader("📊 HELIOS 衛星決策儀表板")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🚀 航速 (SOG)", f"{sog} kn")
            c2.metric("⛽ 節能效益", f"{f_save}%")
            c3.metric("🎯 剩餘距離", f"{d_rem} nmi")
            c4.metric("🧭 建議航向", f"{head}°")
            c5.metric("📡 衛星延遲", f"{l_ms} ms")

            # --- 地圖繪製 (移除所有視覺白框) ---
            fig, ax = plt.subplots(figsize=(base_width, fig_height), 
                                   subplot_kw={'projection': ccrs.PlateCarree()})
            
            ax.set_aspect('equal', adjustable='datalim') 
            ax.set_extent([lon_min, lon_max, lat_min, lat_max])
            
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            
            # 1. 繪製流場：edgecolors=None 移除格子間白框
            ax.pcolormesh(subset.lon, subset.lat, mag, cmap='YlGn', alpha=0.9, edgecolors='none', shading='auto')
            
            # 2. 繪製陸地：調整顏色並移除線條邊框感
            land_color = '#121212'
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor=land_color, edgecolor='none')
            
            # 3. 繪製海岸線：使用與陸地相同的顏色，確保沒有顯眼的邊界白線
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor=land_color, linewidth=0.5)

            # 向量與標註
            ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, label='Actual Current', zorder=4)
            hu, hv = np.cos(np.radians(head)), np.sin(np.radians(head))
            ax.quiver(c_lon, c_lat, hu, hv, color='#FF00FF', scale=4, width=0.012, label='AI Suggested', zorder=4)
            
            ax.plot([c_lon, d_lon], [c_lat, d_lat], 'w:', alpha=0.3, zorder=3)
            ax.scatter(c_lon, c_lat, color='#FF00FF', s=150, edgecolors='white', linewidth=0.5, label='Ship Pos', zorder=5)
            ax.scatter(d_lon, d_lat, color='#00FF00', s=250, marker='*', edgecolors='white', linewidth=0.5, label='Dest', zorder=5)
            
            ax.legend(loc='lower right', framealpha=0.5)
            st.pyplot(fig)
            st.success("✅ 數據可視化完成：已移除所有結構邊框。")

        except Exception as e:
            st.error(f"連線異常：{e}")
