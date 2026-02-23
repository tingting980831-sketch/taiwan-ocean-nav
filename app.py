import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 系統初始化 ---
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
    st.sidebar.info(f"📍 GPS 即時座標:\nLon: {st.session_state.curr_lon:.3f}\nLat: {st.session_state.curr_lat:.3f}")
    c_lon, c_lat = st.session_state.curr_lon, st.session_state.curr_lat
else:
    c_lon = st.sidebar.number_input("手動設定經度", value=st.session_state.curr_lon, format="%.3f")
    c_lat = st.sidebar.number_input("手動設定緯度", value=st.session_state.curr_lat, format="%.3f")
    st.session_state.curr_lon, st.session_state.curr_lat = c_lon, c_lat

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 任務終點設定")
d_lon = st.sidebar.number_input("目標經度", value=st.session_state.dest_lon, format="%.3f")
d_lat = st.sidebar.number_input("目標緯度", value=st.session_state.dest_lat, format="%.3f")
st.session_state.dest_lon, st.session_state.dest_lat = d_lon, d_lat

st.sidebar.markdown("---")
btn_analyze = st.sidebar.button("🚀 執行 AI 路徑分析", use_container_width=True)
btn_move = st.sidebar.button("🚢 模擬移動下一步", use_container_width=True)

# --- 3. 核心數據處理 ---
def get_nav_data(u, v, clat, clon, dlat, dlon):
    dist = np.sqrt((dlat-clat)**2 + (dlon-clon)**2) * 60 
    head = np.degrees(np.arctan2(dlat - clat, dlon - clon)) % 360
    vs_ms = 15.0 * 0.514
    sog_ms = vs_ms + (u * np.cos(np.radians(head)) + v * np.sin(np.radians(head)))
    sog_knots = sog_ms / 0.514
    fuel = max(min((1 - (vs_ms / sog_ms)**3) * 100 + 12.5, 18.4), 0.0)
    latency = (900/300)*4 + 15 + np.random.uniform(0, 5)
    return round(sog_knots,1), round(fuel,1), int(head), round(dist,1), round(latency,1)

# --- 4. 執行與繪圖 ---
if btn_move:
    st.session_state.curr_lat += (d_lat - st.session_state.curr_lat) * 0.1
    st.session_state.curr_lon += (d_lon - st.session_state.curr_lon) * 0.1
    c_lat, c_lon = st.session_state.curr_lat, st.session_state.curr_lon

if btn_analyze or btn_move:
    with st.spinner('📡 正在抓取衛星即時數據...'):
        try:
            DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
            ds = xr.open_dataset(DATA_URL, decode_times=False)
            
            # 抓取範圍：稍微放大以確保視覺舒適
            pad = 0.8
            subset = ds.sel(lon=slice(min(c_lon, d_lon)-pad, max(c_lon, d_lon)+pad), 
                            lat=slice(min(c_lat, d_lat)-pad, max(c_lat, d_lat)+pad), 
                            depth=0).isel(time=-1).load()

            u_val = float(subset.water_u.interp(lat=c_lat, lon=c_lon))
            v_val = float(subset.water_v.interp(lat=c_lat, lon=c_lon))
            sog, f_save, head, d_rem, l_ms = get_nav_data(u_val, v_val, c_lat, c_lon, d_lat, d_lon)

            # --- 儀表板 ---
            st.subheader("📊 HELIOS 衛星決策儀表板")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("🚀 航速", f"{sog} kn")
            m2.metric("⛽ 節能", f"{f_save}%")
            m3.metric("🎯 距離", f"{d_rem} nmi")
            m4.metric("🧭 建議航向", f"{head}°")
            m5.metric("📡 衛星延遲", f"{l_ms} ms")

            # --- 地圖區 ---
            # figsize 調整為正方形比例 (8,8)
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # 重要：強制 1:1 物理比例，格子才不會變扁
            ax.set_aspect('equal', adjustable='box') 
            
            # 設定地圖範圍 (與數據範圍同步)
            ax.set_extent([min(c_lon, d_lon)-pad, max(c_lon, d_lon)+pad, 
                           min(c_lat, d_lat)-pad, max(c_lat, d_lat)+pad])

            # 繪製流場底圖
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            # shading='flat' 或 'auto' 配合正確的 lon/lat 能呈現整齊的格子
            cf = ax.pcolormesh(subset.lon, subset.lat, mag, cmap='viridis', shading='auto', alpha=0.8)
            plt.colorbar(cf, ax=ax, label='Current Speed (m/s)', fraction=0.046, pad=0.04)

            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#121212', zorder=2)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', zorder=3)

            # 導航向量
            ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, label='Actual Current', zorder=4)
            hu, hv = np.cos(np.radians(head)), np.sin(np.radians(head))
            ax.quiver(c_lon, c_lat, hu, hv, color='#FF00FF', scale=4, width=0.012, label='HELIOS Advice', zorder=5)

            # 位置標記
            ax.scatter(c_lon, c_lat, color='#FF00FF', s=150, edgecolors='white', label='Ship', zorder=6)
            ax.scatter(d_lon, d_lat, color='#00FF00', s=250, marker='*', edgecolors='white', label='Goal', zorder=6)
            ax.plot([c_lon, d_lon], [c_lat, d_lat], 'w:', alpha=0.5, zorder=1) 

            ax.legend(loc='lower right', fontsize='small')
            st.pyplot(fig)
            plt.close(fig) 

        except Exception as e:
            st.error(f"分析失敗: {e}")
