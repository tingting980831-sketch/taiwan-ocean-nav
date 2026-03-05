import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd

# ===============================
# 1. 數據抓取 (20N-27N, 117E-125E)
# ===============================
@st.cache_data(ttl=3600)
def get_v8_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        # 抓取最新 24 小時數據
        subset = ds.isel(time=slice(-24, None)).sel(
            depth=0, lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)
        ).load()
        u_4d = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v_4d = np.nan_to_num(subset.water_v.values).astype(np.float32)
        try:
            dt_raw = xr.decode_cf(subset).time.values
            dt_display = pd.to_datetime(dt_raw[0]).strftime('%Y-%m-%d %H:%M')
        except:
            dt_display = "Dynamic 24H Forecast"
        return subset.lat.values.astype(np.float32), subset.lon.values.astype(np.float32), u_4d, v_4d, dt_display
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, None, None, None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統 UI 設定與狀態管理
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V8 Navigation Simulation")
st.title("🛰️ HELIOS 智慧導航系統 (動態模擬版)")

lat, lon, u_4d, v_4d, ocean_time = get_v8_data()

# 初始化 Session State 用於模擬船隻位置
if 'current_ship_pos' not in st.session_state:
    st.session_state.current_ship_pos = None # 初始設為空，由計算後產生
if 'sim_step_idx' not in st.session_state:
    st.session_state.sim_step_idx = 0
if 'computed_path' not in st.session_state:
    st.session_state.computed_path = None

if lat is not None:
    LON, LAT = np.meshgrid(lon, lat)
    # 陸地與澎湖硬障礙
    land_mask = (((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1
    penghu_mask = (((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1
    forbidden = land_mask | penghu_mask 

    with st.sidebar:
        st.header("🚢 導航參數設定")
        s_lat = st.number_input("起點緯度", value=22.35, format="%.2f")
        s_lon = st.number_input("起點經度", value=120.10, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        base_speed = st.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
        
        st.divider()
        run_btn = st.button("🚀 執行航線計算", use_container_width=True)
        
        st.subheader("🕹️ 航行模擬控制")
        col1, col2 = st.columns(2)
        btn_next = col1.button("⏭️ 下一步", use_container_width=True)
        btn_reset = col2.button("🔄 重置", use_container_width=True)

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    # ===============================
    # 3. A* 演算法邏輯
    # ===============================
    if run_btn:
        # 重置模擬狀態
        st.session_state.sim_step_idx = 0
        st.session_state.current_ship_pos = [s_lat, s_lon]
        
        # --- A. 優化航線 (考慮海流) ---
        open_set, came_from, g_score = [], {}, {s_idx: 0.0}
        heapq.heappush(open_set, (0, s_idx))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == g_idx:
                path = []
                while current in came_from:
                    path.append(current); current = came_from[current]
                path.append(s_idx); 
                st.session_state.computed_path = path[::-1]
                break
            
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current[0]+d[0], current[1]+d[1]
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    step_dist = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
                    u_curr, v_curr = u_4d[0, ni, nj], v_4d[0, ni, nj]
                    
                    dx, dy = lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_curr, v_curr])
                    
                    cost = step_dist * (1 - 0.75 * assist)
                    tg = g_score[current] + cost
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj)))

    # ===============================
    # 4. 「下一步」移動邏輯
    # ===============================
    if btn_next and st.session_state.computed_path:
        if st.session_state.sim_step_idx < len(st.session_state.computed_path) - 1:
            st.session_state.sim_step_idx += 1
            next_node = st.session_state.computed_path[st.session_state.sim_step_idx]
            st.session_state.current_ship_pos = [lat[next_node[0]], lon[next_node[1]]]

    if btn_reset:
        st.session_state.sim_step_idx = 0
        st.session_state.current_ship_pos = [s_lat, s_lon]
        st.session_state.computed_path = None
        st.rerun()

    # ===============================
    # 5. 儀表板顯示
    # ===============================
    c1, c2, c3 = st.columns(3)
    c1.metric("🌊 數據時間", ocean_time)
    
    # 計算剩餘距離
    rem_dist = 0
    if st.session_state.computed_path:
        curr_p = st.session_state.sim_step_idx
        rem_path = st.session_state.computed_path[curr_p:]
        for k in range(len(rem_path)-1):
            rem_dist += haversine(lat[rem_path[k][0]], lon[rem_path[k][1]], lat[rem_path[k+1][0]], lon[rem_path[k+1][1]])
    
    c2.metric("📏 剩餘航程", f"{rem_dist:.1f} km")
    c3.metric("📍 模擬進度", f"Step {st.session_state.sim_step_idx}")

    # ===============================
    # 6. 地圖繪製
    # ===============================
    fig, ax = plt.subplots(figsize=(11, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([116.8, 125.2, 19.8, 27.2]) 
    speed_map = np.sqrt(u_4d[0]**2 + v_4d[0]**2)
    ax.pcolormesh(lon, lat, speed_map, cmap='YlGn', alpha=0.7)
    ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1)

    # 畫出路徑
    if st.session_state.computed_path:
        path = st.session_state.computed_path
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=6, label='優化路徑') 
        
        # 標記起點與終點
        ax.scatter(s_lon, s_lat, color='white', s=50, alpha=0.5, zorder=7)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, zorder=7, label='目的地')

        # 模擬船隻目前位置 (綠色大點)
        if st.session_state.current_ship_pos:
            cur_la, cur_lo = st.session_state.current_ship_pos
            ax.scatter(cur_lo, cur_la, color='#00FF00', s=180, edgecolors='black', linewidth=2, zorder=10, label='當前船位')

    ax.legend(loc='upper right')
    st.pyplot(fig)
