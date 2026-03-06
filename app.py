import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd

# ===============================
# 1. 數據抓取 (抓取 4D 序列以實現流場演化)
# ===============================
@st.cache_data(ttl=3600)
def get_v8_pro_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        # 抓取最近 8 個時段 (每段約 3 小時，涵蓋 24 小時預報)
        subset = ds.isel(time=slice(-8, None)).sel(
            depth=0, lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)
        ).load()
        u_4d = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v_4d = np.nan_to_num(subset.water_v.values).astype(np.float32)
        return subset.lat.values, subset.lon.values, u_4d, v_4d
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, None, None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

def calc_bearing(p1, p2):
    """計算兩點間的方位角 (0-360度)"""
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

# ===============================
# 2. 介面設定與狀態管理
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V8.3 Pro")
st.title("🛰️ HELIOS 智慧導航系統 (動態流場版)")

lat, lon, u_4d, v_4d = get_v8_pro_data()

# 初始化 Session State
if 'sim_step_idx' not in st.session_state:
    st.session_state.sim_step_idx = 0
if 'current_ship_pos' not in st.session_state:
    st.session_state.current_ship_pos = None
if 'computed_path' not in st.session_state:
    st.session_state.computed_path = None
if 'current_time_idx' not in st.session_state:
    st.session_state.current_time_idx = 0

if lat is not None:
    # --- 嚴謹的陸地遮罩 ---
    LON, LAT = np.meshgrid(lon, lat)
    # 台灣主島避讓
    taiwan_mask = (((LAT - 23.7) / 1.78) ** 2 + ((LON - 121.0) / 0.88) ** 2) < 1
    # 中國沿岸避讓 (確保路徑不切過福建)
    china_coast = (LAT > (0.85 * LON - 78.0)) 
    # 澎湖區域
    penghu_mask = (((LAT - 23.55) / 0.18) ** 2 + ((LON - 119.6) / 0.18) ** 2) < 1
    forbidden = taiwan_mask | china_coast | penghu_mask

    # ===============================
    # 3. 側邊欄控制
    # ===============================
    with st.sidebar:
        st.header("🚢 導航參數設定")
        s_lat = st.number_input("起點緯度", value=22.35, format="%.2f")
        s_lon = st.number_input("起點經度", value=120.10, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        base_speed_kn = st.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
        
        st.divider()
        run_btn = st.button("🚀 執行航線計算", use_container_width=True)
        btn_next = st.button("⏭️ 下一步 (模擬 1H 航行)", use_container_width=True)
        if st.button("🔄 重置導航", use_container_width=True):
            st.session_state.sim_step_idx = 0
            st.session_state.current_time_idx = 0
            st.session_state.current_ship_pos = [s_lat, s_lon]
            st.rerun()

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    # A* 演算法 (考量海流向量)
    if run_btn:
        st.session_state.sim_step_idx = 0
        st.session_state.current_time_idx = 0
        st.session_state.current_ship_pos = [s_lat, s_lon]
        
        open_set, came_from, g_score = [], {}, {s_idx: 0.0}
        heapq.heappush(open_set, (0, s_idx))
        
        # 使用當前時段的流場計算初始路徑
        u_curr = u_4d[0]
        v_curr = v_4d[0]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == g_idx:
                path = []
                while current in came_from:
                    path.append(current); current = came_from[current]
                path.append(s_idx)
                st.session_state.computed_path = path[::-1]
                break
            
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current[0]+d[0], current[1]+d[1]
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    dist = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
                    # 簡單向量助力模型
                    move_vec = np.array([lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]])
                    move_vec = move_vec / (np.linalg.norm(move_vec) + 1e-6)
                    assist = np.dot(move_vec, [u_curr[ni, nj], v_curr[ni, nj]])
                    
                    cost = dist * (1.0 - 0.6 * assist)
                    tg = g_score[current] + cost
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        heapq.heappush(open_set, (tg + np.hypot(ni-g_idx[0], nj-g_idx[1]), (ni, nj)))

    # 下一步：船隻移動並「演化流場」
    if btn_next and st.session_state.computed_path:
        if st.session_state.sim_step_idx < len(st.session_state.computed_path) - 1:
            st.session_state.sim_step_idx += 1
            # 模擬流場隨時間演化：每走一段步數，切換到 HYCOM 的下一個預報時段
            st.session_state.current_time_idx = min(st.session_state.sim_step_idx // 6, 7)
            
            idx_node = st.session_state.computed_path[st.session_state.sim_step_idx]
            st.session_state.current_ship_pos = [lat[idx_node[0]], lon[idx_node[1]]]
            
            if st.session_state.sim_step_idx % 6 == 0:
                st.toast(f"🌊 已同步更新未來 {st.session_state.current_time_idx * 3} 小時預報流場")

    # ===============================
    # 4. 專業動態儀表板
    # ===============================
    m1, m2, m3, m4 = st.columns(4)
    
    rem_dist = 0
    heading = 0
    if st.session_state.computed_path:
        curr_i = st.session_state.sim_step_idx
        path = st.session_state.computed_path
        # 計算剩餘距離
        for k in range(curr_i, len(path)-1):
            rem_dist += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
        # 計算當前建議航向
        if curr_i < len(path)-1:
            p_now = [lat[path[curr_i][0]], lon[path[curr_i][1]]]
            p_next = [lat[path[curr_i+1][0]], lon[path[curr_i+1][1]]]
            heading = calc_bearing(p_now, p_next)

    eta_hr = rem_dist / (base_speed_kn * 1.852) if base_speed_kn > 0 else 0
    sat_fix = 8 + (st.session_state.sim_step_idx % 5) # 模擬衛星數變動
    
    m1.metric("📏 剩餘航程", f"{rem_dist:.1f} km")
    m2.metric("⏳ 預計所需時間", f"{eta_hr:.1f} hr")
    m3.metric("🧭 建議航向", f"{heading:.0f}°")
    m4.metric("📡 接收衛星數", f"{sat_fix} Fix")

    # ===============================
    # 5. 地圖繪製 (顯示動態流場)
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.0, 125.0, 20.0, 27.0])
    
    # 繪製當前時段的流場背景
    t_idx = st.session_state.current_time_idx
    spd = np.sqrt(u_4d[t_idx]**2 + v_4d[t_idx]**2)
    ax.pcolormesh(lon, lat, spd, cmap='YlGnBu', alpha=0.5, shading='auto')
    
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.6, zorder=3)
    
    if st.session_state.computed_path:
        p = st.session_state.computed_path
        ax.plot([lon[x[1]] for x in p], [lat[x[0]] for x in p], color='#FF00FF', linewidth=2, zorder=5, label='建議航線')
        
        # 顯示船隻目前位置
        cur_p = st.session_state.current_ship_pos or [s_lat, s_lon]
        ax.scatter(cur_p[1], cur_p[0], color='#00FF00', s=180, edgecolors='white', linewidth=2, zorder=10, label='目前位置')
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=300, zorder=10, label='目的地')

    ax.legend(loc='lower right', facecolor='#222222', labelcolor='white')
    st.pyplot(fig)
