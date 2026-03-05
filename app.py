import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd

# ===============================
# 1. 數據抓取
# ===============================
@st.cache_data(ttl=3600)
def get_v8_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.isel(time=-1).sel(
            depth=0, lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)
        ).load()
        u = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values).astype(np.float32)
        return subset.lat.values.astype(np.float32), subset.lon.values.astype(np.float32), u, v
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
# 2. 系統狀態管理
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Pro Navigation")
st.title("🛰️ HELIOS 智慧導航系統 (Pro 版)")

lat, lon, u_2d, v_2d = get_v8_data()

if 'current_ship_pos' not in st.session_state:
    st.session_state.current_ship_pos = None
if 'sim_step_idx' not in st.session_state:
    st.session_state.sim_step_idx = 0
if 'computed_path' not in st.session_state:
    st.session_state.computed_path = None

if lat is not None:
    # --- 嚴謹的陸地遮罩邏輯 ---
    LON, LAT = np.meshgrid(lon, lat)
    # 1. 台灣主島避讓 (橢圓近似)
    taiwan_mask = (((LAT - 23.7) / 1.78) ** 2 + ((LON - 121.0) / 0.88) ** 2) < 1
    # 2. 中國大陸沿岸避讓 (簡單線性劃分，避免穿過大陸)
    china_coast = (LAT > (0.8 * LON - 72.0)) 
    # 3. 澎湖區域
    penghu_mask = (((LAT - 23.55) / 0.15) ** 2 + ((LON - 119.6) / 0.15) ** 2) < 1
    
    forbidden = taiwan_mask | china_coast | penghu_mask

    # ===============================
    # 3. 側邊欄與模擬控制
    # ===============================
    with st.sidebar:
        st.header("🚢 導航參數")
        s_lat = st.number_input("起點緯度", value=22.35, format="%.2f")
        s_lon = st.number_input("起點經度", value=120.10, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        base_speed_kn = st.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
        
        st.divider()
        run_btn = st.button("🚀 重新計算航線", use_container_width=True)
        btn_next = st.button("⏭️ 下一步 (模擬移動)", use_container_width=True)
        if st.button("🔄 重置模擬"):
            st.session_state.current_ship_pos = [s_lat, s_lon]
            st.session_state.sim_step_idx = 0
            st.rerun()

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    # A* 計算 (與之前邏輯相同)
    if run_btn:
        st.session_state.sim_step_idx = 0
        st.session_state.current_ship_pos = [s_lat, s_lon]
        open_set, came_from, g_score = [], {}, {s_idx: 0.0}
        heapq.heappush(open_set, (0, s_idx))
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
                    cost = dist * (1.0 - 0.5 * np.dot([lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]]/np.hypot(1e-6, 1), [u_2d[ni,nj], v_2d[ni,nj]]))
                    tg = g_score[current] + cost
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        heapq.heappush(open_set, (tg + np.hypot(ni-g_idx[0], nj-g_idx[1]), (ni, nj)))

    # 移動邏輯
    if btn_next and st.session_state.computed_path:
        if st.session_state.sim_step_idx < len(st.session_state.computed_path) - 1:
            st.session_state.sim_step_idx += 1
            idx = st.session_state.computed_path[st.session_state.sim_step_idx]
            st.session_state.current_ship_pos = [lat[idx[0]], lon[idx[1]]]

    # ===============================
    # 4. 專業儀表板 (Metrics)
    # ===============================
    m1, m2, m3, m4 = st.columns(4)
    
    # 計算剩餘距離與時間
    rem_dist = 0
    heading = 0
    if st.session_state.computed_path:
        idx = st.session_state.sim_step_idx
        path = st.session_state.computed_path
        # 剩餘航程
        for k in range(idx, len(path)-1):
            rem_dist += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
        # 計算航向 (目前位置到下一點)
        if idx < len(path)-1:
            p1 = [lat[path[idx][0]], lon[path[idx][1]]]
            p2 = [lat[path[idx+1][0]], lon[path[idx+1][1]]]
            heading = calc_bearing(p1, p2)

    eta_hr = rem_dist / (base_speed_kn * 1.852) if base_speed_kn > 0 else 0
    
    m1.metric("📏 剩餘航程", f"{rem_dist:.1f} km")
    m2.metric("⏳ 預計所需時間", f"{eta_hr:.1f} hr")
    m3.metric("🧭 建議航向", f"{heading:.0f}°")
    # 模擬衛星數 (隨機跳動增加真實感)
    sat_count = 8 + (st.session_state.sim_step_idx % 4)
    m4.metric("📡 接收衛星數", f"{sat_count} Fix")

    # ===============================
    # 5. 地圖繪製
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.0, 125.0, 20.0, 27.0])
    
    # 海流背景
    spd = np.sqrt(u_2d**2 + v_2d**2)
    ax.pcolormesh(lon, lat, spd, cmap='YlGnBu', alpha=0.4)
    
    # 陸地特徵
    ax.add_feature(cfeature.LAND, facecolor='#222222', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.8, zorder=3)
    
    if st.session_state.computed_path:
        p = st.session_state.computed_path
        ax.plot([lon[x[1]] for x in p], [lat[x[0]] for x in p], color='#FF00FF', linewidth=2, zorder=5)
        
        # 船隻位置與目的地
        cur_pos = st.session_state.current_ship_pos or [s_lat, s_lon]
        ax.scatter(cur_pos[1], cur_pos[0], color='#00FF00', s=150, edgecolors='white', zorder=10)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=250, zorder=10)

    st.pyplot(fig)
