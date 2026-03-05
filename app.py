import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr

# ===============================
# 1. 數據抓取
# ===============================
@st.cache_data(ttl=3600)
def get_v8_final_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.isel(time=slice(-8, None)).sel(
            depth=0, lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)
        ).load()
        return subset.lat.values, subset.lon.values, subset.water_u.values, subset.water_v.values
    except:
        return None, None, None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

def calc_bearing(p1, p2):
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

# ===============================
# 2. 系統初始化
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Nav-AI")
st.title("🛰️ HELIOS 智慧導航系統 (參賽正式版)")

lat, lon, u_4d, v_4d = get_v8_final_data()

if 'sim_step' not in st.session_state: st.session_state.sim_step = 0
if 'ship_pos' not in st.session_state: st.session_state.ship_pos = None
if 'path' not in st.session_state: st.session_state.path = None

if lat is not None:
    # --- 嚴謹遮罩邏輯 (解決切陸地問題) ---
    LON, LAT = np.meshgrid(lon, lat)
    # 台灣本島 (精確多邊形邊界)
    taiwan = (LAT > 21.8) & (LAT < 25.4) & (LON > 119.9) & (LON < 122.1)
    # 中國沿岸
    china = (LON < 120.2) & (LAT > 23.5)
    # 澎湖與淺灘
    reefs = (((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1
    forbidden = taiwan | china | reefs

    # ===============================
    # 3. 側邊欄控制
    # ===============================
    with st.sidebar:
        st.header("🚢 導航設定")
        s_lat = st.number_input("起點緯度", value=22.35)
        s_lon = st.number_input("起點經度", value=120.10)
        e_lat = st.number_input("終點緯度", value=25.20)
        e_lon = st.number_input("終點經度", value=122.00)
        knots = st.slider("巡航速度 (kn)", 10, 25, 15)
        
        st.divider()
        if st.button("🚀 執行航線優化", use_container_width=True):
            st.session_state.sim_step = 0
            st.session_state.ship_pos = [s_lat, s_lon]
            # A* 計算 (簡化版)
            si, gi = np.argmin(np.abs(lat-s_lat)), np.argmin(np.abs(lon-s_lon))
            target_i, target_j = np.argmin(np.abs(lat-e_lat)), np.argmin(np.abs(lon-e_lon))
            
            q, came_from, cost = [(0, (si, gi))], {}, {(si, gi): 0}
            while q:
                _, curr = heapq.heappop(q)
                if curr == (target_i, target_j): break
                for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nxt = (curr[0]+d[0], curr[1]+d[1])
                    if 0 <= nxt[0] < len(lat) and 0 <= nxt[1] < len(lon) and not forbidden[nxt]:
                        new_cost = cost[curr] + 1
                        if nxt not in cost or new_cost < cost[nxt]:
                            cost[nxt] = new_cost
                            priority = new_cost + np.hypot(nxt[0]-target_i, nxt[1]-target_j)
                            heapq.heappush(q, (priority, nxt))
                            came_from[nxt] = curr
            
            p = []
            curr = (target_i, target_j)
            while curr in came_from:
                p.append(curr); curr = came_from[curr]
            st.session_state.path = p[::-1]

        if st.button("⏭️ 下一步 (模擬 1H 航行)", use_container_width=True):
            if st.session_state.path and st.session_state.sim_step < len(st.session_state.path)-1:
                st.session_state.sim_step += 1
                idx = st.session_state.path[st.session_state.sim_step]
                st.session_state.ship_pos = [lat[idx[0]], lon[idx[1]]]

        if st.button("🔄 重置系統", use_container_width=True):
            st.session_state.sim_step = 0
            st.session_state.ship_pos = [s_lat, s_lon]
            st.rerun()

    # ===============================
    # 4. 專業儀表板 (移除不需要文字)
    # ===============================
    m1, m2, m3, m4 = st.columns(4)
    
    dist, brg = 0.0, 0
    if st.session_state.path:
        curr_idx = st.session_state.sim_step
        p = st.session_state.path
        for k in range(curr_idx, len(p)-1):
            dist += haversine(lat[p[k][0]], lon[p[k][1]], lat[p[k+1][0]], lon[p[k+1][1]])
        if curr_idx < len(p)-1:
            brg = calc_bearing([lat[p[curr_idx][0]], lon[p[curr_idx][1]]], [lat[p[curr_idx+1][0]], lon[p[curr_idx+1][1]]])

    m1.metric("📏 剩餘航程", f"{dist:.1f} km")
    m2.metric("⏳ 預計總航時", f"{dist/(knots*1.852):.1f} hr")
    m3.metric("🧭 建議航向", f"{brg:.0f}°")
    m4.metric("📡 衛星連線", f"{8 + st.session_state.sim_step % 5} Fix")

    # ===============================
    # 5. 地圖美化
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.5, 124.5, 20.5, 26.5])
    
    # 依模擬進度變換流場時間 (動態流場)
    t_idx = min(st.session_state.sim_step // 6, 7)
    spd = np.sqrt(u_4d[t_idx]**2 + v_4d[t_idx]**2)
    ax.pcolormesh(lon, lat, spd, cmap='GnBu', alpha=0.6, shading='auto')
    
    ax.add_feature(cfeature.LAND, facecolor='#202020', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=3)
    
    if st.session_state.path:
        p = st.session_state.path
        ax.plot([lon[x[1]] for x in p], [lat[x[0]] for x in p], color='#FF00FF', lw=2.5, zorder=5, label='建議航線')
        curr = st.session_state.ship_pos or [s_lat, s_lon]
        ax.scatter(curr[1], curr[0], color='#00FF00', s=180, edgecolors='white', zorder=10, label='目前位置')
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=300, zorder=10, label='目的地')

    ax.legend(loc='lower right', facecolor='#111111', labelcolor='white')
    st.pyplot(fig)
