import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr

# ===============================
# 1. 數據抓取與緩存
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
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

# ===============================
# 2. 系統初始化
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Nav-AI")
st.title("🛰️ HELIOS 智慧導航系統 (路徑優化版)")

lat, lon, u_4d, v_4d = get_v8_final_data()

if 'sim_step' not in st.session_state: st.session_state.sim_step = 0
if 'ship_pos' not in st.session_state: st.session_state.ship_pos = None
if 'path' not in st.session_state: st.session_state.path = None

if lat is not None:
    # --- 遮罩定義 ---
    LON, LAT = np.meshgrid(lon, lat)
    taiwan = (LAT > 21.85) & (LAT < 25.35) & (LON > 120.05) & (LON < 122.05)
    china = (LON < 119.85) & (LAT > 23.75)
    forbidden = taiwan | china

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
            si, sj = np.argmin(np.abs(lat-s_lat)), np.argmin(np.abs(lon-s_lon))
            gi, gj = np.argmin(np.abs(lat-e_lat)), np.argmin(np.abs(lon-e_lon))
            
            # 自動脫困
            while forbidden[si, sj]: si += 1
            st.session_state.ship_pos = [lat[si], lon[sj]]
            
            u_now, v_now = u_4d[0], v_4d[0]
            # (優先值, 座標, 前一個前進方向)
            q = [(0, (si, sj), (0,0))]
            came_from, cost = {(si, sj): 0}, {(si, sj): 0}
            
            while q:
                _, curr, last_d = heapq.heappop(q)
                if curr == (gi, gj): break
                
                for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nxt = (curr[0]+d[0], curr[1]+d[1])
                    if 0 <= nxt[0] < len(lat) and 0 <= nxt[1] < len(lon) and not forbidden[nxt]:
                        # 1. 基礎距離成本
                        dist_step = 1.414 if abs(d[0])+abs(d[1])==2 else 1.0
                        
                        # 2. 流場成本 (降低流場敏感度，增加 0.5 偏移)
                        u, v = u_now[nxt], v_now[nxt]
                        assist = np.dot([d[1], d[0]], [u, v])
                        flow_cost = dist_step * (1.0 - 0.4 * assist) 
                        
                        # 3. 平滑成本 (如果轉彎，增加額外成本，減少鋸齒)
                        turn_penalty = 0.5 if d != last_d and last_d != (0,0) else 0
                        
                        new_cost = cost[curr] + flow_cost + turn_penalty
                        if nxt not in cost or new_cost < cost[nxt]:
                            cost[nxt] = new_cost
                            # 啟發函數加重直線引導
                            h = np.hypot(nxt[0]-gi, nxt[1]-gj) * 1.2 
                            heapq.heappush(q, (new_cost + h, nxt, d))
                            came_from[nxt] = curr
            
            p = []
            c = (gi, gj)
            while c in came_from: p.append(c); c = came_from[c]
            st.session_state.path = p[::-1]

        if st.button("⏭️ 下一步 (模擬 1H 航行)", use_container_width=True):
            if st.session_state.path and st.session_state.sim_step < len(st.session_state.path)-1:
                st.session_state.sim_step += 1
                idx = st.session_state.path[st.session_state.sim_step]
                st.session_state.ship_pos = [lat[idx[0]], lon[idx[1]]]

        if st.button("🔄 重置系統", use_container_width=True):
            st.session_state.sim_step = 0
            st.session_state.ship_pos = [s_lat, s_lon]
            st.session_state.path = None
            st.rerun()

    # ===============================
    # 3. 儀表板
    # ===============================
    m1, m2, m3, m4 = st.columns(4)
    dist, brg = 0.0, 0
    if st.session_state.path:
        idx = st.session_state.sim_step
        p = st.session_state.path
        for k in range(idx, len(p)-1):
            dist += haversine(lat[p[k][0]], lon[p[k][1]], lat[p[k+1][0]], lon[p[k+1][1]])
        if idx < len(p)-1:
            brg = calc_bearing([lat[p[idx][0]], lon[p[idx][1]]], [lat[p[idx+1][0]], lon[p[idx+1][1]]])

    m1.metric("📏 剩餘航程", f"{dist:.1f} km")
    m2.metric("⏳ 預計航時", f"{dist/(knots*1.852):.1f} hr" if dist > 0 else "0.0 hr")
    m3.metric("🧭 建議航向", f"{brg:.0f}°")
    m4.metric("📡 衛星連線", f"{8 + st.session_state.sim_step % 3} Fix")

    # ===============================
    # 4. 地圖
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.5, 124.5, 20.5, 26.5])
    
    t_idx = min(st.session_state.sim_step // 6, 7)
    spd = np.sqrt(u_4d[t_idx]**2 + v_4d[t_idx]**2)
    ax.pcolormesh(lon, lat, spd, cmap='YlGnBu', alpha=0.4, shading='auto')
    
    ax.add_feature(cfeature.LAND, facecolor='#111111', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=3)
    
    if st.session_state.path:
        p = st.session_state.path
        ax.plot([lon[x[1]] for x in p], [lat[x[0]] for x in p], color='#FF00FF', lw=3, zorder=5, label='HELIOS 優化路徑')
        curr = st.session_state.ship_pos
        ax.scatter(curr[1], curr[0], color='#00FF00', s=200, edgecolors='white', zorder=10, label='目前位置')
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=350, zorder=10, label='目的地')

    ax.legend(loc='lower right', facecolor='#000000', labelcolor='white')
    st.pyplot(fig)
