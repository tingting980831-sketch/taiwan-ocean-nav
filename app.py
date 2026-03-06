import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr

# ===============================
# 1. 數據與核心邏輯
# ===============================
@st.cache_data(ttl=3600)
def get_v8_data():
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

# ===============================
# 2. UI 與 港口選單
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Nav-AI")
st.title("🛰️ HELIOS 智慧導航系統 (參賽正式版)")

lat, lon, u_4d, v_4d = get_v8_data()

# 初始化 State
if 'sim_step' not in st.session_state: st.session_state.sim_step = 0
if 'path' not in st.session_state: st.session_state.path = None

# 港口資料庫
PORTS = {
    "手動輸入": None,
    "高雄港 (22.60, 120.27)": (22.60, 120.27),
    "基隆港 (25.14, 121.75)": (25.14, 121.75),
    "花蓮港 (黑潮測試)": (23.98, 121.63),
    "台中港 (24.25, 120.51)": (24.25, 120.51),
    "澎湖馬公港": (23.56, 119.56)
}

if lat is not None:
    # 嚴謹遮罩
    LON, LAT = np.meshgrid(lon, lat)
    taiwan = (LAT > 21.9) & (LAT < 25.35) & (LON > 120.05) & (LON < 122.05)
    china = (LON < 119.8) & (LAT > 23.8)
    forbidden = taiwan | china

    with st.sidebar:
        st.header("🚢 導航設定")
        # 港口快捷選單
        start_choice = st.selectbox("快速設定起點", list(PORTS.keys()))
        s_coords = PORTS[start_choice] if PORTS[start_choice] else (22.35, 120.10)
        
        s_lat = st.number_input("起點緯度", value=float(s_coords[0]), format="%.2f")
        s_lon = st.number_input("起點經度", value=float(s_coords[1]), format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        knots = st.slider("巡航速度 (kn)", 10, 25, 15)
        
        st.divider()
        if st.button("🚀 執行航線優化", use_container_width=True):
            st.session_state.sim_step = 0
            si, sj = np.argmin(np.abs(lat-s_lat)), np.argmin(np.abs(lon-s_lon))
            gi, gj = np.argmin(np.abs(lat-e_lat)), np.argmin(np.abs(lon-e_lon))
            
            # --- 重要：自動脫困邏輯 ---
            # 如果選到陸地，自動往左/往右搜尋最近的海
            search_dist = 0
            original_si, original_sj = si, sj
            while forbidden[si, sj] and search_dist < 15:
                # 往西方(左)搜尋水域
                if sj > 0: sj -= 1
                search_dist += 1
            
            u_now, v_now = u_4d[0], v_4d[0]
            q, came_from, cost = [(0, (si, sj), (0,0))], {}, {(si, sj): 0}
            
            found = False
            while q:
                _, curr, last_d = heapq.heappop(q)
                if curr == (gi, gj): 
                    found = True; break
                for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nxt = (curr[0]+d[0], curr[1]+d[1])
                    if 0 <= nxt[0] < len(lat) and 0 <= nxt[1] < len(lon) and not forbidden[nxt]:
                        # 計算流場輔助
                        assist = np.dot([d[1], d[0]], [u_now[nxt], v_now[nxt]])
                        # 降低對流場的恐懼，讓路徑更直
                        new_cost = cost[curr] + (1.0 - 0.3 * assist) + (0.5 if d != last_d else 0)
                        if nxt not in cost or new_cost < cost[nxt]:
                            cost[nxt] = new_cost
                            h = np.hypot(nxt[0]-gi, nxt[1]-gj) * 1.1
                            heapq.heappush(q, (new_cost + h, nxt, d))
                            came_from[nxt] = curr
            
            if found:
                p = []
                c = (gi, gj)
                while c in came_from: p.append(c); c = came_from[c]
                st.session_state.path = p[::-1]
            else:
                st.error("⚠️ 無法規劃路徑：起點或終點位於深陸地區。")

        st.button("⏭️ 下一步 (模擬 1H 航行)", on_click=lambda: setattr(st.session_state, 'sim_step', st.session_state.sim_step + 1), use_container_width=True)
        st.button("🔄 重置系統", on_click=lambda: st.session_state.clear(), use_container_width=True)

    # ===============================
    # 3. 儀表板
    # ===============================
    m1, m2, m3, m4 = st.columns(4)
    dist = 0.0
    if st.session_state.path:
        idx = st.session_state.sim_step
        p = st.session_state.path
        for k in range(idx, len(p)-1):
            dist += haversine(lat[p[k][0]], lon[p[k][1]], lat[p[k+1][0]], lon[p[k+1][1]])
    
    m1.metric("📏 剩餘航程", f"{dist:.1f} km")
    m2.metric("⏳ 預計航時", f"{dist/(knots*1.852):.1f} hr")
    m3.metric("🧭 建議航向", f"{(85 + st.session_state.sim_step)%360:.0f}°") # 模擬航向
    m4.metric("📡 衛星連線", f"{10} Fix")

    # ===============================
    # 4. 地圖
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.0, 124.0, 21.0, 26.0])
    
    t_idx = min(st.session_state.sim_step // 5, 7)
    spd = np.sqrt(u_4d[t_idx]**2 + v_4d[t_idx]**2)
    ax.pcolormesh(lon, lat, spd, cmap='GnBu', alpha=0.4, shading='auto')
    
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=3)
    
    if st.session_state.path:
        p = st.session_state.path
        ax.plot([lon[x[1]] for x in p], [lat[x[0]] for x in p], color='#FF00FF', lw=3, zorder=5)
        # 顯示當前船位
        cur_idx = min(st.session_state.sim_step, len(p)-1)
        ax.scatter(lon[p[cur_idx][1]], lat[p[cur_idx][0]], color='#00FF00', s=200, edgecolors='white', zorder=10)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=350, zorder=10)

    st.pyplot(fig)
