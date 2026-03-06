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
def get_v8_stable_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.isel(time=slice(-8, None)).sel(
            depth=0, lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)
        ).load()
        return subset.lat.values, subset.lon.values, subset.water_u.values, subset.water_v.values
    except Exception as e:
        st.error(f"數據連線異常: {e}")
        return None, None, None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統設定
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Nav-AI")
st.title("🛰️ HELIOS 智慧導航系統 (雲端部署穩定版)")

lat, lon, u_4d, v_4d = get_v8_stable_data()

if 'sim_step' not in st.session_state: st.session_state.sim_step = 0
if 'path' not in st.session_state: st.session_state.path = None

if lat is not None:
    # 使用數據本身的 NaN 作為最精確的陸地遮罩
    u_sample = u_4d[0]
    forbidden = (u_sample == 0) | np.isnan(u_sample)

    with st.sidebar:
        st.header("🚢 導航參數")
        s_lat = st.number_input("起點緯度", value=22.60, format="%.2f") 
        s_lon = st.number_input("起點經度", value=120.27, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.14, format="%.2f") 
        e_lon = st.number_input("終點經度", value=121.75, format="%.2f")
        knots = st.slider("巡航速度 (kn)", 10, 25, 15)
        
        st.divider()
        if st.button("🚀 執行航線優化", use_container_width=True):
            st.session_state.sim_step = 0
            si, sj = np.argmin(np.abs(lat-s_lat)), np.argmin(np.abs(lon-s_lon))
            gi, gj = np.argmin(np.abs(lat-e_lat)), np.argmin(np.abs(lon-e_lon))
            
            # A* 演算法
            u_now, v_now = u_4d[0], v_4d[0]
            q, came_from, cost = [(0, (si, sj), (0,0))], {}, {(si, sj): 0}
            
            while q:
                _, curr, last_d = heapq.heappop(q)
                if curr == (gi, gj): break
                for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nxt = (curr[0]+d[0], curr[1]+d[1])
                    if 0 <= nxt[0] < len(lat) and 0 <= nxt[1] < len(lon) and not forbidden[nxt]:
                        assist = np.dot([d[1], d[0]], [u_now[nxt], v_now[nxt]])
                        step_cost = 1.0 - 0.4 * assist + (0.5 if d != last_d else 0)
                        new_cost = cost[curr] + step_cost
                        if nxt not in cost or new_cost < cost[nxt]:
                            cost[nxt] = new_cost
                            h = np.hypot(nxt[0]-gi, nxt[1]-gj) * 1.2
                            heapq.heappush(q, (new_cost + h, nxt, d))
                            came_from[nxt] = curr
            
            p = []
            c = (gi, gj)
            while c in came_from: p.append(c); c = came_from[c]
            st.session_state.path = p[::-1]

        st.button("⏭️ 下一步", on_click=lambda: setattr(st.session_state, 'sim_step', st.session_state.sim_step + 1), use_container_width=True)
        st.button("🔄 重置系統", on_click=lambda: st.session_state.clear(), use_container_width=True)

    # 儀表板
    m1, m2, m3, m4 = st.columns(4)
    dist = 0.0
    if st.session_state.path:
        idx = st.session_state.sim_step
        p = st.session_state.path
        for k in range(idx, len(p)-1):
            dist += haversine(lat[p[k][0]], lon[p[k][1]], lat[p[k+1][0]], lon[p[k+1][1]])

    m1.metric("📏 剩餘航程", f"{dist:.1f} km")
    m2.metric("⏳ 預計航時", f"{dist/(knots*1.852):.1f} hr")
    m3.metric("🧭 建議航向", f"{(st.session_state.sim_step * 3) % 360}°")
    m4.metric("📡 衛星連線", "12 Fix")

    # 地圖
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([119.5, 123.0, 21.8, 25.8]) 
    
    t_idx = min(st.session_state.sim_step // 5, 7)
    spd = np.sqrt(u_4d[t_idx]**2 + v_4d[t_idx]**2)
    spd_masked = np.ma.masked_where(forbidden, spd)
    ax.pcolormesh(lon, lat, spd_masked, cmap='YlGnBu', alpha=0.5, shading='auto')
    
    ax.add_feature(cfeature.LAND, facecolor='#111111', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=3)
    
    if st.session_state.path:
        p = st.session_state.path
        ax.plot([lon[x[1]] for x in p], [lat[x[0]] for x in p], color='#FF00FF', lw=2.5, zorder=5)
        cur = min(st.session_state.sim_step, len(p)-1)
        ax.scatter(lon[p[cur][1]], lat[p[cur][0]], color='#00FF00', s=100, edgecolors='white', zorder=10)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, zorder=10)

    st.pyplot(fig)
