import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr

# 1. 數據抓取
@st.cache_data(ttl=3600)
def get_v9_2_data():
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

# 2. 系統初始化
st.set_page_config(layout="wide", page_title="HELIOS Nav-AI")
st.title("🛰️ HELIOS 智慧導航系統 (穩定運行版)")

lat, lon, u_4d, v_4d = get_v9_2_data()

if 'sim_step' not in st.session_state: st.session_state.sim_step = 0
if 'path' not in st.session_state: st.session_state.path = None

if lat is not None:
    # 定義當前流場變數 (這兩行是解決 NameError 的關鍵)
    u_now = u_4d[0]
    v_now = v_4d[0]
    forbidden = np.isnan(u_now) | (u_now == 0)

    with st.sidebar:
        st.header("🚢 導航設定")
        s_lat = st.number_input("起點緯度", value=22.60) 
        s_lon = st.number_input("起點經度", value=120.31)
        e_lat = st.number_input("終點緯度", value=23.98) 
        e_lon = st.number_input("終點經度", value=121.70)
        
        if st.button("🚀 執行航線優化", use_container_width=True):
            st.session_state.sim_step = 0
            si, sj = np.argmin(np.abs(lat-s_lat)), np.argmin(np.abs(lon-s_lon))
            gi, gj = np.argmin(np.abs(lat-e_lat)), np.argmin(np.abs(lon-e_lon))
            
            # (優先級, 坐標, 方向)
            q = [(0, (si, sj), (0,0))]
            came_from, cost_so_far = {(si, sj): None}, {(si, sj): 0}
            
            while q:
                _, curr, last_dir = heapq.heappop(q)
                if curr == (gi, gj): break
                
                for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nxt = (curr[0]+d[0], curr[1]+d[1])
                    if 0 <= nxt[0] < len(lat) and 0 <= nxt[1] < len(lon) and not forbidden[nxt]:
                        # 計算流場輔助 (確保 v_now 已定義)
                        assist = np.dot([d[1], d[0]], [u_now[nxt], v_now[nxt]])
                        # 增加轉彎懲罰以消除「閃電型」路徑
                        turn_penalty = 1.5 if d != last_dir and last_dir != (0,0) else 0
                        new_cost = cost_so_far[curr] + (1.0 - 0.3 * assist) + turn_penalty
                        
                        if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                            cost_so_far[nxt] = new_cost
                            h = np.hypot(nxt[0]-gi, nxt[1]-gj) * 1.5
                            heapq.heappush(q, (new_cost + h, nxt, d))
                            came_from[nxt] = curr
            
            p = []
            c = (gi, gj)
            while c is not None:
                p.append(c); c = came_from.get(c)
            st.session_state.path = p[::-1]

        st.button("🔄 重置系統", on_click=lambda: st.session_state.clear(), use_container_width=True)

    # 4. 地圖顯示
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.5, 124.0, 21.0, 26.5])
    
    # 背景海流
    spd = np.sqrt(u_now**2 + v_now**2)
    ax.pcolormesh(lon, lat, np.ma.masked_where(forbidden, spd), cmap='YlGnBu', alpha=0.4)
    ax.add_feature(cfeature.LAND, facecolor='#151515')
    ax.add_feature(cfeature.COASTLINE, edgecolor='white')
    
    if st.session_state.path:
        p = st.session_state.path
        ax.plot([lon[x[1]] for x in p], [lat[x[0]] for x in p], color='#FF00FF', lw=3)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=300, zorder=10)

    st.pyplot(fig)
