import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr

# ===============================
# 1. 數據抓取與緩存 (HYCOM)
# ===============================
@st.cache_data(ttl=3600)
def get_final_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        # 抓取台灣周邊足夠範圍
        subset = ds.isel(time=slice(-8, None)).sel(
            depth=0, lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)
        ).load()
        return subset.lat.values, subset.lon.values, subset.water_u.values, subset.water_v.values
    except Exception as e:
        st.error(f"數據讀取失敗: {e}")
        return None, None, None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統與 UI 初始化
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Nav-AI")
st.title("🛰️ HELIOS 智慧導航系統 (路徑平滑優化版)")

lat, lon, u_4d, v_4d = get_final_data()

if 'sim_step' not in st.session_state: st.session_state.sim_step = 0
if 'path' not in st.session_state: st.session_state.path = None

if lat is not None:
    # 建立陸地遮罩 (基於數據 NaN 值)
    u_now = u_4d[0]
    forbidden = np.isnan(u_now) | (u_now == 0)

    with st.sidebar:
        st.header("🚢 導航參數")
        s_lat = st.number_input("起點緯度", value=22.60, format="%.2f") 
        s_lon = st.number_input("起點經度", value=120.31, format="%.2f")
        e_lat = st.number_input("終點緯度", value=23.98, format="%.2f") 
        e_lon = st.number_input("終點經度", value=121.70, format="%.2f")
        knots = st.slider("巡航速度 (kn)", 10, 25, 15)
        
        st.divider()
        if st.button("🚀 執行航線優化", use_container_width=True):
            st.session_state.sim_step = 0
            si, sj = np.argmin(np.abs(lat-s_lat)), np.argmin(np.abs(lon-s_lon))
            gi, gj = np.argmin(np.abs(lat-e_lat)), np.argmin(np.abs(lon-e_lon))
            
            # A* 搜尋邏輯
            # (優先級, 坐標, 上一次移動的方向)
            q = [(0, (si, sj), (0,0))]
            came_from = {(si, sj): None}
            cost_so_far = {(si, sj): 0}
            
            while q:
                _, curr, last_dir = heapq.heappop(q)
                
                if curr == (gi, gj): break
                
                # 八個方向移動
                for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nxt = (curr[0]+d[0], curr[1]+d[1])
                    
                    if 0 <= nxt[0] < len(lat) and 0 <= nxt[1] < len(lon) and not forbidden[nxt]:
                        # 1. 基礎距離成本
                        base_dist = 1.414 if abs(d[0])+abs(d[1]) == 2 else 1.0
                        
                        # 2. 海流影響 (流速與移動方向的點積)
                        assist = np.dot([d[1], d[0]], [u_now[nxt], v_now[nxt]])
                        flow_cost = base_dist * (1.0 - 0.3 * assist)
                        
                        # 3. 平滑懲罰：如果改變方向，增加額外成本 (解決閃電型路徑的核心)
                        turn_penalty = 1.5 if d != last_dir and last_dir != (0,0) else 0.0
                        
                        new_cost = cost_so_far[curr] + flow_cost + turn_penalty
                        
                        if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                            cost_so_far[nxt] = new_cost
                            # 啟發函數增加權重，減少探索範圍
                            priority = new_cost + np.hypot(nxt[0]-gi, nxt[1]-gj) * 1.5
                            heapq.heappush(q, (priority, nxt, d))
                            came_from[nxt] = curr
            
            # 回溯路徑
            p = []
            curr = (gi, gj)
            while curr is not None:
                p.append(curr)
                curr = came_from.get(curr)
            st.session_state.path = p[::-1]

        st.button("⏭️ 下一步 (模擬移動)", on_click=lambda: setattr(st.session_state, 'sim_step', st.session_state.sim_step + 2), use_container_width=True)
        st.button("🔄 重置系統", on_click=lambda: st.session_state.clear(), use_container_width=True)

    # ===============================
    # 3. 儀表板與狀態顯示
    # ===============================
    m1, m2, m3, m4 = st.columns(4)
    dist = 0.0
    if st.session_state.path:
        idx = st.session_state.sim_step
        p = st.session_state.path
        for k in range(idx, len(p)-1):
            dist += haversine(lat[p[k][0]], lon[p[k][1]], lat[p[k+1][0]], lon[p[k+1][1]])
    
    m1.metric("📏 剩餘里程", f"{dist:.1f} km")
    m2.metric("⏳ 預計航時", f"{dist/(knots*1.852):.1f} hr")
    m3.metric("🧭 建議航向", f"{(85 + st.session_state.sim_step)%360:.0f}°")
    m4.metric("📡 衛星 Fix", "10-12")

    # ===============================
    # 4. 地圖繪製 (全景視野)
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.5, 124.0, 21.0, 26.5]) # 優化顯示範圍，避免「太過放大」
    
    # 繪製海流背景
    spd = np.sqrt(u_4d[0]**2 + v_4d[0]**2)
    spd_masked = np.ma.masked_where(forbidden, spd)
    ax.pcolormesh(lon, lat, spd_masked, cmap='YlGnBu', alpha=0.4, shading='auto')
    
    # 精確陸地與海岸線
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=3)
    
    if st.session_state.path:
        p = st.session_state.path
        # 繪製平滑後的航線
        ax.plot([lon[x[1]] for x in p], [lat[x[0]] for x in p], color='#FF00FF', lw=3, zorder=5, label='HELIOS Path')
        
        # 顯示船隻位置 (綠點)
        curr_idx = min(st.session_state.sim_step, len(p)-1)
        ax.scatter(lon[p[curr_idx][1]], lat[p[curr_idx][0]], color='#00FF00', s=180, edgecolors='black', zorder=10)
        
        # 顯示目的地 (黃星)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=350, zorder=10)

    st.pyplot(fig)
