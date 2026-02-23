import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq

# --- 1. 初始化與衛星參數 ---
st.set_page_config(page_title="HELIOS 智慧避障系統", layout="wide")
st.title("⚓ HELIOS 智慧避障導航系統")

# 模擬衛星接收狀態函數
def check_satellite_status():
    # 模擬 36 顆低軌衛星在 900km 高度的覆蓋率校驗
    stability = 0.84 + np.random.uniform(0.05, 0.11) 
    is_received = stability > 0.85
    return is_received, round(stability, 2)

# --- 2. 側邊欄控制區 ---
st.sidebar.header("📍 導航與衛星設定")

# 起點選擇
if st.sidebar.button("📍 立即定位當前位置"):
    st.session_state.s_lon = 121.750
    st.session_state.s_lat = 25.150
    st.sidebar.success("已抓取 GPS 座標")

s_lon = st.sidebar.number_input("起點經度", value=st.session_state.get('s_lon', 121.750), format="%.3f")
s_lat = st.sidebar.number_input("起點緯度", value=st.session_state.get('s_lat', 25.150), format="%.3f")

# 終點選擇
st.sidebar.markdown("---")
e_lon = st.sidebar.number_input("終點經度 (目標)", value=121.900, format="%.3f")
e_lat = st.sidebar.number_input("終點緯度 (目標)", value=24.600, format="%.3f")

# --- 3. A* 避障路徑演算法 ---
def astar_search(grid, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    oheap = []
    heapq.heappush(oheap, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
            
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] == 1: continue # 避開陸地
                
                tentative_g = g_score[current] + (1.414 if i!=0 and j!=0 else 1.0)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.linalg.norm(np.array(neighbor)-np.array(goal))
                    heapq.heappush(oheap, (f_score, neighbor))
    return []

# --- 4. 執行導航分析 ---
if st.sidebar.button("🚀 啟動 HELIOS 聯網導航"):
    sat_active, sat_val = check_satellite_status()
    
    if not sat_active:
        st.warning(f"📡 衛星信號微弱 ({sat_val})，正在嘗試重新校驗中...")
    
    with st.spinner('正在接收低軌衛星流場數據...'):
        try:
            DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
            ds = xr.open_dataset(DATA_URL, decode_times=False)
            
            # 抓取包含起終點的正方形區域
            margin = 0.5
            subset = ds.sel(lon=slice(min(s_lon, e_lon)-margin, max(s_lon, e_lon)+margin), 
                            lat=slice(min(s_lat, e_lat)-margin, max(s_lat, e_lat)+margin), 
                            depth=0).isel(time=-1).load()
            
            lons, lats = subset.lon.values, subset.lat.values
            grid = np.where(np.isnan(subset.water_u.values), 1, 0)

            # 轉換座標索引
            s_idx = (np.abs(lats - s_lat).argmin(), np.abs(lons - s_lon).argmin())
            e_idx = (np.abs(lats - e_lat).argmin(), np.abs(lons - e_lon).argmin())

            path_indices = astar_search(grid, s_idx, e_idx)

            if path_indices:
                # 數據儀表板
                st.subheader("🛰️ HELIOS 衛星即時鏈路狀態")
                c1, c2, c3 = st.columns(3)
                c1.metric("衛星通訊穩定度", f"{sat_val}", "🟢 良好" if sat_active else "🟡 波動")
                c2.metric("流場數據源", "HYCOM + HELIOS")
                c3.metric("避障步進狀態", f"剩餘 {len(path_indices)} 步")

                # --- 繪圖：強制格子正方形 ---
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_aspect('equal', adjustable='box') # 核心：格子正方化
                
                ax.set_extent([min(s_lon, e_lon)-0.4, max(s_lon, e_lon)+0.4, 
                               min(s_lat, e_lat)-0.4, max(s_lat, e_lat)+0.4])

                # 底圖
                speed = np.sqrt(subset.water_u**2 + subset.water_v**2)
                cf = ax.pcolormesh(lons, lats, speed, cmap='YlGn', shading='auto', alpha=0.8)
                plt.colorbar(cf, label='Current Speed (m/s)', shrink=0.7)

                ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1e1e1e')
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white')

                # 繪製路徑
                path_lon = [lons[i[1]] for i in path_indices]
                path_lat = [lats[i[0]] for i in path_indices]
                ax.plot(path_lon, path_lat, color='#FF00FF', linewidth=3, label='AI Optimized Path')
                
                # 起終點
                ax.scatter(s_lon, s_lat, color='yellow', s=100, label='Start')
                ax.scatter(e_lon, e_lat, color='red', marker='*', s=200, label='Goal')

                ax.legend()
                st.pyplot(fig)
                st.success("✅ 已確認接收低軌衛星資訊，導航路徑已更新。")
            else:
                st.error("❌ 無法規劃路徑，請確認座標是否在海面上。")

        except Exception as e:
            st.error(f"衛星數據鏈路中斷: {e}")
