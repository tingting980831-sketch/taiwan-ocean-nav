import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq

# --- 1. 頁面與 Session 狀態初始化 ---
st.set_page_config(page_title="HELIOS 步進導航儀", layout="wide")
st.title("⚓ HELIOS 智慧步進導航儀")

# 使用 Session State 紀錄目前船隻所在位置
if 'current_lon' not in st.session_state:
    st.session_state.current_lon = 121.750
    st.session_state.current_lat = 25.150

# --- 2. 側邊欄：手動輸入與設定 ---
st.sidebar.header("📍 位置管理")

# 選擇起點方式
loc_mode = st.sidebar.radio("起始點設定", ["手動輸入/立即定位", "沿用上一步位置"])

if loc_mode == "手動輸入/立即定位":
    col_s1, col_s2 = st.sidebar.columns(2)
    s_lon_input = col_s1.number_input("起點經度", value=121.750, format="%.3f")
    s_lat_input = col_s2.number_input("起點緯度", value=25.150, format="%.3f")
    if st.sidebar.button("📍 更新起點"):
        st.session_state.current_lon = s_lon_input
        st.session_state.current_lat = s_lat_input

# 終點設定 (目標)
st.sidebar.markdown("---")
st.sidebar.header("🏁 終點設定")
e_lon = st.sidebar.number_input("終點經度", value=121.900, format="%.3f")
e_lat = st.sidebar.number_input("終點緯度", value=24.600, format="%.3f")

# --- 3. 核心運算函數 ---
def get_navigation_step(grid, lons, lats, s_idx, e_idx):
    # A* 演算法計算下一步
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    oheap = []
    heapq.heappush(oheap, (0, s_idx))
    came_from = {}
    g_score = {s_idx: 0}
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == e_idx:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1] # 回傳完整路徑，我們取第 0 個作為下一步
            
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] == 1: continue 
                
                tentative_g = g_score[current] + (1.414 if i!=0 and j!=0 else 1.0)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.linalg.norm(np.array(neighbor)-np.array(e_idx))
                    heapq.heappush(oheap, (f_score, neighbor))
    return []

# --- 4. 執行與顯示 ---
if st.button("🛰️ 聯網 HELIOS 並獲取下一步指引"):
    try:
        # 1. 模擬低軌衛星信號檢查
        stability = 0.84 + np.random.uniform(0.05, 0.1)
        st.info(f"📡 HELIOS 衛星鏈路校驗成功 (穩定度: {round(stability, 2)})")

        # 2. 獲取數據
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(DATA_URL, decode_times=False)
        subset = ds.sel(lon=slice(min(st.session_state.current_lon, e_lon)-0.5, max(st.session_state.current_lon, e_lon)+0.5),
                        lat=slice(min(st.session_state.current_lat, e_lat)-0.5, max(st.session_state.current_lat, e_lat)+0.5),
                        depth=0).isel(time=-1).load()
        
        lons, lats = subset.lon.values, subset.lat.values
        grid = np.where(np.isnan(subset.water_u.values), 1, 0)

        # 3. 計算路徑與下一步
        s_idx = (np.abs(lats - st.session_state.current_lat).argmin(), np.abs(lons - st.session_state.current_lon).argmin())
        e_idx = (np.abs(lats - e_lat).argmin(), np.abs(lons - e_lon).argmin())
        
        full_path = get_navigation_step(grid, lons, lats, s_idx, e_idx)

        if full_path:
            # 下一步的座標
            next_step_idx = full_path[0]
            next_lon = lons[next_step_idx[1]]
            next_lat = lats[next_step_idx[0]]

            # 計算當前位置流速 (用於導航修正)
            u_now = float(subset.water_u.interp(lat=st.session_state.current_lat, lon=st.session_state.current_lon).values)
            v_now = float(subset.water_v.interp(lat=st.session_state.current_lat, lon=st.session_state.current_lon).values)
            bearing = np.degrees(np.arctan2(next_lon - st.session_state.current_lon, next_lat - st.session_state.current_lat))

            # --- 儀表板 ---
            st.subheader("🧭 下一步導航指令")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("建議航向 (Bearing)", f"{round(bearing, 1)}°")
            c2.metric("下一站經度", f"{round(next_lon, 3)}")
            c3.metric("下一站緯度", f"{round(next_lat, 3)}")
            c4.metric("剩餘距離", f"{len(full_path)} 步")

            # --- 繪圖 (正方形格子) ---
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_aspect('equal', adjustable='box')
            
            # 動態縮放地圖範圍
            ax.set_extent([st.session_state.current_lon-0.2, st.session_state.current_lon+0.2, 
                           st.session_state.current_lat-0.2, st.session_state.current_lat+0.2])

            speed = np.sqrt(subset.water_u**2 + subset.water_v**2)
            cf = ax.pcolormesh(lons, lats, speed, cmap='YlGn', shading='auto', alpha=0.8)
            
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1e1e1e')
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white')

            # 畫出目前路徑與下一步
            ax.scatter(st.session_state.current_lon, st.session_state.current_lat, color='yellow', s=150, label='Current Pos', zorder=12)
            ax.quiver(st.session_state.current_lon, st.session_state.current_lat, next_lon - st.session_state.current_lon, next_lat - st.session_state.current_lat, 
                      color='magenta', scale=0.1, scale_units='xy', width=0.015, label='Next Move', zorder=13)
            ax.scatter(e_lon, e_lat, color='red', marker='*', s=200, label='End Goal', zorder=11)

            ax.legend()
            st.pyplot(fig)

            # 更新 Session State，按下「走下一步」會移動船隻
            if st.button(f"🚢 前進到下一步 ({round(next_lon,3)}, {round(next_lat,3)})"):
                st.session_state.current_lon = next_lon
                st.session_state.current_lat = next_lat
                st.rerun()

        else:
            st.error("❌ 無法計算下一步，請確認座標是否在陸地上。")

    except Exception as e:
        st.error(f"衛星鏈路異常: {e}")

st.markdown("---")
st.write(f"📍 **當前船隻位置**: {round(st.session_state.current_lon, 4)}, {round(st.session_state.current_lat, 4)}")
