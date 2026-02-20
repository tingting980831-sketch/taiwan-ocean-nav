import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

# 網頁 UI 設定 (Streamlit 渲染中文沒問題)
st.set_page_config(page_title="AI 海象導航系統", layout="wide")
st.title("⚓ 智慧避障導航系統")
st.write("這套系統結合了 A* 演算法與 HYCOM 全球海象即時數據。")

# 側邊欄輸入
st.sidebar.header("📍 座標輸入")
s_lon = st.sidebar.number_input("起點經度 (Start Lon)", value=121.750, format="%.3f")
s_lat = st.sidebar.number_input("起點緯度 (Start Lat)", value=25.150, format="%.3f")
e_lon = st.sidebar.number_input("終點經度 (End Lon)", value=121.900, format="%.3f")
e_lat = st.sidebar.number_input("終點緯度 (End Lat)", value=24.600, format="%.3f")

# 核心 A* 演算法
def astar_search(grid, safety_map, start, goal):
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
                if grid[neighbor[0], neighbor[1]] == 1: continue 
                dist = 1.414 if i != 0 and j != 0 else 1.0
                safety_cost = safety_map[neighbor[0], neighbor[1]] * 1.5
                tentative_g = g_score[current] + dist + safety_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.linalg.norm(np.array(neighbor)-np.array(goal))
                    heapq.heappush(oheap, (f_score, neighbor))
    return []

if st.sidebar.button("🚀 開始規劃航線"):
    with st.spinner('正在從 HYCOM 獲取數據...'):
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        try:
            ds = xr.open_dataset(DATA_URL, decode_times=False, engine='netcdf4')
            margin = 1.0
            subset = ds.sel(lon=slice(min(s_lon, e_lon)-margin, max(s_lon, e_lon)+margin), 
                            lat=slice(min(s_lat, e_lat)-margin, max(s_lat, e_lat)+margin), 
                            depth=0).isel(time=-1).load()
            
            lons, lats = subset.lon.values, subset.lat.values
            grid = np.where(np.isnan(subset.water_u.values), 1, 0)
            dist_from_land = distance_transform_edt(1 - grid)
            safety_map = np.exp(-dist_from_land / 0.5)

            # 座標索引
            iy_s, ix_s = np.abs(lats - s_lat).argmin(), np.abs(lons - s_lon).argmin()
            iy_e, ix_e = np.abs(lats - e_lat).argmin(), np.abs(lons - e_lon).argmin()
            
            # 起點入海補償
            def get_water_idx(iy, ix):
                if grid[iy, ix] == 0: return (iy, ix)
                y_idx, x_idx = np.indices(grid.shape)
                dists = np.sqrt((y_idx - iy)**2 + (x_idx - ix)**2)
                dists[grid == 1] = 1e9
                return np.unravel_index(np.argmin(dists), grid.shape)

            s_idx, e_idx = get_water_idx(iy_s, ix_s), get_water_idx(iy_e, ix_e)
            path_indices = astar_search(grid, safety_map, s_idx, e_idx)

            if path_indices:
                path_lon = [s_lon] + [lons[i[1]] for i in path_indices] + [e_lon]
                path_lat = [s_lat] + [lats[i[0]] for i in path_indices] + [e_lat]

                # --- 繪圖 (全部使用英文以解決中文亂碼問題) ---
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent([min(path_lon)-0.4, max(path_lon)+0.4, min(path_lat)-0.4, max(path_lat)+0.4])
                
                speed = np.sqrt(subset.water_u**2 + subset.water_v**2)
                cf = ax.pcolormesh(lons, lats, speed, cmap='viridis', shading='auto', alpha=0.7)
                
                # 英文圖表標籤
                cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.03, shrink=0.6)
                cbar.set_label('Current Speed (m/s)')
                
                ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#222222')
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white')
                
                # 英文圖例
                ax.plot(path_lon, path_lat, color='magenta', linewidth=3, label='AI Path')
                ax.scatter(s_lon, s_lat, color='yellow', s=100, label='Start')
                ax.scatter(e_lon, e_lat, color='red', marker='*', s=200, label='Goal')
                
                ax.set_title("AI Marine Navigation & Obstacle Avoidance", fontsize=14)
                ax.legend(loc='lower right')
                
                # 網頁顯示
                st.pyplot(fig)
                st.success(f"規劃完成 (Success)! 航線經緯度已精確對齊。")
            else:
                st.error("找不到路徑 (Path not found).")
        except Exception as e:
            st.error(f"Error: {e}")
