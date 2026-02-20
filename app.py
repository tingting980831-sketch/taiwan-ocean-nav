import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

# 網頁設定
st.set_page_config(page_title="AI 海象導航系統", layout="wide")
st.title("⚓ 全台通用：AI 避障導航系統")
st.markdown("---")

# 側邊欄輸入
st.sidebar.header("📍 航線座標設定")
s_lon = st.sidebar.number_input("起點經度 (Lon)", value=121.7500, format="%.4f")
s_lat = st.sidebar.number_input("起點緯度 (Lat)", value=25.1500, format="%.4f")
e_lon = st.sidebar.number_input("終點經度 (Lon)", value=121.9000, format="%.4f")
e_lat = st.sidebar.number_input("終點緯度 (Lat)", value=24.6000, format="%.4f")

# A* 演算法
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
                safety_cost = safety_map[neighbor[0], neighbor[1]] * 1.2
                tentative_g = g_score[current] + dist + safety_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.linalg.norm(np.array(neighbor)-np.array(goal))
                    heapq.heappush(oheap, (f_score, neighbor))
    return []

if st.sidebar.button("🚀 執行路徑規劃"):
    with st.spinner('正在從 HYCOM 數據中心獲取即時海象...'):
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        try:
            # 讀取數據
            ds = xr.open_dataset(DATA_URL, decode_times=False, engine='netcdf4')
            margin = 1.0
            subset = ds.sel(lon=slice(min(s_lon, e_lon)-margin, max(s_lon, e_lon)+margin), 
                            lat=slice(min(s_lat, e_lat)-margin, max(s_lat, e_lat)+margin), 
                            depth=0).isel(time=-1).load()
            
            lons, lats = subset.lon.values, subset.lat.values
            grid = np.where(np.isnan(subset.water_u.values), 1, 0)
            
            # 安全場
            dist_from_land = distance_transform_edt(1 - grid)
            safety_map = np.exp(-dist_from_land / 0.5)

            # 座標轉換
            iy_s, ix_s = np.abs(lats - s_lat).argmin(), np.abs(lons - s_lon).argmin()
            iy_e, ix_e = np.abs(lats - e_lat).argmin(), np.abs(lons - e_lon).argmin()
            
            # 起點校正邏輯 (確保從水域出發)
            def get_water_idx(iy, ix):
                if grid[iy, ix] == 0: return (iy, ix)
                y_idx, x_idx = np.indices(grid.shape)
                dists = np.sqrt((y_idx - iy)**2 + (x_idx - ix)**2)
                dists[grid == 1] = 1e9
                return np.unravel_index(np.argmin(dists), grid.shape)

            s_idx, e_idx = get_water_idx(iy_s, ix_s), get_water_idx(iy_e, ix_e)
            
            # 搜尋
            path_indices = astar_search(grid, safety_map, s_idx, e_idx)

            if path_indices:
                path_lon = [s_lon] + [lons[i[1]] for i in path_indices] + [e_lon]
                path_lat = [s_lat] + [lats[i[0]] for i in path_indices] + [e_lat]

                # 繪圖
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent([min(path_lon)-0.4, max(path_lon)+0.4, min(path_lat)-0.4, max(path_lat)+0.4])
                
                speed = np.sqrt(subset.water_u**2 + subset.water_v**2)
                cf = ax.pcolormesh(lons, lats, speed, cmap='viridis', shading='auto', alpha=0.7)
                plt.colorbar(cf, ax=ax, label='海流速度 (m/s)', shrink=0.6)
                
                ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#222222')
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white')
                
                ax.plot(path_lon, path_lat, color='magenta', linewidth=3, label='AI 最佳避障路徑')
                ax.scatter([s_lon, e_lon], [s_lat, e_lat], c=['yellow', 'red'], s=[60, 150], zorder=5)
                
                st.pyplot(fig)
                st.success(f"規劃成功！預計路徑點數：{len(path_lon)}")
            else:
                st.error("找不到路徑，可能是地形太過狹窄，請嘗試更換座標。")
        except Exception as e:
            st.error(f"系統錯誤: {e}")
