import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

# ==================== 1️⃣ 設定 ====================
st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧導航控制台")

# ==================== 2️⃣ 側邊欄輸入 ====================
with st.sidebar:
    st.header("📍 航點座標輸入")
    st.info("請輸入經緯度，系統將自動定位至最近海域。")
    s_lon = st.number_input("起點經度 (118-124)", value=120.30, format="%.2f")
    s_lat = st.number_input("起點緯度 (20-26)", value=22.60, format="%.2f")
    e_lon = st.number_input("終點經度 (118-124)", value=122.00, format="%.2f")
    e_lat = st.number_input("終點緯度 (20-26)", value=24.50, format="%.2f")
    ship_speed = st.number_input("🚤 船速 (km/h)", value=20.0, step=1.0)
    run_nav = st.button("🚀 啟動衛星導航計算", use_container_width=True)

# ==================== 3️⃣ 讀取 HYCOM ====================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        lat_slice, lon_slice = slice(20, 26), slice(118, 124)
        u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        
        # 建立陸地遮罩
        land_mask = np.isnan(u_data.values)
        # 清洗數據：將 NaN 轉為 0 避免繪圖報錯
        u_clean = np.nan_to_num(u_data.values)
        v_clean = np.nan_to_num(v_data.values)
        
        return u_data['lon'].values, u_data['lat'].values, u_clean, v_clean, land_mask, latest_time
    except Exception as e:
        st.error(f"資料連線失敗: {e}")
        return None, None, None, None, None, None

lons, lats, u, v, land_mask, obs_time = load_hycom_data()

# ==================== 4️⃣ 最近海格點與安全距離 ====================
def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()
    if not land_mask[lat_idx, lon_idx]:
        return (lat_idx, lon_idx)
    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]] - lat)**2 + (lons[ocean[1]] - lon)**2)
    i = dist.argmin()
    return (ocean[0][i], ocean[1][i])

def compute_safety(land_mask):
    return distance_transform_edt(~land_mask)

if lons is not None:
    safety = compute_safety(land_mask)
    start_idx = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
    goal_idx = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)

    # ==================== 5️⃣ A* 導航 ====================
    def astar(start, goal, u, v, land_mask, safety):
        rows, cols = land_mask.shape
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        pq = []
        heapq.heappush(pq, (0, start))
        came_from = {}
        cost = {start: 0}
        visited = set()

        while pq:
            _, cur = heapq.heappop(pq)
            if cur in visited: continue
            visited.add(cur)
            if cur == goal: break
            
            for d in dirs:
                ni, nj = cur[0]+d[0], cur[1]+d[1]
                if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni,nj]:
                    # 海流助推計算
                    flow_assist = u[cur[0], cur[1]]*d[1] + v[cur[0], cur[1]]*d[0]
                    # 安全距離懲罰 (距離陸地越近懲罰越高)
                    land_penalty = 10.0 / (safety[ni,nj] + 1.0)
                    dist_step = np.sqrt(d[0]**2 + d[1]**2)
                    
                    new_cost = cost[cur] + dist_step + land_penalty - 0.4 * flow_assist
                    
                    if (ni,nj) not in cost or new_cost < cost[(ni,nj)]:
                        cost[(ni,nj)] = new_cost
                        priority = new_cost + np.sqrt((ni-goal[0])**2 + (nj-goal[1])**2)
                        heapq.heappush(pq, (priority, (ni,nj)))
                        came_from[(ni,nj)] = cur
        
        path = []
        curr = goal
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        if path: path.append(start)
        return path[::-1]

    path = astar(start_idx, goal_idx, u, v, land_mask, safety)

    # ==================== 6️⃣ 視覺化設定 ====================
    colors = ["#E5F0FF","#CCE0FF","#99C2FF","#66A3FF","#3385FF","#0066FF","#001433"]
    levels = np.linspace(0, 1.5, len(colors))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("flow_map", list(zip(levels/1.5, colors)))

    def plot_navigation(lons, lats, u, v, path, s_lon, s_lat, e_lon, e_lat):
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([118, 124, 20, 26], crs=ccrs.PlateCarree())
        
        # 繪製灰色陸地
        ax.add_feature(cfeature.LAND, facecolor='#444444', zorder=2)
        ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.5, zorder=3)
        
        # 清洗數據後計算速度，確保無 NaN
        speed = np.sqrt(u**2 + v**2)
        
        # 背景海流
        im = ax.pcolormesh(lons, lats, speed, cmap=custom_cmap, shading='auto', alpha=0.9, zorder=1)
        plt.colorbar(im, label='流速 (m/s)', shrink=0.5)
        
        # 海流箭頭 (抽樣)
        ax.quiver(lons[::3], lats[::3], u[::3, ::3], v[::3, ::3], 
                  color='white', alpha=0.3, scale=12, zorder=4)
        
        # 繪製導航路徑
        if path:
            p_lons = [lons[p[1]] for p in path]
            p_lats = [lats[p[0]] for p in path]
            ax.plot(p_lons, p_lats, color='#FF00FF', linewidth=3, label='HELIOS AI 航線', zorder=5)
            
        # 起點終點標註
        ax.scatter(s_lon, s_lat, color='lime', s=100, edgecolors='black', label='起點', zorder=6)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, edgecolors='black', label='終點', zorder=6)
        
        ax.legend(loc='upper left', frameon=True)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.2)
        plt.title(f"HELIOS V6 智慧海流導航系統\n最新觀測時間: {obs_time}", fontsize=10)
        st.pyplot(fig)

    plot_navigation(lons, lats, u, v, path, s_lon, s_lat, e_lon, e_lat)

    # ==================== 7️⃣ 儀表板 ====================
    # 實際路徑距離計算
    if path:
        total_dist = 0
        for i in range(len(path)-1):
            p1 = (lats[path[i][0]], lons[path[i][1]])
            p2 = (lats[path[i+1][0]], lons[path[i+1][1]])
            total_dist += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 111
    else:
        total_dist = np.sqrt((s_lat-e_lat)**2 + (s_lon-e_lon)**2) * 111

    estimated_hours = total_dist / ship_speed

    c1, c2, c3 = st.columns(3)
    c1.metric("⏱️ 預估航行時間", f"{estimated_hours:.1f} 小時")
    c2.metric("📏 總航行距離", f"{total_dist:.1f} km")
    c3.metric("🛰️ 系統狀態", "Active", delta="衛星鏈路正常")
