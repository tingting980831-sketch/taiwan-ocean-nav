import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq
import time
import xarray as xr
import pandas as pd

# ==========================================
# 1. Streamlit 頁面配置
# ==========================================
st.set_page_config(page_title="⚓ AI Maritime Navigator", layout="wide")
st.title("⚓ AI Maritime Navigation System")
st.markdown("### Real-time HYCOM Data & A* Route Optimization")

# ==========================================
# 2. 核心數據功能：連線 HYCOM 伺服器
# ==========================================
@st.cache_data(ttl=3600)
def get_hycom_realtime_data():
    # 目前 2026 年最穩定的 HYCOM 實時數據網址
    tds_url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    
    try:
        # 使用 decode_times=False 避開時間格式錯誤
        ds = xr.open_dataset(tds_url, decode_times=False)
        
        # 抓取最後一筆預報 (Latest) 並切換至台灣海域範圍
        subset = ds.isel(time=-1).sel(
            depth=0,
            lon=slice(118, 124),
            lat=slice(20, 27)
        ).load()
        
        # 嘗試解碼時間字串以供顯示
        try:
            decoded_ds = xr.decode_cf(subset)
            data_ts = pd.to_datetime(decoded_ds.time.values).strftime('%Y-%m-%d %H:%M')
        except:
            data_ts = "Latest Available Forecast (Manual Sync)"

        # 轉換為 NumPy 格式
        lat_arr = subset.lat.values
        lon_arr = subset.lon.values
        u_arr = np.nan_to_num(subset.water_u.values)
        v_arr = np.nan_to_num(subset.water_v.values)
        
        return lat_arr, lon_arr, u_arr, v_arr, data_ts
    
    except Exception as e:
        st.error(f"⚠️ Connection to HYCOM failed: {e}")
        return None, None, None, None, None

# 執行資料抓取
lat, lon, u, v, data_time = get_hycom_realtime_data()

if lat is not None:
    st.sidebar.success(f"✅ Data Loaded: {data_time}")
    
    # 建立網格與陸地遮罩
    LON, LAT = np.meshgrid(lon, lat)
    
    # 台灣陸地定義 (橢圓形近似 + 緩衝區)
    center_lat, center_lon = 23.7, 121.0
    a, b = 1.6, 0.8
    land_mask = (((LAT - center_lat) / a) ** 2 + ((LON - center_lon) / b) ** 2) < 1
    
    # 外擴緩衝區
    grid_res = lat[1] - lat[0]
    buffer_cells = int(0.05 / grid_res) 
    dist_map = distance_transform_edt(~land_mask)
    forbidden = land_mask | (dist_map <= buffer_cells)

    # ==========================================
    # 3. A* 搜尋演算法邏輯
    # ==========================================
    def heuristic(p1, p2):
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def flow_cost(i, j, ni, nj):
        dx = lon[nj] - lon[j]
        dy = lat[ni] - lat[i]
        move_vec = np.array([dx, dy])
        norm = np.linalg.norm(move_vec)
        move_vec = move_vec / (norm + 1e-6)
        flow_vec = np.array([u[i,j], v[i,j]])
        assist = np.dot(move_vec, flow_vec)
        # 基本物理距離成本 * 海流修正因子 (順流成本降低)
        return norm * (1 - 0.7 * assist)

    def astar(start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for d in neighbors:
                ni, nj = current[0] + d[0], current[1] + d[1]
                if ni < 0 or nj < 0 or ni >= len(lat) or nj >= len(lon): continue
                if forbidden[ni, nj]: continue
                tentative_g = g_score[current] + flow_cost(current[0], current[1], ni, nj)
                neighbor = (ni, nj)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return None

    # ==========================================
    # 4. UI 互動與參數輸入
    # ==========================================
    with st.sidebar:
        st.header("Navigation Parameters")
        s_lat = st.number_input("Start Lat", value=22.3, step=0.1, format="%.2f")
        s_lon = st.number_input("Start Lon", value=120.0, step=0.1, format="%.2f")
        e_lat = st.number_input("End Lat", value=25.2, step=0.1, format="%.2f")
        e_lon = st.number_input("End Lon", value=122.0, step=0.1, format="%.2f")
        
        # 修正此處的參數錯誤
        run_btn = st.button("🚀 Calculate Optimized Route", use_container_width=True)

    def get_idx(la, lo):
        return np.argmin(np.abs(lat - la)), np.argmin(np.abs(lon - lo))

    start_idx = get_idx(s_lat, s_lon)
    goal_idx = get_idx(e_lat, e_lon)

    # ==========================================
    # 5. 計算與視覺化
    # ==========================================
    if run_btn:
        t_start = time.time()
        path = astar(start_idx, goal_idx)
        t_end = time.time()

        if path:
            st.success(f"Route Found! Optimization Time: {t_end - t_start:.2f}s")
            fig = plt.figure(figsize=(12, 9))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([118.5, 123.5, 21.0, 26.5])
            
            ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', zorder=1)
            ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.8, zorder=2)
            
            # 繪製海流
            ax.quiver(LON[::6, ::6], LAT[::6, ::6], u[::6, ::6], v[::6, ::6], 
                      color='cyan', alpha=0.35, scale=20, zorder=3)

            # 繪製優化路徑
            py = [lat[p[0]] for p in path]
            px = [lon[p[1]] for p in path]
            ax.plot(px, py, color='#FF00FF', linewidth=3, label='AI Optimized Path', zorder=5)
            ax.plot([s_lon, e_lon], [s_lat, e_lat], color='white', linestyle='--', alpha=0.4, label='Direct Route')

            ax.scatter(s_lon, s_lat, color='lime', s=100, label='Start', zorder=6)
            ax.scatter(e_lon, e_lat, color='yellow', s=100, label='End', zorder=6)
            ax.legend(loc='lower right')
            
            st.pyplot(fig)
        else:
            st.error("No valid route found. Check if coordinates are in the ocean.")
