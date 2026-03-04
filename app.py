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
st.markdown("### Real-time Ocean Current Speed & A* Optimization")

# ==========================================
# 2. 數據功能：獲取 HYCOM 數據並計算流速強度
# ==========================================
@st.cache_data(ttl=3600)
def get_hycom_navigation_data():
    tds_url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        # 使用 decode_times=False 避開時間解碼報錯
        ds = xr.open_dataset(tds_url, decode_times=False)
        
        # 抓取最新海表面預報並載入台灣範圍
        subset = ds.isel(time=-1).sel(
            depth=0,
            lon=slice(118, 124),
            lat=slice(20, 27)
        ).load()
        
        # 轉換數據與處理陸地 NaN
        u_arr = np.nan_to_num(subset.water_u.values)
        v_arr = np.nan_to_num(subset.water_v.values)
        
        # 計算流速強度 (Speed)
        speed_arr = np.sqrt(u_arr**2 + v_arr**2)
        
        try:
            decoded_ds = xr.decode_cf(subset)
            data_ts = pd.to_datetime(decoded_ds.time.values).strftime('%Y-%m-%d %H:%M')
        except:
            data_ts = "Latest Forecast"

        return subset.lat.values, subset.lon.values, u_arr, v_arr, speed_arr, data_ts
    except Exception as e:
        st.error(f"⚠️ HYCOM Connection Error: {e}")
        return None, None, None, None, None, None

# 執行資料抓取
lat, lon, u, v, speed, data_time = get_hycom_navigation_data()

if lat is not None:
    st.sidebar.success(f"✅ Live Data: {data_time}")
    
    # 建立網格與安全禁區 (維持原有 A* 邏輯)
    LON, LAT = np.meshgrid(lon, lat)
    center_lat, center_lon = 23.7, 121.0
    a, b = 1.6, 0.8
    land_mask = (((LAT - center_lat) / a) ** 2 + ((LON - center_lon) / b) ** 2) < 1
    grid_res = lat[1] - lat[0]
    forbidden = land_mask | (distance_transform_edt(~land_mask) <= int(0.05 / grid_res))

    # ==========================================
    # 3. A* 演算法邏輯
    # ==========================================
    def heuristic(p1, p2): return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def flow_cost(i, j, ni, nj):
        dx, dy = lon[nj] - lon[j], lat[ni] - lat[i]
        norm = np.hypot(dx, dy)
        move_vec = np.array([dx, dy]) / (norm + 1e-6)
        flow_vec = np.array([u[i,j], v[i,j]])
        assist = np.dot(move_vec, flow_vec)
        return norm * (1 - 0.7 * assist)

    def astar(start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from, g_score = {}, {start: 0}
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current); current = came_from[current]
                path.append(start); return path[::-1]
            for d in neighbors:
                ni, nj = current[0] + d[0], current[1] + d[1]
                if ni < 0 or nj < 0 or ni >= len(lat) or nj >= len(lon) or forbidden[ni, nj]: continue
                tg = g_score[current] + flow_cost(current[0], current[1], ni, nj)
                if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                    came_from[(ni, nj)] = current
                    g_score[(ni, nj)] = tg
                    heapq.heappush(open_set, (tg + heuristic((ni, nj), goal), (ni, nj)))
        return None

    # ==========================================
    # 4. UI 互動介面
    # ==========================================
    with st.sidebar:
        st.header("Parameters")
        s_lat = st.number_input("Start Lat", value=22.3, step=0.1, format="%.2f")
        s_lon = st.number_input("Start Lon", value=120.0, step=0.1, format="%.2f")
        e_lat = st.number_input("End Lat", value=25.2, step=0.1, format="%.2f")
        e_lon = st.number_input("End Lon", value=122.0, step=0.1, format="%.2f")
        run_btn = st.button("🚀 Calculate Optimized Route", use_container_width=True)

    # ==========================================
    # 5. 視覺化 (整合綠色色階背景)
    # ==========================================
    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118.5, 123.5, 21.0, 26.5])
    
    # 【關鍵更新】使用 YlGn 綠色色階顯示流速強度
    v_min, v_max = 0, 1.2 # 流速範圍
    mesh = ax.pcolormesh(lon, lat, speed, cmap='YlGn', shading='auto', alpha=0.8, vmin=v_min, vmax=v_max, zorder=0)
    
    # 底圖細節
    ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=3)
    
    # 疊加海流向量 (箭頭)
    ax.quiver(LON[::6, ::6], LAT[::6, ::6], u[::6, ::6], v[::6, ::6], color='cyan', alpha=0.3, scale=20, zorder=4)

    if run_btn:
        path = astar((np.argmin(np.abs(lat - s_lat)), np.argmin(np.abs(lon - s_lon))), 
                     (np.argmin(np.abs(lat - e_lat)), np.argmin(np.abs(lon - e_lon))))
        if path:
            py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
            ax.plot(px, py, color='#FF00FF', linewidth=3, label='AI Optimized Path', zorder=6)
            ax.plot([s_lon, e_lon], [s_lat, e_lat], color='white', linestyle='--', alpha=0.5, label='Direct Route', zorder=5)

    ax.scatter([s_lon, e_lon], [s_lat, e_lat], color=['lime', 'yellow'], s=100, edgecolors='black', zorder=7)
    
    # 添加流速色條
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
    cbar.set_label('Current Speed (m/s)', fontsize=10)
    
    st.pyplot(fig)
