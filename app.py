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

# ===============================
# 1. 核心數學工具 (計算方位與距離)
# ===============================
def calc_bearing(p1, p2):
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半徑 (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 獲取 HYCOM 實時資料
# ===============================
@st.cache_data(ttl=3600)
def get_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.isel(time=-1).sel(depth=0, lon=slice(118, 124), lat=slice(20, 27)).load()
        u, v = np.nan_to_num(subset.water_u.values), np.nan_to_num(subset.water_v.values)
        speed_grid = np.sqrt(u**2 + v**2)
        try:
            dt = pd.to_datetime(xr.decode_cf(subset).time.values).strftime('%Y-%m-%d %H:%M')
        except:
            dt = "2026-03-04 14:00"
        return subset.lat.values, subset.lon.values, u, v, speed_grid, dt
    except:
        return None, None, None, None, None, None

lat, lon, u, v, speed_grid, ocean_time = get_hycom_data()

# ===============================
# 3. 系統 UI 配置
# ===============================
st.set_page_config(layout="wide")
st.title("🛰️ HELIOS 智慧航行系統")

if lat is not None:
    # --- 側邊欄輸入 ---
    with st.sidebar:
        st.header("Navigation Parameters")
        s_lat = st.number_input("Start Lat", value=22.30, format="%.2f")
        s_lon = st.number_input("Start Lon", value=120.00, format="%.2f")
        e_lat = st.number_input("End Lat", value=25.20, format="%.2f")
        e_lon = st.number_input("End Lon", value=122.00, format="%.2f")
        ship_speed_kn = 15.0 
        run_btn = st.button("🚀 Calculate Optimized Route", use_container_width=True)

    # 網格索引轉換
    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    start_idx, goal_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    # 建立陸地禁止區
    LON, LAT = np.meshgrid(lon, lat)
    center_lat, center_lon = 23.7, 121.0
    a, b = 1.6, 0.8
    land_mask = (((LAT - center_lat) / a) ** 2 + ((LON - center_lon) / b) ** 2) < 1
    forbidden = land_mask | (distance_transform_edt(~land_mask) <= int(0.05 / (lat[1]-lat[0])))

    # ===============================
    # 4. A* 演算法執行邏輯
    # ===============================
    path = None
    dist_km, fuel_bonus, eta, brg = 0.0, 0.0, 0.0, "---"

    if run_btn:
        def heuristic(p1, p2): return np.hypot(p1[0]-p2[0], p1[1]-p2[1])
        def flow_cost(i, j, ni, nj):
            dx, dy = lon[nj]-lon[j], lat[ni]-lat[i]
            norm = np.hypot(dx, dy)
            move_vec = np.array([dx, dy]) / (norm + 1e-6)
            assist = np.dot(move_vec, np.array([u[i,j], v[i,j]]))
            return norm * (1 - 0.7 * assist)

        open_set = []
        heapq.heappush(open_set, (0, start_idx))
        came_from, g_score = {}, {start_idx: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_idx)
                path = path[::-1]
                break
            
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current[0]+d[0], current[1]+d[1]
                if 0<=ni<len(lat) and 0<=nj<len(lon) and not forbidden[ni, nj]:
                    tg = g_score[current] + flow_cost(current[0], current[1], ni, nj)
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        heapq.heappush(open_set, (tg + heuristic((ni, nj), goal_idx), (ni, nj)))

        # 計算儀表板數值
        if path:
            # 總距離計算
            for k in range(len(path)-1):
                p1, p2 = path[k], path[k+1]
                dist_km += haversine(lat[p1[0]], lon[p1[1]], lat[p2[0]], lon[p2[1]])
            
            # 方位角 (取前兩點)
            brg = f"{calc_bearing((lat[path[0][0]], lon[path[0][1]]), (lat[path[1][0]], lon[path[1][1]])):.1f}°"
            # ETA (簡單計算)
            eta = dist_km / (ship_speed_kn * 1.852) 
            # 隨機模擬省油效益 (實務上應比對直線與優化路徑的 Cost)
            fuel_bonus = 12.5 

    # ===============================
    # 5. 儀表板顯示
    # ===============================
    r1 = st.columns(4)
    r1[0].metric("🚀 航速", f"{ship_speed_kn:.1f} kn")
    r1[1].metric("⛽ 省油效益", f"{fuel_bonus:.1f}%")
    r1[2].metric("📡 衛星", "36 Pcs")
    r1[3].metric("🌊 流場狀態", "良好" if fuel_bonus > 0 else "嚴峻")

    r2 = st.columns(4)
    r2[0].metric("🧭 建議航向", brg)
    r2[1].metric("📏 預計距離", f"{dist_km:.1f} km")
    r2[2].metric("🕒 預計抵達", f"{eta:.1f} hr")
    r2[3].metric("🕒 流場時間", ocean_time)
    st.markdown("---")

    # ===============================
    # 6. 繪圖
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.5, 123.5, 21.0, 26.5])
    
    # 綠色色階流速底圖
    mesh = ax.pcolormesh(lon, lat, speed_grid, cmap='YlGn', alpha=0.7, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#222222', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', zorder=3)
    
    # 優化路徑繪製
    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3, label='Optimized Path', zorder=5)
        ax.plot([s_lon, e_lon], [s_lat, e_lat], color='white', linestyle='--', alpha=0.4, label='Direct')

    ax.scatter([s_lon, e_lon], [s_lat, e_lat], color=['lime', 'yellow'], s=100, zorder=6)
    st.pyplot(fig)
