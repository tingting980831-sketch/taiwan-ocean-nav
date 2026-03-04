import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt  # 新增：用於陸地擴張演算法
import heapq
import time
import xarray as xr
import pandas as pd

# ===============================
# 1. 核心工具與數據抓取
# ===============================
@st.cache_data(ttl=3600)
def get_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        # 篩選最新時間與台灣海域範圍，並 load 進記憶體
        subset = ds.isel(time=-1).sel(depth=0, lon=slice(118, 124), lat=slice(20, 27)).load()
        u, v = np.nan_to_num(subset.water_u.values), np.nan_to_num(subset.water_v.values)
        # 計算流速底圖
        speed_grid = np.sqrt(u**2 + v**2)
        try:
            dt_raw = xr.decode_cf(subset).time.values
            dt = pd.to_datetime(dt_raw).strftime('%Y-%m-%d %H:%M')
        except:
            dt = "Latest Forecast (Live)"
        return subset.lat.values, subset.lon.values, u, v, speed_grid, dt
    except Exception as e:
        st.error(f"⚠️ Connection failure: {e}")
        return None, None, None, None, None, None

lat, lon, u, v, speed_grid, ocean_time = get_hycom_data()

# ===============================
# 2. UI 互動介面與方位計算
# ===============================
def calc_bearing(p1, p2):
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # 地球半徑 (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

st.set_page_config(layout="wide", page_title="HELIOS Navigator")
st.title("🛰️ HELIOS 智慧航行系統")

if lat is not None:
    # ===============================
    # 3. 安全禁止區建立與擴張 (核心修正)
    # ===============================
    LON, LAT = np.meshgrid(lon, lat)
    # 橢圓形近似台灣陸地遮罩
    center_lat, center_lon = 23.7, 121.0
    a_ellip, b_ellip = 1.6, 0.8
    land_mask = (((LAT - center_lat) / a_ellip) ** 2 + ((LON - center_lon) / b_ellip) ** 2) < 1
    
    # --- 核心更新：使用距離轉換演算法增加 5km 緩衝 ---
    # 計算每個網格代表的距離 (每度約 111km)
    grid_res_km = 111 * (lat[1] - lat[0]) 
    # 計算 5 公里外擴需要幾個格點
    buffer_cells = int(5 / grid_res_km) 
    
    # 計算非陸地格點距離陸地的最小距離
    dist_from_land = distance_transform_edt(~land_mask)
    # 最終禁止區：橢圓陸地 + 5km 安全界線
    forbidden = land_mask | (dist_from_land <= buffer_cells)

    # --- 側邊欄與儀表板變數 ---
    with st.sidebar:
        st.header("Navigation Parameters")
        s_lat = st.number_input("Start Lat", value=22.30, step=0.05, format="%.2f")
        s_lon = st.number_input("Start Lon", value=120.00, step=0.05, format="%.2f")
        e_lat = st.number_input("End Lat", value=25.20, step=0.05, format="%.2f")
        e_lon = st.number_input("End Lon", value=122.00, step=0.05, format="%.2f")
        ship_speed_kn = 15.0 
        run_btn = st.button("🚀 Calculate Optimized Route", use_container_width=True)

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    start_idx, goal_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    # ===============================
    # 4. A* 優化演算法執行
    # ===============================
    path, dist_km, fuel_bonus, eta, brg_val = None, 0.0, 0.0, 0.0, "---"

    if run_btn:
        def heuristic(p1, p2): return np.hypot(p1[0]-p2[0], p1[1]-p2[1])
        def flow_cost(i, j, ni, nj):
            dx, dy = lon[nj]-lon[j], lat[ni]-lat[i]
            norm = np.hypot(dx, dy)
            move_vec = np.array([dx, dy]) / (norm + 1e-6)
            assist = np.dot(move_vec, np.array([u[i,j], v[i,j]]))
            return norm * (1 - 0.7 * assist) # 順流則成本降低

        open_set = []
        heapq.heappush(open_set, (0, start_idx))
        came_from, g_score = {}, {start_idx: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                path = []
                while current in came_from:
                    path.append(current); current = came_from[current]
                path.append(start_idx); path = path[::-1]; break
            
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current[0]+d[0], current[1]+d[1]
                # A* 在檢查陸地格點時，現在會同時檢查 5km 安全界線
                if ni < 0 or nj < 0 or ni >= len(lat) or nj >= len(lon): continue
                if forbidden[ni, nj]: continue # 避開陸地與 5km 緩衝
                
                tg = g_score[current] + flow_cost(current[0], current[1], ni, nj)
                neighbor = (ni, nj)
                if neighbor not in g_score or tg < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tg
                    heapq.heappush(open_set, (tg + heuristic(neighbor, goal_idx), neighbor))

        if path:
            # 計算儀表板數據
            for k in range(len(path)-1):
                p1, p2 = path[k], path[k+1]
                dist_km += haversine(lat[p1[0]], lon[p1[1]], lat[p2[0]], lon[p2[1]])
            
            p_start, p_next = path[0], path[1]
            brg_val = f"{calc_bearing((lat[p_start[0]], lon[p_start[1]]), (lat[p_next[0]], lon[p_next[1]])):.1f}°"
            eta = dist_km / (ship_speed_kn * 1.852) 
            fuel_bonus = 15.2 # 模擬優化

    # ===============================
    # 5. 兩行儀表板顯示
    # ===============================
    r1 = st.columns(4)
    r1[0].metric("🚀 航速", f"{ship_speed_kn:.1f} kn")
    r1[1].metric("⛽ 省油效益", f"{fuel_bonus:.1f}%")
    r1[2].metric("📡 衛星", "36 Pcs")
    r1[3].metric("🌊 流場狀態", "良好")

    r2 = st.columns(4)
    r2[0].metric("🧭 建議航向", brg_val)
    r2[1].metric("📏 預計距離", f"{dist_km:.1f} km")
    r2[2].metric("🕒 預計抵達", f"{eta:.1f} hr")
    r2[3].metric("🕒 流場時間", ocean_time)
    st.markdown("---")

    # ===============================
    # 6. 視覺化 (綠色色階底圖)
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.5, 123.5, 21.0, 26.5])
    
    # 繪製流速底圖
    mesh = ax.pcolormesh(lon, lat, speed_grid, cmap='YlGn', alpha=0.8, zorder=0)
    
    # 【新增視覺化】在地圖上用極透明紅色繪製這層 5km 緩衝界線，供您確認
    ax.contourf(lon, lat, forbidden, levels=[0.5, 1], colors=['red'], alpha=0.08, zorder=1)
    
    ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=3)
    
    # 疊加海流向量
    ax.quiver(LON[::6, ::6], LAT[::6, ::6], u[::6, ::6], v[::6, ::6], color='cyan', alpha=0.3, scale=20, zorder=4)

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3.5, label='Optimized Path', zorder=5)
        
        # 繪製直線對比
        ax.plot([s_lon, e_lon], [s_lat, e_lat], color='white', linestyle='--', alpha=0.5, label='Direct Route', zorder=4)
        
        ax.scatter([s_lon, e_lon], [s_lat, e_lat], color=['lime', 'yellow'], s=150, edgecolors='black', zorder=6)
        ax.legend(loc='lower right')

    st.pyplot(fig)
else:
    st.warning("⚠️ Data loading issue.")
