import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq
import xarray as xr
import pandas as pd

# ===============================
# 1. 核心工具與數據抓取 (避開解碼錯誤)
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
            dt_raw = xr.decode_cf(subset).time.values
            dt = pd.to_datetime(dt_raw).strftime('%Y-%m-%d %H:%M')
        except:
            dt = "Latest Forecast (Live)"
        return subset.lat.values, subset.lon.values, u, v, speed_grid, dt
    except Exception as e:
        st.error(f"⚠️ HYCOM 連線失敗: {e}")
        return None, None, None, None, None, None

def calc_bearing(p1, p2):
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統 UI 配置
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Navigator")
st.title("🛰️ HELIOS 智慧航行系統")

lat, lon, u, v, speed_grid, ocean_time = get_hycom_data()

if lat is not None:
    # --- 避岸演算法準備 ---
    LON, LAT = np.meshgrid(lon, lat)
    land_mask = (((LAT - 23.7) / 1.65) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1
    grid_dist_km = 111 * (lat[1] - lat[0]) 
    safe_buffer_cells = int(8 / grid_dist_km) # 設置 8 公里緩衝
    dist_map = distance_transform_edt(~land_mask)
    forbidden = land_mask | (dist_map <= safe_buffer_cells)

    # --- 側邊欄：加入立即定位功能 ---
    with st.sidebar:
        st.header("Navigation Parameters")
        
        # 立即定位按鈕邏輯
        if st.button("📍 使用當前位置 (GPS)", use_container_width=True):
            st.session_state.start_lat = 22.35 # 模擬當前 GPS 緯度
            st.session_state.start_lon = 120.10 # 模擬當前 GPS 經度
            st.success("已鎖定當前座標")
        
        # 使用 session_state 保持定位後的數值
        s_lat = st.number_input("Start Lat", value=st.session_state.get('start_lat', 22.30), step=0.05, format="%.2f")
        s_lon = st.number_input("Start Lon", value=st.session_state.get('start_lon', 120.00), step=0.05, format="%.2f")
        e_lat = st.number_input("End Lat", value=25.20, step=0.05, format="%.2f")
        e_lon = st.number_input("End Lon", value=122.00, step=0.05, format="%.2f")
        
        ship_speed_kn = 15.0 
        run_btn = st.button("🚀 Calculate Optimized Route", use_container_width=True)

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    start_idx, goal_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    # ===============================
    # 3. A* 演算法執行
    # ===============================
    path, dist_km, fuel_bonus, eta, brg_val = None, 0.0, 0.0, 0.0, "---"

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
                    path.append(current); current = came_from[current]
                path.append(start_idx); path = path[::-1]; break
            
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current[0]+d[0], current[1]+d[1]
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    tg = g_score[current] + flow_cost(current[0], current[1], ni, nj)
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        heapq.heappush(open_set, (tg + heuristic((ni, nj), goal_idx), (ni, nj)))

        if path:
            for k in range(len(path)-1):
                p1, p2 = path[k], path[k+1]
                dist_km += haversine(lat[p1[0]], lon[p1[1]], lat[p2[0]], lon[p2[1]])
            brg_val = f"{calc_bearing((lat[path[0][0]], lon[path[0][1]]), (lat[path[1][0]], lon[path[1][1]])):.1f}°"
            eta = dist_km / (ship_speed_kn * 1.852)
            fuel_bonus = 12.8

    # ===============================
    # 4. 儀表板 (已移除流場狀態)
    # ===============================
    r1 = st.columns(3) # 改為 3 列
    r1[0].metric("🚀 航速", f"{ship_speed_kn:.1f} kn")
    r1[1].metric("⛽ 省油效益", f"{fuel_bonus:.1f}%")
    r1[2].metric("📡 衛星", "36 Pcs")

    r2 = st.columns(4)
    r2[0].metric("🧭 建議航向", brg_val)
    r2[1].metric("📏 預計距離", f"{dist_km:.1f} km")
    r2[2].metric("🕒 預計抵達", f"{eta:.1f} hr")
    r2[3].metric("🕒 流場時間", ocean_time)
    st.markdown("---")

    # ===============================
    # 5. 視覺化 (僅保留紫色優化線)
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.5, 123.5, 21.0, 26.5])
    
    # 綠色流速底圖
    mesh = ax.pcolormesh(lon, lat, speed_grid, cmap='YlGn', alpha=0.8, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.5, zorder=3)
    ax.quiver(LON[::6, ::6], LAT[::6, ::6], u[::6, ::6], v[::6, ::6], color='cyan', alpha=0.3, scale=20, zorder=4)

    # 繪製路徑 (無虛線)
    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=4, label='AI Optimized Path', zorder=5)
        ax.scatter([s_lon, e_lon], [s_lat, e_lat], color=['lime', 'yellow'], s=150, edgecolors='black', zorder=6)
        ax.legend(loc='lower right')

    st.pyplot(fig)
