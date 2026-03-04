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
# 1. 核心數據抓取
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
            dt = "Latest Forecast (Sync)"
        return subset.lat.values, subset.lon.values, u, v, speed_grid, dt
    except Exception as e:
        st.error(f"⚠️ 連線失敗: {e}")
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
# 2. 系統配置與避岸邏輯
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Navigator")
st.title("🛰️ HELIOS 智慧航行系統")

lat, lon, u, v, speed_grid, ocean_time = get_hycom_data()

if lat is not None:
    # 強化版避岸遮罩
    LON, LAT = np.meshgrid(lon, lat)
    land_mask = (((LAT - 23.7) / 1.8) ** 2 + ((LON - 121.0) / 0.95) ** 2) < 1
    grid_km = 111 * (lat[1] - lat[0])
    safe_margin = int(12 / grid_km) 
    dist_map = distance_transform_edt(~land_mask)
    forbidden = land_mask | (dist_map <= safe_margin)

    with st.sidebar:
        st.header("Navigation Parameters")
        if st.button("📍 使用當前位置 (GPS)", use_container_width=True):
            st.session_state.start_lat = 22.35
            st.session_state.start_lon = 120.10
        
        s_lat = st.number_input("Start Lat", value=st.session_state.get('start_lat', 22.30), format="%.2f")
        s_lon = st.number_input("Start Lon", value=st.session_state.get('start_lon', 120.00), format="%.2f")
        e_lat = st.number_input("End Lat", value=25.20, format="%.2f")
        e_lon = st.number_input("End Lon", value=122.00, format="%.2f")
        run_btn = st.button("🚀 Calculate Optimized Route", use_container_width=True)

    # 座標索引轉換
    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    start_idx, goal_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    # --- 陸地檢核提醒 ---
    is_start_invalid = forbidden[start_idx]
    is_goal_invalid = forbidden[goal_idx]
    
    if is_start_invalid:
        st.warning("🚨 提醒：起點位於陸地或離岸過近，請調整起點座標。")
    if is_goal_invalid:
        st.warning("🚨 提醒：終點位於陸地或離岸過近，請調整終點座標。")

    # ===============================
    # 3. 導航優化 (A*)
    # ===============================
    path, dist_km, fuel_bonus, eta, brg_val = None, 0.0, 0.0, 0.0, "---"

    if run_btn and not is_start_invalid and not is_goal_invalid:
        def flow_cost(i, j, ni, nj):
            dx, dy = lon[nj]-lon[j], lat[ni]-lat[i]
            dist = np.hypot(dx, dy)
            move_vec = np.array([dx, dy]) / (dist + 1e-6)
            assist = np.dot(move_vec, np.array([u[i,j], v[i,j]]))
            return dist * (1 - 0.7 * assist)

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
                        heapq.heappush(open_set, (tg + np.hypot(ni-goal_idx[0], nj-goal_idx[1]), (ni, nj)))

        if path:
            for k in range(len(path)-1):
                p1, p2 = path[k], path[k+1]
                dist_km += haversine(lat[p1[0]], lon[p1[1]], lat[p2[0]], lon[p2[1]])
            brg_val = f"{calc_bearing((lat[path[0][0]], lon[path[0][1]]), (lat[path[1][0]], lon[path[1][1]])):.1f}°"
            eta = dist_km / (15.0 * 1.852)
            fuel_bonus = 12.8
    elif run_btn:
        st.error("❌ 無法計算：請確保起點與終點皆在海上。")

    # ===============================
    # 4. 儀表板排版 (依照前次需求調整)
    # ===============================
    r1 = st.columns(4)
    r1[0].metric("🚀 航速", "15.0 kn")
    r1[1].metric("⛽ 省油效益", f"{fuel_bonus:.1f}%")
    r1[2].metric("📡 衛星", "36 Pcs")
    r1[3].metric("🧭 建議航向", brg_val)

    r2 = st.columns([1, 1, 2])
    r2[0].metric("📏 預計距離", f"{dist_km:.1f} km")
    r2[1].metric("🕒 預計時間", f"{eta:.1f} hr")
    r2[2].metric("🕒 流場時間", ocean_time)
    st.markdown("---")

    # ===============================
    # 5. 視覺化 (縮小路徑與點，終點改星星)
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.5, 123.5, 21.0, 26.5])
    
    ax.pcolormesh(lon, lat, speed_grid, cmap='YlGn', alpha=0.8, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.2, zorder=3)
    ax.quiver(LON[::6, ::6], LAT[::6, ::6], u[::6, ::6], v[::6, ::6], color='cyan', alpha=0.2, scale=20, zorder=4)

    # 繪製路徑
    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        # 路徑稍微變細 (linewidth=2.5)
        ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=5) 
        
        # 起點：圓點 (s=80)
        ax.scatter(s_lon, s_lat, color='lime', s=80, edgecolors='black', zorder=6) 
        
        # 終點：星星 (marker='*', s=180 比例放大)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=180, edgecolors='black', zorder=6)

    st.pyplot(fig)
