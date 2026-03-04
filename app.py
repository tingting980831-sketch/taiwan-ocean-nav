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
# 1. 精確海域 4D 數據抓取 (僅抓取台灣周遭)
# ===============================
@st.cache_data(ttl=3600)
def get_taiwan_4d_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        # 縮小範圍：僅抓取台灣周邊 (Memory-friendly slicing)
        subset = ds.isel(time=slice(-24, None)).sel(
            depth=0, 
            lon=slice(119.0, 123.0), # 鎖定台灣經度
            lat=slice(21.5, 26.0)    # 鎖定台灣緯度
        ).load()
        
        # 轉成 float32 節省記憶體
        u_4d = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v_4d = np.nan_to_num(subset.water_v.values).astype(np.float32)
        
        try:
            dt_raw = xr.decode_cf(subset).time.values
            dt_display = pd.to_datetime(dt_raw[0]).strftime('%Y-%m-%d %H:%M')
        except:
            dt_display = "Latest Taiwan Forecast"
            
        return subset.lat.values.astype(np.float32), subset.lon.values.astype(np.float32), u_4d, v_4d, dt_display
    except Exception as e:
        st.error(f"⚠️ 台灣海域數據抓取失敗: {e}")
        return None, None, None, None, None

def calc_bearing(p1, p2):
    lat1, lon1 = np.radians(p1); lat2, lon2 = np.radians(p2)
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統初始化
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Taiwan 4D")
st.title("🛰️ HELIOS 智慧航行系統 (台灣海域 4D 版)")

lat, lon, u_4d, v_4d, ocean_time = get_taiwan_4d_data()

if lat is not None:
    # 建立 12km 安全避岸界線
    LON, LAT = np.meshgrid(lon, lat)
    # 針對台灣地形稍微優化遮罩
    land_mask = (((LAT - 23.7) / 1.7) ** 2 + ((LON - 121.0) / 0.8) ** 2) < 1
    grid_res = lat[1] - lat[0]
    safe_margin = int(12 / (111 * grid_res))
    forbidden = land_mask | (distance_transform_edt(~land_mask) <= safe_margin)

    with st.sidebar:
        st.header("導航參數設定")
        if st.button("📍 定位至高雄港外", use_container_width=True):
            st.session_state.start_lat, st.session_state.start_lon = 22.50, 120.20
        
        s_lat = st.number_input("起點緯度 (Lat)", value=st.session_state.get('start_lat', 22.30), format="%.2f")
        s_lon = st.number_input("起點經度 (Lon)", value=st.session_state.get('start_lon', 120.00), format="%.2f")
        e_lat = st.number_input("終點緯度 (Lat)", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度 (Lon)", value=121.50, format="%.2f")
        ship_speed_kn = 15.0
        run_btn = st.button("🚀 開始 4D 路徑規劃", use_container_width=True)

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    # 陸地提醒
    if forbidden[s_idx] or forbidden[g_idx]:
        st.warning("⚠️ 提醒：設定點過於接近陸地或在禁航區內。")

    # ===============================
    # 3. 4D A* 動態規劃
    # ===============================
    path, dist_km, fuel_bonus, eta, brg_val = None, 0.0, 0.0, 0.0, "---"
    
    if run_btn and not forbidden[s_idx] and not forbidden[g_idx]:
        open_set = []
        heapq.heappush(open_set, (0, s_idx, 0.0))
        came_from, g_score = {}, {s_idx: 0.0}
        
        while open_set:
            _, current, curr_h = heapq.heappop(open_set)
            if current == g_idx:
                path = []
                while current in came_from:
                    path.append(current); current = came_from[current]
                path.append(s_idx); path = path[::-1]; break
            
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current[0]+d[0], current[1]+d[1]
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    # 抓取「抵達時」對應的小時索引
                    t_idx = min(int(curr_h), 23)
                    
                    step_dist = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
                    u_curr, v_curr = u_4d[t_idx, ni, nj], v_4d[t_idx, ni, nj]
                    
                    dx, dy = lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]
                    move_vec = np.array([dx, dy], dtype=np.float32) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_curr, v_curr])
                    
                    cost = step_dist * (1 - 0.7 * assist)
                    tg = g_score[current] + cost
                    
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        new_h = curr_h + (step_dist / (ship_speed_kn * 1.852))
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj), new_h))

        if path:
            for k in range(len(path)-1):
                dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
            brg_val = f"{calc_bearing((lat[path[0][0]], lon[path[0][1]]), (lat[path[1][0]], lon[path[1][1]])):.1f}°"
            eta = dist_km / (ship_speed_kn * 1.852)
            fuel_bonus = 14.5

    # ===============================
    # 4. 儀表板排版 (依照您的需求)
    # ===============================
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚀 航速", f"{ship_speed_kn} kn")
    m2.metric("⛽ 省油效益", f"{fuel_bonus:.1f}%")
    m3.metric("📡 衛星", "36 Pcs")
    m4.metric("🧭 建議航向", brg_val)

    m5, m6, m7 = st.columns([1, 1, 2])
    m5.metric("📏 預計距離", f"{dist_km:.1f} km")
    m6.metric("🕒 預計時間", f"{eta:.1f} hr")
    m7.metric("🕒 流場時間", ocean_time)
    st.markdown("---")

    # ===============================
    # 5. 視覺化 (縮小點位，終點星星)
    # ===============================
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([119.2, 122.8, 21.8, 25.8]) # 精確縮放到台灣周邊
    
    speed_0 = np.sqrt(u_4d[0]**2 + v_4d[0]**2)
    ax.pcolormesh(lon, lat, speed_0, cmap='YlGn', alpha=0.8, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.2, zorder=3)
    ax.quiver(LON[::6, ::6], LAT[::6, ::6], u_4d[0, ::6, ::6], v_4d[0, ::6, ::6], 
              color='cyan', alpha=0.2, scale=20, zorder=4)

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=5) 
        ax.scatter(s_lon, s_lat, color='lime', s=80, edgecolors='black', zorder=6) 
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, edgecolors='black', zorder=6)

    st.pyplot(fig)
