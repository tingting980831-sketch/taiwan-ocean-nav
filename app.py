import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd
from datetime import datetime, timezone

# ===============================
# 1. 數據抓取：強制對接 2026 ACTIVE 數據源 (ESPC)
# ===============================
@st.cache_data(ttl=600)
def get_v15_live_data():
    # 這是目前唯一能提供 2026 實時數據的網址
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/uv3z"
    try:
        # 使用 netcdf4 引擎解決 Errno -72 報錯
        ds = xr.open_dataset(url, decode_times=False, engine='netcdf4')
        
        # 獲取現在 UTC 時間並尋找最接近的時間格點
        units = ds.time.units
        base_str = units.split("since ")[1].split(".")[0]
        base_date = pd.to_datetime(base_str).replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        
        target_h = (now_utc - base_date).total_seconds() / 3600
        time_vals = ds.time.values
        idx = np.abs(time_vals - target_h).argmin()
        
        # 讀取台灣周邊海域
        subset = ds.isel(time=idx).sel(
            depth=0, lon=slice(117.2, 124.8), lat=slice(20.2, 26.8)
        ).load()
        
        u = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values).astype(np.float32)
        
        sync_label = (base_date + pd.Timedelta(hours=float(time_vals[idx]))).strftime('%Y-%m-%d %H:%M UTC')
        return subset.lat.values, subset.lon.values, u, v, sync_label
    except Exception as e:
        st.error(f"數據對接失敗: {e}。請點擊右上角選單執行 'Clear Cache' 後重試。")
        return None, None, None, None, "Link Fail"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統 UI 與遮罩設定
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V15.2 Live")
st.title("🛰️ HELIOS 智慧導航系統 (2026 實時同步版)")

lat_vals, lon_vals, u_2d, v_2d, ocean_time = get_v15_live_data()

if lat_vals is not None:
    LON, LAT = np.meshgrid(lon_vals, lat_vals)
    
    # 修正布林運算優先權 (增加括號)
    mask_tw = (((LAT - 23.75) / 1.85) ** 2 + ((LON - 121.0) / 0.75) ** 2) < 1
    mask_is = (((LAT - 23.5) / 0.3) ** 2 + ((LON - 119.6) / 0.3) ** 2) < 1
    forbidden = mask_tw | mask_is

    with st.sidebar:
        st.header("📍 導航任務設定")
        s_lat = st.number_input("起點緯度", value=22.35, format="%.2f")
        s_lon = st.number_input("起點經度", value=120.10, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        base_speed = st.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
        run_btn = st.button("🚀 執行實時優化路徑", use_container_width=True)

    # ===============================
    # 3. A* 演算法
    # ===============================
    def get_idx(la, lo): return np.argmin(np.abs(lat_vals-la)), np.argmin(np.abs(lon_vals-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)
    path, dist_km, eta = None, 0.0, 0.0

    if run_btn and not forbidden[s_idx] and not forbidden[g_idx]:
        open_set, came_from, g_score = [], {}, {s_idx: 0.0}
        heapq.heappush(open_set, (0, s_idx))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == g_idx:
                path = []
                while current in came_from:
                    path.append(current); current = came_from[current]
                path.append(s_idx); path = path[::-1]; break
            
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current[0]+d[0], current[1]+d[1]
                if 0 <= ni < len(lat_vals) and 0 <= nj < len(lon_vals) and not forbidden[ni, nj]:
                    step_dist = haversine(lat_vals[current[0]], lon_vals[current[1]], lat_vals[ni], lon_vals[nj])
                    u_f, v_f = u_2d[ni, nj], v_2d[ni, nj]
                    dx, dy = lon_vals[nj]-lon_vals[current[1]], lat_vals[ni]-lat_vals[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_f, v_f])
                    
                    cost = step_dist * (1 - 0.75 * assist)
                    tg = g_score[current] + cost
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj)))

    # ===============================
    # 4. 儀表板與地圖
    # ===============================
    if path:
        for k in range(len(path)-1):
            dist_km += haversine(lat_vals[path[k][0]], lon_vals[path[k][1]], lat_vals[path[k+1][0]], lon_vals[path[k+1][1]])
        eta = dist_km / (base_speed * 1.852)

    st.success(f"✅ 已成功對接實時數據源：{ocean_time}")
    
    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8]) 
    
    speed = np.sqrt(u_2d**2 + v_2d**2)
    # 使用與 NODASS 同級別的 turbo 色彩映射
    mesh = ax.pcolormesh(lon_vals, lat_vals, speed, cmap='turbo', shading='auto', alpha=0.9, zorder=0)
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)', fraction=0.03, pad=0.04)

    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.8, zorder=6)
    
    if path:
        py, px = [lat_vals[p[0]] for p in path], [lon_vals[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=10, label='優化路徑')
        ax.scatter(s_lon, s_lat, color='lime', s=100, zorder=11)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=250, zorder=11)
    
    st.pyplot(fig)
