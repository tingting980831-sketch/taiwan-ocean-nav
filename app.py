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
# 1. 數據實時同步嫁接 (UTC Now Sync)
# ===============================
@st.cache_data(ttl=600)
def get_v12_sync_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        # 1. 以非解碼模式開啟，處理報錯問題
        ds = xr.open_dataset(url, decode_times=False)
        
        # 2. 獲取數據庫的時間基準
        units = ds.time.units  # "hours since 2000-01-01..."
        base_time_str = units.split("since ")[1].split(".")[0]
        base_date = pd.to_datetime(base_time_str).replace(tzinfo=timezone.utc)
        
        # 3. 計算「現在 UTC 時間」與數據基準的差值 (小時)
        now_utc = datetime.now(timezone.utc)
        hours_since_base = (now_utc - base_date).total_seconds() / 3600
        
        # 4. 在數據軸中尋找「最接近現在」的時間索引
        time_array = ds.time.values
        # 找出與當前時間差值最小的索引
        nearest_idx = np.argmin(np.abs(time_array - hours_since_base))
        
        # 5. 抓取該同步時間點的資料
        subset = ds.isel(time=nearest_idx).sel(
            depth=0, lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)
        ).load()
        
        u_val = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v_val = np.nan_to_num(subset.water_v.values).astype(np.float32)
        
        # 6. 格式化顯示同步的時間標籤
        sync_time = (base_date + pd.Timedelta(hours=float(time_array[nearest_idx]))).strftime('%Y-%m-%d %H:%M UTC')
        
        return subset.lat.values.astype(np.float32), subset.lon.values.astype(np.float32), u_val, v_val, sync_time
    except Exception as e:
        st.error(f"實時嫁接失敗: {e}")
        return None, None, None, None, "Sync Error"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統 UI
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V12 Real-Time Sync")
st.title("🛰️ HELIOS 智慧導航系統 (實時流場同步版)")

lat, lon, u_2d, v_2d, sync_time_label = get_v12_sync_data()

if lat is not None:
    # 建立 2D 遮罩 (此版本不使用 4D，改用當前最準確的平面流場)
    LON, LAT = np.meshgrid(lon, lat)
    forbidden = (((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1 | \
                (((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1

    with st.sidebar:
        st.header("📍 導航任務")
        s_lat = st.number_input("起點緯度", value=22.35, format="%.2f")
        s_lon = st.number_input("起點經度", value=120.10, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        base_speed = st.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
        run_btn = st.button("🚀 執行實時同步路徑計算", use_container_width=True)

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    path, dist_km, eta = None, 0.0, 0.0

    # ===============================
    # 3. A* 演算法 (對接實時流場)
    # ===============================
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
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    step_d = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
                    
                    # 抓取「現在這一刻」的流場向量
                    u_curr, v_curr = u_2d[ni, nj], v_2d[ni, nj]
                    dx, dy = lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_curr, v_curr])
                    
                    cost = step_d * (1 - 0.75 * assist)
                    tg = g_score[current] + cost
                    
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj)))

    # ===============================
    # 4. 數據儀表板
    # ===============================
    if path:
        for k in range(len(path)-1):
            dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
        # 計算 ETA (考慮平均流速影響)
        eta = g_score[g_idx] / (base_speed * 1.852 / 10) # 簡易成本轉換

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 航行距離", f"{dist_km:.1f} km" if path else "---")
    c2.metric("🕒 預計到達時間", f"{eta:.1f} hr" if path else "---")
    c3.metric("🚢 動力模式", "100% Fixed")
    c4.metric("🛰️ 同步狀態", "Live UTC Sync")

    st.info(f"🕒 數據源同步點: {sync_time_label} (自動對接最接近現在之流場)")
    st.markdown("---")

    # ===============================
    # 5. 地圖顯示
    # ===============================
    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8]) 
    
    speed = np.sqrt(u_2d**2 + v_2d**2)
    ax.pcolormesh(lon, lat, speed, cmap='YlGnBu', alpha=0.7, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#202020', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.2, zorder=3)

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=5, label='同步優化路徑') 
        ax.scatter(s_lon, s_lat, color='lime', s=100, zorder=6) 
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=250, zorder=6)
    
    st.pyplot(fig)
