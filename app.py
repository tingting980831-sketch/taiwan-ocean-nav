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
# 1. 數據源嫁接：ESPC FMRC 最佳路徑 (避開 -72 錯誤)
# ===============================
@st.cache_data(ttl=600)
def get_v15_3_stable_data():
    # 使用 FMRC Best Path，這是目前最穩定的實時嫁接點
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/FMRC/runs/ESPC-D-V02_RUN_best.ncd"
    
    try:
        # 強制指定 netcdf4 引擎，並關閉時間解碼以解決解析問題
        ds = xr.open_dataset(url, decode_times=False, engine='netcdf4')
        
        # 獲取現在 UTC 時間並進行時間嫁接
        units = ds.time.units
        base_str = units.split("since ")[1].split(".")[0]
        base_date = pd.to_datetime(base_str).replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        
        # 尋找與現在最接近的時間索引
        target_h = (now_utc - base_date).total_seconds() / 3600
        time_vals = ds.time.values
        idx = np.abs(time_vals - target_h).argmin()
        
        # 先選取海域再 load，減少傳輸負荷防止伺服器斷線
        subset = ds.isel(time=idx).sel(
            depth=0, 
            lon=slice(117.2, 124.8), 
            lat=slice(20.2, 26.8)
        ).load()
        
        u = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values).astype(np.float32)
        
        sync_label = (base_date + pd.Timedelta(hours=float(time_vals[idx]))).strftime('%Y-%m-%d %H:%M UTC')
        return subset.lat.values, subset.lon.values, u, v, sync_label
    except Exception as e:
        st.error(f"❌ 數據嫁接持續失敗: {e}")
        st.info("💡 解決方案：這通常是 HYCOM 伺服器流量管制。請在 Sidebar 點擊『重新連線』或切換網路環境。")
        return None, None, None, None, "Link Fail"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統介面與陸地避讓
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V15.3 Stable")
st.title("🛰️ HELIOS 智慧導航 (ESPC 實時穩定版)")

lat, lon, u_raw, v_raw, sync_time = get_v15_3_stable_data()

if lat is not None:
    LON, LAT = np.meshgrid(lon, lat)
    
    # 精確遮罩：修正布林運算優先權並移除 12km 緩衝
    mask_tw = (((LAT - 23.75) / 1.85) ** 2 + ((LON - 121.0) / 0.75) ** 2) < 1
    mask_is = (((LAT - 23.5) / 0.3) ** 2 + ((LON - 119.6) / 0.3) ** 2) < 1
    forbidden = mask_tw | mask_is

    with st.sidebar:
        st.header("📍 導航設定")
        s_lat = st.number_input("起點緯度", value=22.35)
        s_lon = st.number_input("起點經度", value=120.10)
        e_lat = st.number_input("終點緯度", value=25.20)
        e_lon = st.number_input("終點經度", value=122.00)
        base_speed = st.slider("巡航航速 (kn)", 10, 25, 15)
        if st.button("🔄 清除快取並重試"):
            st.cache_data.clear()
            st.rerun()
        run_btn = st.button("🚀 計算同步優化路徑", use_container_width=True)

    # ===============================
    # 3. 4D A* 實時運算
    # ===============================
    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
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
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    step_d = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
                    u_f, v_f = u_raw[ni, nj], v_raw[ni, nj]
                    dx, dy = lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_f, v_f])
                    
                    cost = step_d * (1 - 0.75 * assist)
                    tg = g_score[current] + cost
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj)))

    # ===============================
    # 4. 顯示與繪圖
    # ===============================
    if path:
        for k in range(len(path)-1):
            dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
        eta = dist_km / (base_speed * 1.852)

    st.success(f"✅ 已成功對接 2026-03-05 ESPC 數據：{sync_time}")
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8])
    
    speed = np.sqrt(u_raw**2 + v_raw**2)
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', shading='auto', alpha=0.9)
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)', fraction=0.03, pad=0.04)
    
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.6, zorder=6)

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=10, label='HELIOS 同步路徑')
        ax.scatter(s_lon, s_lat, color='lime', s=100, zorder=11)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=250, zorder=11)
    
    st.pyplot(fig)
