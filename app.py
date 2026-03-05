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
# 1. 數據源嫁接：NCEP RTOFS 實時同步
# ===============================
@st.cache_data(ttl=3600)
def get_rtofs_data():
    # 自動獲取今日日期對接 NCEP RTOFS 伺服器
    today_str = datetime.now(timezone.utc).strftime('%Y%m%d')
    url = f"https://nomads.ncep.noaa.gov/dods/rtofs/rtofs{today_str}/rtofs_glo_2ds_forecast_daily_diag"
    
    try:
        # RTOFS 伺服器穩定度較高，直接開啟
        ds = xr.open_dataset(url, decode_times=True)
        
        # 定位到 2026-03-05 最接近的時間點
        now_utc = datetime.now(timezone.utc)
        
        # RTOFS 經度為 0~360，需轉換為 -180~180
        subset = ds.sel(time=now_utc, method='nearest').sel(
            lat=slice(20.0, 27.0),
            lon=slice(117.0, 125.0)
        ).load()
        
        # RTOFS 的變數名稱為 u_velocity, v_velocity
        u = np.nan_to_num(subset.u_velocity.values[0]).astype(np.float32)
        v = np.nan_to_num(subset.v_velocity.values[0]).astype(np.float32)
        
        sync_time = pd.to_datetime(subset.time.values).strftime('%Y-%m-%d %H:%M UTC')
        return subset.lat.values, subset.lon.values, u, v, sync_time
    except Exception as e:
        st.error(f"🛰️ RTOFS 數據對接失敗: {e}")
        st.info("💡 提示：若出現 404，可能是 NOAA 正在更新今日數據，請稍後重試。")
        return None, None, None, None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 介面設定與陸地遮罩
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V16.2 RTOFS Live")
st.title("🛰️ HELIOS 智慧導航系統 (NCEP 實時版)")

lat, lon, u_raw, v_raw, live_time = get_rtofs_data()

if lat is not None:
    LON, LAT = np.meshgrid(lon, lat)
    
    # 台灣精細避讓遮罩 (V16 修正語法)
    mask_tw = (((LAT - 23.75) / 1.85) ** 2 + ((LON - 121.0) / 0.75) ** 2) < 1
    mask_ph = (((LAT - 23.5) / 0.3) ** 2 + ((LON - 119.6) / 0.3) ** 2) < 1
    forbidden = mask_tw | mask_ph

    with st.sidebar:
        st.header("📍 任務參數設定")
        s_lat = st.number_input("起點緯度", value=22.35, format="%.2f")
        s_lon = st.number_input("起點經度", value=120.10, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        
        st.markdown("---")
        base_speed = st.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
        run_btn = st.button("🚀 執行優化路徑計算", use_container_width=True)
        if st.button("🔄 強制刷新數據"):
            st.cache_data.clear()
            st.rerun()

    # ===============================
    # 3. A* 路徑優化 (4D 合成流場)
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
                    
                    # 取得 RTOFS 實時向量
                    u_f, v_f = u_raw[ni, nj], v_raw[ni, nj]
                    dx, dy = lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_f, v_f])
                    
                    # 考慮海流正向輔助則減少成本 (輔助係數 0.8)
                    cost = step_d * (1 - 0.8 * assist)
                    tg = g_score[current] + cost
                    
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj)))

    # ===============================
    # 4. 儀表板與視覺化
    # ===============================
    if path:
        for k in range(len(path)-1):
            dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
        eta = dist_km / (base_speed * 1.852)

    st.success(f"✅ RTOFS 實時同步點：{live_time}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📏 航行距離", f"{dist_km:.1f} km" if path else "---")
    col2.metric("🕒 預計到達時間", f"{eta:.1f} hr" if path else "---")

    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8]) 
    
    # 繪製流速熱圖 (仿 NODASS 色彩)
    speed = np.sqrt(u_raw**2 + v_raw**2)
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', alpha=0.9, shading='auto')
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)', fraction=0.03, pad=0.04)

    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=0.8, zorder=6)

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=10, label='HELIOS 優化路徑')
        ax.scatter(s_lon, s_lat, color='lime', s=100, zorder=11)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=250, zorder=11)
    
    ax.legend(loc='lower right')
    st.pyplot(fig)
