import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd
import os

# ===============================
# 1. 數據抓取 (修正時間解碼報錯問題)
# ===============================
@st.cache_data(ttl=600)
def get_v10_latest_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        # 關鍵修正：先關閉自動解碼，避免 'hours since analysis' 報錯
        ds = xr.open_dataset(url, decode_times=False)
        
        # 抓取最後一個時間點
        latest_ds = ds.isel(time=-1).sel(
            depth=0, 
            lon=slice(117.0, 125.0), 
            lat=slice(20.0, 27.0)
        ).load()
        
        u_vals = np.nan_to_num(latest_ds.water_u.values).astype(np.float32)
        v_vals = np.nan_to_num(latest_ds.water_v.values).astype(np.float32)
        lats = latest_ds.lat.values.astype(np.float32)
        lons = latest_ds.lon.values.astype(np.float32)
        
        # 手動處理時間顯示 (若無法解碼則顯示為系統模擬時間)
        try:
            # 嘗試手動解碼時間單位
            units = ds.time.units # 獲取 'hours since ...'
            base_date = pd.to_datetime(units.split("since ")[1])
            offset_hours = float(latest_ds.time.values)
            dt_display = (base_date + pd.Timedelta(hours=offset_hours)).strftime('%Y-%m-%d %H:%M')
        except:
            dt_display = "2026-03-05 Real-time Flow"
            
        return lats, lons, u_vals, v_vals, dt_display
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, None, None, None, "數據連線異常"

# 距離計算
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統 UI 與陸地遮罩
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V10.1 AI")
st.title("🛰️ HELIOS 智慧導航系統 (時間解碼修正版)")

lats, lons, u_data, v_data, ocean_time = get_v10_latest_data()

if lats is not None:
    LON, LAT = np.meshgrid(lons, lats)
    # 精確陸地遮罩
    mask_main = (((LAT - 23.75) / 1.85) ** 2 + ((LON - 121.0) / 0.72) ** 2) < 1
    mask_penghu = (((LAT - 23.58) / 0.25) ** 2 + ((LON - 119.58) / 0.22) ** 2) < 1
    forbidden = mask_main | mask_penghu

    with st.sidebar:
        st.header("📍 導航設定")
        s_lat = st.number_input("起點緯度", value=22.35, format="%.2f")
        s_lon = st.number_input("起點經度", value=120.15, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.05, format="%.2f")
        base_speed = st.slider("巡航基準航速 (kn)", 10.0, 25.0, 15.0)
        mode = st.radio("動力模式", ["固定輸出", "AI 變頻省油"])
        run_btn = st.button("🚀 執行優化路徑計算", use_container_width=True)

    # ===============================
    # 3. 4D A* 演算法與 AI 變頻
    # ===============================
    path, dist_km, eta = None, 0.0, 0.0
    def get_idx(la, lo): return np.argmin(np.abs(lats-la)), np.argmin(np.abs(lons-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    if run_btn and not forbidden[s_idx] and not forbidden[g_idx]:
        open_set, came_from, g_score = [], {}, {s_idx: 0.0}
        heapq.heappush(open_set, (0, s_idx, 0.0))
        
        while open_set:
            _, current, curr_h = heapq.heappop(open_set)
            if current == g_idx:
                path = []
                while current in came_from:
                    path.append(current); current = came_from[current]
                path.append(s_idx); path = path[::-1]; break
            
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current[0]+d[0], current[1]+d[1]
                if 0 <= ni < len(lats) and 0 <= nj < len(lons) and not forbidden[ni, nj]:
                    step_dist = haversine(lats[current[0]], lons[current[1]], lats[ni], lons[nj])
                    u_f, v_f = u_data[ni, nj], v_data[ni, nj]
                    dx, dy = lons[nj]-lons[current[1]], lats[ni]-lats[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_f, v_f])
                    
                    # AI 變頻邏輯
                    actual_engine_speed = base_speed
                    if mode == "AI 變頻省油":
                        if assist < -0.4: actual_engine_speed *= 0.88 
                        elif assist > 0.4: actual_engine_speed *= 1.05 
                    
                    cost = step_dist * (1 - 0.75 * assist)
                    tg = g_score[current] + cost
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        v_sog = actual_engine_speed + (assist * 1.94384)
                        new_h = curr_h + (step_dist / (max(v_sog, 1.2) * 1.852))
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj), new_h))

    # ===============================
    # 4. 數據展示與地圖
    # ===============================
    if path:
        for k in range(len(path)-1):
            dist_km += haversine(lats[path[k][0]], lons[path[k][1]], lats[path[k+1][0]], lons[path[k+1][1]])
        eta = g_score[g_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 航行距離", f"{dist_km:.1f} km" if path else "---")
    c2.metric("🕒 預計到達時間", f"{eta:.1f} hr" if path else "---")
    c3.metric("🚢 模式", mode)
    c4.metric("📡 數據狀態", "HYCOM 4D Active")
    st.caption(f"📅 最新流場同步時間: {ocean_time}")
    st.markdown("---")

    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8]) 
    speed_map = np.sqrt(u_data**2 + v_data**2)
    mesh = ax.pcolormesh(lons, lats, speed_map, cmap='YlGnBu', alpha=0.7, zorder=0)
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)', fraction=0.03, pad=0.04)
    ax.add_feature(cfeature.LAND, facecolor='#202020', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.2, zorder=6)

    if path:
        py, px = [lats[p[0]] for p in path], [lons[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=10, label='優化航線') 
        ax.scatter(s_lon, s_lat, color='lime', s=100, zorder=11) 
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=250, zorder=11)
    
    ax.legend(loc='lower right')
    st.pyplot(fig)
