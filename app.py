import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd
import os
from matplotlib.font_manager import FontProperties

# ===============================
# 1. 數據抓取 (自動對接最新時間點)
# ===============================
@st.cache_data(ttl=600)
def get_v10_latest_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        # 使用 decode_times=True 確保時間戳記正確
        ds = xr.open_dataset(url, decode_times=True)
        # 抓取最後一個時間點 (最即時資料)
        latest_ds = ds.sel(
            time=ds.time[-1], 
            depth=0, 
            lon=slice(117.0, 125.0), 
            lat=slice(20.0, 27.0)
        ).load()
        
        u_vals = np.nan_to_num(latest_ds.water_u.values).astype(np.float32)
        v_vals = np.nan_to_num(latest_ds.water_v.values).astype(np.float32)
        lats = latest_ds.lat.values.astype(np.float32)
        lons = latest_ds.lon.values.astype(np.float32)
        
        dt_display = pd.to_datetime(latest_ds.time.values).strftime('%Y-%m-%d %H:%M')
        return lats, lons, u_vals, v_vals, dt_display
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, None, None, None, "數據連線異常"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統 UI 與參數
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V10 AI Inverter")
st.title("🛰️ HELIOS 智慧導航系統 (AI 變頻 + 最新流場對接)")

lats, lons, u_data, v_data, ocean_time = get_v10_latest_data()

if lats is not None:
    LON, LAT = np.meshgrid(lons, lats)
    
    # --- 極致陸地遮罩 (無 12km 緩衝，路徑可緊貼) ---
    mask_main = (((LAT - 23.75) / 1.85) ** 2 + ((LON - 121.0) / 0.72) ** 2) < 1
    mask_penghu = (((LAT - 23.58) / 0.25) ** 2 + ((LON - 119.58) / 0.22) ** 2) < 1
    forbidden = mask_main | mask_penghu

    with st.sidebar:
        st.header("📍 導航任務設定")
        s_lat = st.number_input("起點緯度", value=22.35, format="%.2f")
        s_lon = st.number_input("起點經度", value=120.15, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.05, format="%.2f")
        
        st.markdown("---")
        base_speed = st.slider("巡航基準航速 (kn)", 10.0, 25.0, 15.0)
        mode = st.radio("動力模式", ["固定輸出", "AI 變頻省油"])
        run_btn = st.button("🚀 執行優化路徑計算", use_container_width=True)

    def get_idx(la, lo): return np.argmin(np.abs(lats-la)), np.argmin(np.abs(lons-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    path, dist_km, eta = None, 0.0, 0.0

    # ===============================
    # 3. 4D A* AI 變頻演算法
    # ===============================
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
                    
                    # 向量計算
                    dx, dy = lons[nj]-lons[current[1]], lats[ni]-lats[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_f, v_f])
                    
                    # AI 變頻邏輯
                    actual_engine_speed = base_speed
                    if mode == "AI 變頻省油":
                        if assist < -0.4: actual_engine_speed *= 0.88 # 遇逆流降速
                        elif assist > 0.4: actual_engine_speed *= 1.05 # 遇順流微增
                    
                    cost = step_dist * (1 - 0.75 * assist)
                    tg = g_score[current] + cost
                    
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        # 計算對地速度合成 SOG
                        v_sog_kts = actual_engine_speed + (assist * 1.94384)
                        v_sog_kts = max(v_sog_kts, 1.2)
                        new_h = curr_h + (step_dist / (v_sog_kts * 1.852))
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj), new_h))

    # ===============================
    # 4. 儀表板與地圖繪製 (移除虛線)
    # ===============================
    if path:
        for k in range(len(path)-1):
            dist_km += haversine(lats[path[k][0]], lons[path[k][1]], lats[path[k+1][0]], lons[path[k+1][1]])
        eta = g_score[g_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 航行距離", f"{dist_km:.1f} km" if path else "---")
    c2.metric("🕒 預計到達時間", f"{eta:.1f} hr" if path else "---")
    c3.metric("⛽ 模式", mode)
    c4.metric("📡 衛星技術", "SIC Active")
    st.caption(f"📅 最新流場同步時間: {ocean_time} (HYCOM 4D)")
    st.markdown("---")

    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8]) 
    
    speed_0 = np.sqrt(u_data**2 + v_data**2)
    mesh = ax.pcolormesh(lons, lats, speed_0, cmap='YlGnBu', alpha=0.7, zorder=0)
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)', fraction=0.03, pad=0.04)

    # 專業陸地遮罩效果 (zorder=5 蓋掉邊緣海流)
    ax.add_feature(cfeature.LAND, facecolor='#202020', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.2, zorder=6)

    if path:
        py, px = [lats[p[0]] for p in path], [lons[p[1]] for p in path]
        # 繪製單一優化實線 (桃紅色)
        ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=10, label='HELIOS AI 優化航線') 
        ax.scatter(s_lon, s_lat, color='lime', s=100, edgecolors='black', zorder=11) 
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=250, edgecolors='black', zorder=11)
    
    ax.legend(loc='lower right')
    st.pyplot(fig)
