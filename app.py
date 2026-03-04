# ==========================================
# AI 海象導航系統 - HYCOM 實時資料整合版
# ==========================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq
import time
import xarray as xr  # 處理遠端海洋大數據
import pandas as pd

# ------------------------------------------
# Streamlit 基本頁面設定
# ------------------------------------------
st.set_page_config(page_title="⚓ AI 海象導航系統", layout="wide")
st.title("⚓ AI 海象導航系統 (HYCOM Real-time Integration)")
st.markdown("本系統直接連線至 **HYCOM 官方伺服器 (OPeNDAP)**，獲取當前全球海洋預報資料。")

# ------------------------------------------
# 核心功能：連線 HYCOM 官方伺服器
# ------------------------------------------
@st.cache_data(ttl=3600)  # 資料快取 1 小時，避免重複抓取造成網路卡頓
def get_hycom_realtime_data():
    # HYCOM 官方最新實驗數據網址 (expt_93.0 為目前活躍版本)
    # uv3z 代表包含 U/V 向量與 3D 深度資料的檔案路徑
    tds_url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    
    try:
        # 開啟遠端數據集 (不下載整個檔案，僅建立數據索引)
        ds = xr.open_dataset(tds_url, decode_times=True)
        
        # 篩選最新時間點、海表面 (depth=0)、台灣周邊範圍
        # 為了效能，範圍設定在經度 118~124，緯度 20~27
        latest_subset = ds.sel(
            time=ds.time[-1],           # 抓取最後一筆 (最新預報)
            depth=0,                   # 海表面
            lon=slice(118, 124),       # 台灣經度範圍
            lat=slice(20, 27)          # 台灣緯度範圍
        ).load() # 僅將這部分小範圍資料 load 進記憶體
        
        # 轉換為原系統所需的 NumPy 陣列格式
        lat_arr = latest_subset.lat.values
        lon_arr = latest_subset.lon.values
        u_arr = latest_subset.water_u.values
        v_arr = latest_subset.water_v.values
        
        # 處理陸地 NaN 值：HYCOM 的陸地格點為 NaN，將其轉為 0 避免計算錯誤
        u_arr = np.nan_to_num(u_arr)
        v_arr = np.nan_to_num(v_arr)
        
        # 取得資料時間戳記
        data_ts = pd.to_datetime(latest_subset.time.values).strftime('%Y-%m-%d %H:%M')
        
        return lat_arr, lon_arr, u_arr, v_arr, data_ts
    
    except Exception as e:
        st.error(f"連線 HYCOM 伺服器失敗: {e}")
        return None, None, None, None, None

# 執行資料抓取
lat, lon, u, v, data_time = get_hycom_realtime_data()

if lat is not None:
    st.success(f"已成功獲取 HYCOM 實時海象數據 (數據更新時間: {data_time})")
    
    # 建立網格
    LON, LAT = np.meshgrid(lon, lat)

    # ------------------------------------------
    # 台灣陸地遮罩與禁止區 (維持原邏輯)
    # ------------------------------------------
    # 粗略台灣形狀（橢圓）
    center_lat, center_lon = 23.7, 121.0
    a, b = 1.6, 0.8
    land_mask = (((LAT - center_lat) / a) ** 2 + ((LON - center_lon) / b) ** 2) < 1

    # 外擴 5 km 緩衝區
    grid_km = 111 * (lat[1] - lat[0])
    expand_cells = int(5 / grid_km)
    dist = distance_transform_edt(~land_mask)
    buffer_mask = dist <= expand_cells
    
    # 最終導航禁止區
    forbidden = land_mask | buffer_mask

    # ------------------------------------------
    # A* 搜尋演算法 (維持原邏輯)
    # ------------------------------------------
    def heuristic(a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def flow_cost(i, j, ni, nj):
        dx = lon[nj] - lon[j]
        dy = lat[ni] - lat[i]
        move_vec = np.array([dx, dy])
        move_vec = move_vec / (np.linalg.norm(move_vec)+1e-6)
        flow_vec = np.array([u[i,j], v[i,j]])
        assist = np.dot(move_vec, flow_vec)
        base = np.hypot(dx, dy)
        # 利用海流輔助降低成本 (assist > 0 代表順流)
        return base * (1 - 0.6 * assist)

    def astar(start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came = {}
        g = {start: 0}
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came:
                    path.append(current)
                    current = came[current]
                path.append(start)
                return path[::-1]

            for d in dirs:
                ni, nj = current[0] + d[0], current[1] + d[1]
                if ni < 0 or nj < 0 or ni >= len(lat) or nj >= len(lon):
                    continue
                if forbidden[ni, nj]:
                    continue

                new_cost = g[current] + flow_cost(current[0], current[1], ni, nj)
                nxt = (ni, nj)
                if nxt not in g or new_cost < g[nxt]:
                    g[nxt] = new_cost
                    f = new_cost + heuristic(nxt, goal)
                    heapq.heappush(open_set, (f, nxt))
                    came[nxt] = current
        return None

    # ------------------------------------------
    # UI 輸入部分
    # ------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Start Point")
        start_lat = st.number_input("起點緯度", value=22.3, step=0.1, format="%.2f")
        start_lon = st.number_input("起點經度", value=120.0, step=0.1, format="%.2f")

    with col2:
        st.subheader("End Point")
        end_lat = st.number_input("終點緯度", value=25.2, step=0.1, format="%.2f")
        end_lon = st.number_input("終點經度", value=122.0, step=0.1, format="%.2f")

    # 座標轉格點索引
    def nearest_idx(lat0, lon0):
        i = np.argmin(np.abs(lat - lat0))
        j = np.argmin(np.abs(lon - lon0))
        return i, j

    start_idx = nearest_idx(start_lat, start_lon)
    goal_idx = nearest_idx(end_lat, end_lon)

    # ------------------------------------------
    # 路徑計算
    # ------------------------------------------
    t0 = time.time()
    path = astar(start_idx, goal_idx)
    elapsed = time.time() - t0

    st.write(f"⏱️ A* 優化路徑計算完成，耗時: {elapsed:.2f} 秒")

    # ------------------------------------------
    # 結果視覺化
    # ------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 設定顯示範圍
    ax.set_extent([118.5, 123.5, 21.0, 26.5])
    
    # 專業底圖設定
    ax.add_feature(cfeature.LAND, facecolor='#444444', zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor='#1a1a1a')

    # 繪製 HYCOM 實時海流向量
    # 使用 quiver 繪製箭頭，[::6] 是抽樣步長，避免箭頭太擠
    q = ax.quiver(
        LON[::6, ::6], LAT[::6, ::6], 
        u[::6, ::6], v[::6, ::6], 
        color='cyan', alpha=0.4, scale=20, zorder=3
    )

    # 繪製優化路徑
    if path:
        py = [lat[p[0]] for p in path]
        px = [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3, label='Optimized Path (HYCOM)', zorder=5)
        
        # 繪製直線對比 (原始路徑)
        ax.plot([start_lon, end_lon], [start_lat, end_lat], color='white', 
                linestyle='--', alpha=0.5, label='Original (Direct) Path', zorder=4)

    # 標示點位
    ax.scatter(start_lon, start_lat, color='#00FF00', s=120, label='Departure', zorder=6, edgecolors='black')
    ax.scatter(end_lon, end_lat, color='#FFFF00', s=120, label='Destination', zorder=6, edgecolors='black')

    ax.legend(loc='lower right', frameon=True)
    ax.set_title(f"HYCOM Real-time Route Optimization\n(Data Time: {data_time})", color='black', fontsize=14)

    st.pyplot(fig)

else:
    st.error("無法載入海象數據，請確認伺服器狀態或網路連線。")
