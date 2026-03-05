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
# 1. 本地數據讀取引擎 (NetCDF)
# ===============================
@st.cache_data
def get_local_hycom_data(t_str):
    base_dir = r"C:\NODASS\HYCOM\2024\03"
    file_name = f"hycom_glby_930_2024030612_{t_str}_uv3z_subscene.nc"
    full_path = os.path.join(base_dir, file_name)
    
    if not os.path.exists(full_path):
        st.error(f"❌ 找不到檔案: {full_path}")
        return None, None, None, None, None

    try:
        ds = xr.open_dataset(full_path)
        subset = ds.isel(depth=0).sel(
            lon=slice(117.0, 125.0), 
            lat=slice(20.0, 27.0)
        ).load()
        
        u_val = np.nan_to_num(subset.water_u.values[0]).astype(np.float32)
        v_val = np.nan_to_num(subset.water_v.values[0]).astype(np.float32)
        data_time = pd.to_datetime(subset.time.values[0]).strftime('%Y-%m-%d %H:%M')
        
        return subset.lat.values.astype(np.float32), subset.lon.values.astype(np.float32), u_val, v_val, data_time
    except Exception as e:
        st.error(f"讀取 NetCDF 失敗: {e}")
        return None, None, None, None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統 UI 與遮罩定義
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V21.1 Coastal Mask")
st.title("🛰️ HELIOS 智慧導航系統 (全域陸地遮罩版)")

with st.sidebar:
    st.header("📂 資料與任務設定")
    t_step = st.selectbox("預報時段", ["t000", "t003", "t006", "t009", "t012", "t015", "t018", "t021"])
    s_lat = st.number_input("起點緯度", value=21.50, format="%.2f")
    s_lon = st.number_input("起點經度", value=120.50, format="%.2f")
    e_lat = st.number_input("終點緯度", value=26.20, format="%.2f")
    e_lon = st.number_input("終點經度", value=123.50, format="%.2f")
    base_speed = st.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
    run_btn = st.button("🚀 執行避讓路徑計算", use_container_width=True)

lat, lon, u_2d, v_2d, data_time_label = get_local_hycom_data(t_step)

if lat is not None:
    LON, LAT = np.meshgrid(lon, lat)
    
    # --- 強化陸地遮罩 ---
    # 1. 台灣本島避讓
    mask_tw = (((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1
    # 2. 澎湖群島避讓
    mask_ph = (((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1
    # 3. 中國大陸沿岸遮罩 (線性邊界模擬：福建至浙江)
    # 定義一條從 (117, 24) 到 (121, 28) 的界線，界線左上方全部遮蔽
    mask_china = (LAT > (1.0 * LON - 93.5)) 
    
    forbidden = mask_tw | mask_ph | mask_china

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    path, dist_km, eta = None, 0.0, 0.0

    # ===============================
    # 3. A* 路徑優化
    # ===============================
    if run_btn:
        if forbidden[s_idx] or forbidden[g_idx]:
            st.error("❌ 警告：起點或終點位於陸地/中國大陸遮罩區內！")
        else:
            open_set, came_from, g_score = [], {}, {s_idx: 0.0}
            heapq.heappush(open_set, (0, s_idx))
            while open_set:
                _, current = heapq.heappop(open_set)
                if current == g_idx:
                    path = [current]
                    while current in came_from:
                        current = came_from[current]; path.append(current)
                    path = path[::-1]; break
                for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ni, nj = current[0]+d[0], current[1]+d[1]
                    if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                        step_d = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
                        u_f, v_f = u_2d[ni, nj], v_2d[ni, nj]
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
    # 4. 儀表板與繪圖
    # ===============================
    if path:
        for k in range(len(path)-1):
            dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
        eta = dist_km / (base_speed * 1.852)

    st.success(f"📊 同步時段: {t_step} | 數據基準: {data_time_label}")
    col1, col2, col3 = st.columns(3)
    col1.metric("📏 航行路徑總長", f"{dist_km:.1f} km")
    col2.metric("🕒 預計到達時間", f"{eta:.1f} hr")
    col3.metric("🚫 避讓狀態", "台灣/澎湖/中國沿岸")

    fig, ax = plt.subplots(figsize=(11, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8]) 
    
    speed = np.sqrt(u_2d**2 + v_2d**2)
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', alpha=0.6, shading='auto')
    
    # 繪製遮罩區域供視覺確認 (半透明灰色)
    ax.contourf(lon, lat, forbidden, levels=[0.5, 1], colors=['#555555'], alpha=0.3, zorder=4)

    ax.add_feature(cfeature.LAND, facecolor='#121212', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=6)

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=10, label='HELIOS 安全路徑') 
        ax.scatter(s_lon, s_lat, color='lime', s=100, zorder=11) 
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=250, zorder=11)
        ax.legend()
    
    st.pyplot(fig)
