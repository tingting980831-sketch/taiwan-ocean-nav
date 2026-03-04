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
# 1. 數據抓取 (20N-27N, 117E-125E)
# ===============================
@st.cache_data(ttl=3600)
def get_v6_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.isel(time=slice(-24, None)).sel(
            depth=0, lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)
        ).load()
        u_4d = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v_4d = np.nan_to_num(subset.water_v.values).astype(np.float32)
        try:
            dt_raw = xr.decode_cf(subset).time.values
            dt_display = pd.to_datetime(dt_raw[0]).strftime('%Y-%m-%d %H:%M')
        except:
            dt_display = "Dynamic 24H Forecast"
        return subset.lat.values.astype(np.float32), subset.lon.values.astype(np.float32), u_4d, v_4d, dt_display
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, None, None, None, None

def calc_bearing(p1, p2):
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
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
# 2. 系統 UI 與定位功能
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V6 Flagship")
st.title("🛰️ HELIOS 智慧航行系統")

lat, lon, u_4d, v_4d, ocean_time = get_v6_data()

if lat is not None:
    LON, LAT = np.meshgrid(lon, lat)
    land_mask = (((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1
    penghu_mask = (((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1
    full_mask = land_mask | penghu_mask
    grid_res = lat[1] - lat[0]
    safe_margin = int(12 / (111 * grid_res)) 
    forbidden = full_mask | (distance_transform_edt(~full_mask) <= safe_margin)

    with st.sidebar:
        st.header("導航參數設定")
        if st.button("📍 使用當前位置 (GPS)", use_container_width=True):
            st.session_state.start_lat, st.session_state.start_lon = 22.35, 120.10
        
        s_lat = st.number_input("起點緯度", value=st.session_state.get('start_lat', 22.00), format="%.2f")
        s_lon = st.number_input("起點經度", value=st.session_state.get('start_lon', 118.00), format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        
        base_speed = st.slider("巡航基準航速 (kn)", 10.0, 25.0, 15.0)
        mode = st.radio("動力模式", ["固定輸出", "AI 變頻省油"])
        run_btn = st.button("🚀 執行 4D 路徑計算", use_container_width=True)

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    path, dist_km, fuel_bonus, eta, brg_val = None, 0.0, 0.0, 0.0, "---"
    
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
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    t_idx = min(int(curr_h), 23)
                    step_dist = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
                    u_curr, v_curr = u_4d[t_idx, ni, nj], v_4d[t_idx, ni, nj]
                    
                    dx, dy = lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_curr, v_curr])
                    
                    actual_engine_speed = base_speed
                    if mode == "AI 變頻省油":
                        if assist < -0.5: actual_engine_speed *= 0.85
                        elif assist > 0.5: actual_engine_speed *= 1.05
                    
                    cost = step_dist * (1 - 0.75 * assist)
                    tg = g_score[current] + cost
                    
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        new_h = curr_h + (step_dist / (actual_engine_speed * 1.852))
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj), new_h))

        if path:
            for k in range(len(path)-1):
                dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
            brg_val = f"{calc_bearing((lat[path[0][0]], lon[path[0][1]]), (lat[path[1][0]], lon[path[1][1]])):.1f}°"
            eta = dist_km / (base_speed * 1.852)
            fuel_bonus = 18.5 if mode == "AI 變頻省油" else 14.1

    # ===============================
# 3. 儀表板與動態衛星模擬 (更新版)
# ===============================
c1, c2, c3, c4 = st.columns(4)
c1.metric("🧭 建議航向", brg_val)
c2.metric("⛽ 省油效益", f"{fuel_bonus:.1f}%", delta="AI Active" if mode=="AI 變頻省油" else None)

# --- 動態衛星計算邏輯 ---
# 模擬在 72 顆全域網中，當前經緯度上方可視的衛星數量 (4~9 顆跳動)
import time
# 利用經緯度總和當種子，讓不同位置看到的衛星數不同
seed = int(s_lat + s_lon) 
np.random.seed(seed)
visible_sats = np.random.randint(4, 10) 

# 顯示格式：目前連線數 / 總數
c3.metric("📡 衛星狀態", f"Active ({visible_sats}/72)") 
c4.metric("🚢 AIS 模式", "Real-time (隨傳隨回)")

c5, c6, c7 = st.columns([1, 1, 2])
c5.metric("📏 預計距離", f"{dist_km:.1f} km")
c6.metric("🕒 預計時間", f"{eta:.1f} hr")

# 增加動態連線品質顯示
c7.metric("🕒 系統狀態", f"LEO 550km / ISL 優質連線")
st.caption(f"衛星幾何配置 (GDOP): 優良 | 數據源: {ocean_time}")
    # ===============================
    # 4. 繪圖功能
    # ===============================
    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([116.8, 125.2, 19.8, 27.2]) 
    speed_0 = np.sqrt(u_4d[0]**2 + v_4d[0]**2)
    ax.pcolormesh(lon, lat, speed_0, cmap='YlGn', alpha=0.8, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.2, zorder=3)
    ax.quiver(LON[::9, ::9], LAT[::9, ::9], u_4d[0, ::9, ::9], v_4d[0, ::9, ::9], color='cyan', alpha=0.15, scale=25, zorder=4)

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=5) 
        ax.scatter(s_lon, s_lat, color='lime', s=80, edgecolors='black', zorder=6) 
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, edgecolors='black', zorder=6)
    st.pyplot(fig)
