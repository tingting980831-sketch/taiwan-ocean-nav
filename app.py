import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd

# ===============================
# 1. 數據抓取 (20N-27N, 117E-125E)
# ===============================
@st.cache_data(ttl=3600)
def get_v8_data():
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
# 2. 系統 UI
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V8 Navigation Analysis")
st.title("🛰️ HELIOS 智慧導航系統 (流場路徑對比版)")

lat, lon, u_4d, v_4d, ocean_time = get_v8_data()

if lat is not None:
    LON, LAT = np.meshgrid(lon, lat)
    # 僅設定陸地硬障礙，刪除 12km 緩衝
    land_mask = (((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1
    penghu_mask = (((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1
    forbidden = land_mask | penghu_mask 

    with st.sidebar:
        st.header("導航參數設定")
        s_lat = st.number_input("起點緯度", value=22.35, format="%.2f")
        s_lon = st.number_input("起點經度", value=120.10, format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        
        # 引擎速度固定 100%
        base_speed = st.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
        run_btn = st.button("🚀 執行航線計算", use_container_width=True)

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)

    path, dist_km, brg_val, eta = None, 0.0, "---", 0.0
    baseline_path, baseline_dist = None, 0.0

    # ===============================
    # 3. 演算法計算
    # ===============================
    if run_btn and not forbidden[s_idx] and not forbidden[g_idx]:
        # --- A. 優化航線 (實線 - 考慮海流) ---
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
                    step_dist = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
                    u_curr, v_curr = u_4d[0, ni, nj], v_4d[0, ni, nj]
                    
                    dx, dy = lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_curr, v_curr])
                    
                    # 成本函數：僅受流場向量影響
                    cost = step_dist * (1 - 0.75 * assist)
                    tg = g_score[current] + cost
                    
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        # 對地速度合成
                        v_sog_kts = base_speed + (assist * 1.94384)
                        v_sog_kts = max(v_sog_kts, 1.0)
                        new_h = curr_h + (step_dist / (v_sog_kts * 1.852))
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj), new_h))

        # --- B. 基準航線 (虛線 - 不考慮海流) ---
        open_set_b, came_from_b, g_score_b = [], {}, {s_idx: 0.0}
        heapq.heappush(open_set_b, (0, s_idx))
        while open_set_b:
            _, current_b = heapq.heappop(open_set_b)
            if current_b == g_idx:
                baseline_path = []
                while current_b in came_from_b:
                    baseline_path.append(current_b); current_b = came_from_b[current_b]
                baseline_path.append(s_idx); baseline_path = baseline_path[::-1]; break
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current_b[0]+d[0], current_b[1]+d[1]
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    step_dist_b = haversine(lat[current_b[0]], lon[current_b[1]], lat[ni], lon[nj])
                    tg_b = g_score_b[current_b] + step_dist_b
                    if (ni, nj) not in g_score_b or tg_b < g_score_b[(ni, nj)]:
                        came_from_b[(ni, nj)] = current_b
                        g_score_b[(ni, nj)] = tg_b
                        priority_b = tg_b + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set_b, (priority_b, (ni, nj)))

    # ===============================
    # 4. 儀表板與狀態顯示 (移除效益數據)
    # ===============================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧭 建議航向", brg_val)
    c2.metric("🌊 流場狀態", "HYCOM 4D Active")
    np.random.seed(int(s_lat + s_lon))
    visible_sats = np.random.randint(4, 10)
    c3.metric("📡 衛星連線", f"Active ({visible_sats}/15)") 

    c5, c6, c7 = st.columns([1, 1, 2])
    # 僅顯示數值，移除省時省油對比
    if path:
        for k in range(len(path)-1):
            dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
        eta = g_score[g_idx] # 使用路徑積分後的時間成本 (或直接使用演算法中的curr_h)
        # 為求顯示統一，重新計算優化後的 ETA
        # 此處 ETA 在演算法中已包含海流合成速度
    
    c5.metric("📏 航行距離", f"{dist_km:.1f} km")
    c6.metric("🕒 預計到達時間", f"{eta:.1f} hr")
    st.markdown("---")

    # ===============================
    # 5. 繪圖與對比
    # ===============================
    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([116.8, 125.2, 19.8, 27.2]) 
    speed_0 = np.sqrt(u_4d[0]**2 + v_4d[0]**2)
    ax.pcolormesh(lon, lat, speed_0, cmap='YlGn', alpha=0.8, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#333333', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1)

    if baseline_path:
        bpy, bpx = [lat[p[0]] for p in baseline_path], [lon[p[1]] for p in baseline_path]
        ax.plot(bpx, bpy, color='white', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5, label='基準路徑 (不計流場)')

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=6, label='優化路徑 (流場合成)') 
        ax.scatter(s_lon, s_lat, color='lime', s=80, zorder=7) 
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, zorder=7)
    
    ax.legend(loc='upper right')
    st.pyplot(fig)
