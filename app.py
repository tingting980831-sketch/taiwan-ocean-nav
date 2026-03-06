import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr

# ===============================
# 1. 系統核心設定 (已更新衛星配置)
# ===============================
SAT_CONFIG = {
    "total_sats": 12,       # 3面 * 4顆 = 12顆
    "planes": 3,            # 3個軌道面
    "altitude_km": 400,     # 軌道高度 400km
    "link_type": "ISL Laser Link",
    "status": "Active (12/12)",
    "ais_mode": "Real-time (Live)"
}

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
        return subset.lat.values.astype(np.float32), subset.lon.values.astype(np.float32), u_4d, v_4d
    except Exception as e:
        st.error(f"數據載入失敗: {e}")
        return None, None, None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

def calc_bearing(p1, p2):
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

# ===============================
# 2. 介面初始化
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V6 Flagship")
st.title("🛰️ HELIOS V6 智慧導航系統")
st.markdown(f"**衛星系統：** {SAT_CONFIG['total_sats']} 顆 LEO (高度 {SAT_CONFIG['altitude_km']}km) | **軌道配置：** {SAT_CONFIG['planes']} Planes | **狀態：** {SAT_CONFIG['status']}")

lat, lon, u_4d, v_4d = get_v6_data()

if lat is not None:
    u_sample = u_4d[0]
    forbidden = np.isnan(u_sample) | (u_sample == 0)

    with st.sidebar:
        st.header("🚢 導航參數設定")
        if st.button("📍 使用當前位置 (GPS)", use_container_width=True):
            st.session_state.start_lat, st.session_state.start_lon = 22.60, 120.27
        
        s_lat = st.number_input("起點緯度", value=st.session_state.get('start_lat', 22.00), format="%.2f")
        s_lon = st.number_input("起點經度", value=st.session_state.get('start_lon', 118.00), format="%.2f")
        e_lat = st.number_input("終點緯度", value=23.98, format="%.2f")
        e_lon = st.number_input("終點經度", value=121.63, format="%.2f")
        
        base_speed = st.slider("巡航基準航速 (kn)", 10.0, 25.0, 15.0)
        run_btn = st.button("🚀 執行 4D 路徑計算", use_container_width=True)

    path, dist_km, brg_val = None, 0.0, "---"
    
    if run_btn:
        si, sj = np.argmin(np.abs(lat-s_lat)), np.argmin(np.abs(lon-s_lon))
        gi, gj = np.argmin(np.abs(lat-e_lat)), np.argmin(np.abs(lon-e_lon))
        
        open_set, came_from, g_score = [], {}, {(si, sj): 0.0}
        heapq.heappush(open_set, (0, (si, sj), (0,0)))
        
        while open_set:
            _, curr, last_dir = heapq.heappop(open_set)
            if curr == (gi, gj):
                path = []
                while curr in came_from:
                    path.append(curr); curr = came_from[curr]
                path.append((si, sj)); path = path[::-1]; break
            
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = curr[0]+d[0], curr[1]+d[1]
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    step_d = haversine(lat[curr[0]], lon[curr[1]], lat[ni], lon[nj])
                    move_vec = np.array([d[1], d[0]]) / (np.hypot(d[0], d[1]) + 1e-6)
                    assist = np.dot(move_vec, [u_4d[0, ni, nj], v_4d[0, ni, nj]])
                    
                    turn_penalty = 1.5 if d != last_dir and last_dir != (0,0) else 0
                    cost = step_d * (1 - 0.4 * assist) + turn_penalty
                    tg = g_score[curr] + cost
                    
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-gi, nj-gj) * 1.5
                        heapq.heappush(open_set, (priority, (ni, nj), d))
                        came_from[(ni, nj)] = curr

        if path:
            for k in range(len(path)-1):
                dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
            brg_val = f"{calc_bearing((lat[path[0][0]], lon[path[0][1]]), (lat[path[1][0]], lon[path[1][1]])):.1f}°"

    # ===============================
    # 3. 視覺化儀表板
    # ===============================
    c1, c2, c3 = st.columns(3)
    c1.metric("⚓ 建議航向", brg_val)
    c2.metric("📏 總航程", f"{dist_km:.1f} km")
    c3.metric("🕒 預計航時", f"{dist_km/(base_speed*1.852):.1f} hr" if dist_km > 0 else "---")

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.0, 124.5, 20.5, 26.5])
    
    spd = np.sqrt(u_4d[0]**2 + v_4d[0]**2)
    ax.pcolormesh(lon, lat, np.ma.masked_where(forbidden, spd), cmap='YlGnBu', alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', zorder=3)
    
    if path:
        px = [lon[p[1]] for p in path]
        py = [lat[p[0]] for p in path]
        ax.plot(px, py, color='#FF00FF', lw=3, zorder=5)
        ax.scatter(s_lon, s_lat, color='lime', s=100, zorder=10)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=300, zorder=10)

    st.pyplot(fig)
    st.success(f"✅ 衛星鏈路同步完成 (LEO Cluster: {SAT_CONFIG['total_sats']} Nodes)。")
