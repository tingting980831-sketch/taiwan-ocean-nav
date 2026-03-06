import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr

# ===============================
# 1. 系統核心設定
# ===============================
SAT_CONFIG = {
    "total_sats": 72,
    "planes": 6,
    "altitude_km": 550,
    "link_type": "ISL Laser Link",
    "status": "Active (72/72)",
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
# 2. 介面與數據讀取
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰️ HELIOS V6 智慧導航系統")

lat, lon, u_4d, v_4d = get_v6_data()

if lat is not None:
    u_now = u_4d[0]
    v_now = v_4d[0]
    forbidden = np.isnan(u_now) | (u_now == 0)

    # --- 重要：將變數定義移出計算區塊 ---
    with st.sidebar:
        st.header("🚢 導航參數設定")
        if st.button("📍 使用高雄港 (GPS)", use_container_width=True):
            st.session_state.start_lat, st.session_state.start_lon = 22.60, 120.27
        
        s_lat = st.number_input("起點緯度", value=st.session_state.get('start_lat', 22.00), format="%.2f")
        s_lon = st.number_input("起點經度", value=st.session_state.get('start_lon', 118.00), format="%.2f")
        e_lat = st.number_input("終點緯度", value=23.98, format="%.2f")
        e_lon = st.number_input("終點經度", value=121.63, format="%.2f")
        
        # 這裡定義了 base_speed 和 mode
        base_speed = st.slider("巡航基準航速 (kn)", 10.0, 25.0, 15.0)
        mode = st.radio("動力模式", ["固定輸出", "AI 變頻省油"])
        run_btn = st.button("🚀 執行航線優化", use_container_width=True)

    path, dist_km, brg_val = None, 0.0, "---"
    
    if run_btn:
        si, sj = np.argmin(np.abs(lat-s_lat)), np.argmin(np.abs(lon-s_lon))
        gi, gj = np.argmin(np.abs(lat-e_lat)), np.argmin(np.abs(lon-e_lon))
        
        q = [(0, (si, sj), (0,0))]
        came_from, g_score = {(si, sj): None}, {(si, sj): 0.0}
        
        while q:
            _, curr, last_dir = heapq.heappop(q)
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
                    assist = np.dot(move_vec, [u_now[ni, nj], v_now[ni, nj]])
                    
                    # 使用剛才定義的 base_speed 和 mode
                    actual_speed = base_speed
                    if mode == "AI 變頻省油":
                        actual_speed *= (1.05 if assist > 0.5 else 0.85 if assist < -0.5 else 1.0)
                    
                    # 成本計算 (流場輔助 + 轉彎懲罰)
                    turn_penalty = 1.5 if d != last_dir and last_dir != (0,0) else 0
                    cost = step_d * (1 - 0.35 * assist) + turn_penalty
                    tg = g_score[curr] + cost
                    
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-gi, nj-gj) * 1.5
                        heapq.heappush(q, (priority, (ni, nj), d))
                        came_from[(ni, nj)] = curr

        if path:
            for k in range(len(path)-1):
                dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
            brg_val = f"{calc_bearing((lat[path[0][0]], lon[path[0][1]]), (lat[path[1][0]], lon[path[1][1]])):.1f}°"

    # ===============================
    # 3. 視覺化
    # ===============================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚓ 建議航向", brg_val)
    c2.metric("🍃 省油模式", "Active" if mode=="AI 變頻省油" else "Off")
    c3.metric("📏 總航程", f"{dist_km:.1f} km")
    c4.metric("🕒 預計航時", f"{dist_km/(base_speed*1.852):.1f} hr" if dist_km > 0 else "---")

    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118.0, 124.5, 20.5, 26.5])
    
    spd = np.sqrt(u_now**2 + v_now**2)
    ax.pcolormesh(lon, lat, np.ma.masked_where(forbidden, spd), cmap='GnBu', alpha=0.4)
    ax.add_feature(cfeature.LAND, facecolor='#151515')
    ax.add_feature(cfeature.COASTLINE, edgecolor='white')
    
    if path:
        ax.plot([lon[p[1]] for p in path], [lat[p[0]] for p in path], color='#FF00FF', lw=3)
        ax.scatter(s_lon, s_lat, color='lime', s=100)
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=300)

    st.pyplot(fig)
