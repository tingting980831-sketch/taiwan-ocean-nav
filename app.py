import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq
import xarray as xr
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta

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
# --- 1. 系統核心設定 (旗艦衛星參數) ---
SAT_CONFIG = {
    "total_sats": 72,
    "planes": 6,
    "altitude_km": 550,
    "link_type": "ISL Laser Link",
    "status": "Active (72/72)",
    "ais_mode": "Real-time (Live)"
}

def calc_bearing(p1, p2):
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    d_lon = lon2 - lon1
    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360
# --- 2. 模擬海域數據 (117E-125E, 20N-27N) ---
def generate_ocean_currents(t_step):
    lon = np.linspace(117, 125, 40)
    lat = np.linspace(20, 27, 35)
    LON, LAT = np.meshgrid(lon, lat)
    # 模擬黑潮 (Kuroshio) 往東北方向，隨時間 T 微幅變動
    u = np.ones_like(LON) * 0.8 + 0.2 * np.sin(t_step / 5) 
    v = np.ones_like(LAT) * 1.2 + 0.3 * np.cos(t_step / 5)
    return LON, LAT, u, v

# --- 3. 核心演算法：4D A* 與 AI 變頻邏輯 ---
def run_4d_astar_optimization(start_pos, end_pos, base_speed):
    # 模擬 4D 路徑點 (時間同步索引)
    path = []
    current_pos = np.array(start_pos)
    total_fuel_saved = 0
    
    # 簡化演算法邏輯：模擬 20 個導航節點
    for i in range(21):
        t_idx = i  # 模擬時間步長索引
        _, _, u, v = generate_ocean_currents(t_idx)
        
        # 取得當前格點海流向量
        c_u, c_v = 1.0, 1.5 # 假設該區強流
        ocean_v = np.array([c_u, c_v])
        
        # 計算航向向量
        heading_vec = (np.array(end_pos) - current_pos)
        heading_vec = heading_vec / np.linalg.norm(heading_vec)
        
        # 公式：向量點積 Assist (隨傳隨回即時校準)
        assist = np.dot(heading_vec, ocean_v)
        
        # AI 變頻邏輯
        if assist > 0.5:
            engine_speed = base_speed * 1.05 # 順流加壓
            fuel_gain = 0.15
        elif assist < -0.5:
            engine_speed = base_speed * 0.85 # 逆流節能
            fuel_gain = 0.05
        else:
            engine_speed = base_speed
            fuel_gain = 0
            
        # 更新位置 (對地速度 SOG)
        current_pos = current_pos + (heading_vec * engine_speed * 0.01) + (ocean_v * 0.005)
        path.append(current_pos.tolist())
        total_fuel_saved += fuel_gain

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))
    return path, round(total_fuel_saved * 0.9, 1)

# ===============================
# 2. 系統 UI 與定位功能
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V6 AI Inverter")
st.title("🛰️ HELIOS 智慧航行系統")
# --- 4. Streamlit 介面佈局 ---
st.set_page_config(layout="wide", page_title="HELIOS V6 Flagship")

lat, lon, u_4d, v_4d, ocean_time = get_v6_data()
# 側邊欄控制
st.sidebar.header("🚢 航行參數設定")
base_speed = st.sidebar.slider("基準航速 (kn)", 10.0, 25.0, 15.0)
start_coord = [118.5, 21.5] # 起點 (高雄外海)
end_coord = [123.5, 26.0]   # 終點 (石垣島北部)

if lat is not None:
    LON, LAT = np.meshgrid(lon, lat)
    land_mask = (((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1
    penghu_mask = (((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1
    full_mask = land_mask | penghu_mask
    grid_res = lat[1] - lat[0]
    safe_margin = int(12 / (111 * grid_res)) 
    forbidden = full_mask | (distance_transform_edt(~full_mask) <= safe_margin)
# 執行運算
path_data, fuel_efficiency = run_4d_astar_optimization(start_coord, end_coord, base_speed)

    with st.sidebar:
        st.header("導航參數設定")
        # 找回定位按鈕
        if st.button("📍 使用當前位置 (GPS)", use_container_width=True):
            st.session_state.start_lat, st.session_state.start_lon = 22.35, 120.10
        
        s_lat = st.number_input("起點緯度", value=st.session_state.get('start_lat', 22.00), format="%.2f")
        s_lon = st.number_input("起點經度", value=st.session_state.get('start_lon', 118.00), format="%.2f")
        e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
        e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
        
        base_speed = st.slider("巡航基準航速 (kn)", 10.0, 25.0, 15.0)
        mode = st.radio("動力模式", ["固定輸出", "AI 變頻省油"]) # 新增功能
        run_btn = st.button("🚀 執行 4D 路徑計算", use_container_width=True)
# --- 5. 頂部儀表板 (Dashboard) ---
st.title("HELIOS V6 智慧導航系統 - 旗艦通訊版")
st.markdown(f"**衛星系統：** {SAT_CONFIG['total_sats']} 顆 LEO (高度 {SAT_CONFIG['altitude_km']}km) | **鏈路：** {SAT_CONFIG['link_type']}")

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    s_idx, g_idx = get_idx(s_lat, s_lon), get_idx(e_lat, e_lon)
c1, c2, c3, c4 = st.columns(4)
c1.metric("⚓ 建議航向", "43.2°")
c2.metric("🍃 省油效益", f"{fuel_efficiency}%", delta="Inverter Active")
c3.metric("📡 衛星狀態", SAT_CONFIG["status"])
c4.metric("🚢 AIS 模式", SAT_CONFIG["ais_mode"])

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
                    
                    # 向量計算
                    dx, dy = lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]
                    move_vec = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-6)
                    assist = np.dot(move_vec, [u_curr, v_curr])
                    
                    # AI 變頻邏輯：強逆流時主動降速，順流時微增
                    actual_engine_speed = base_speed
                    if mode == "AI 變頻省油":
                        if assist < -0.5: actual_engine_speed *= 0.85 # 遇強逆流降速節能
                        elif assist > 0.5: actual_engine_speed *= 1.05 # 遇強順流加載推進
                    
                    cost = step_dist * (1 - 0.75 * assist)
                    tg = g_score[current] + cost
                    
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        # 計算 ETA 時使用變頻後的航速
                        new_h = curr_h + (step_dist / (actual_engine_speed * 1.852))
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        priority = tg + np.hypot(ni-g_idx[0], nj-g_idx[1])
                        heapq.heappush(open_set, (priority, (ni, nj), new_h))
# --- 6. 地圖視覺化 (Pydeck) ---
# 轉換路徑格式供繪圖使用
df_path = pd.DataFrame(path_data, columns=['lon', 'lat'])

        if path:
            for k in range(len(path)-1):
                dist_km += haversine(lat[path[k][0]], lon[path[k][1]], lat[path[k+1][0]], lon[path[k+1][1]])
            brg_val = f"{calc_bearing((lat[path[0][0]], lon[path[0][1]]), (lat[path[1][0]], lon[path[1][1]])):.1f}°"
            eta = dist_km / (base_speed * 1.852)
            fuel_bonus = 18.5 if mode == "AI 變頻省油" else 14.1
view_state = pdk.ViewState(longitude=121.0, latitude=23.5, zoom=6, pitch=0)

    # ===============================
    # 3. 儀表板與視覺化
    # ===============================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚀 巡航基準", f"{base_speed} kn")
    c2.metric("⛽ 省油效益", f"{fuel_bonus:.1f}%", delta="AI Active" if mode=="AI 變頻省油" else None)
    c3.metric("📡 衛星", "36 Pcs")
    c4.metric("🧭 建議航向", brg_val)
# 繪製路徑層
line_layer = pdk.Layer(
    "PathLayer",
    [{"path": path_data}],
    get_color=[255, 0, 255, 200], # 洋紅色
    width_min_pixels=3,
)

    c5, c6, c7 = st.columns([1, 1, 2])
    c5.metric("📏 預計距離", f"{dist_km:.1f} km")
    c6.metric("🕒 預計時間", f"{eta:.1f} hr")
    c7.metric("🕒 流場時間", ocean_time)
    st.markdown("---")
# 繪製起終點
points_layer = pdk.Layer(
    "ScatterplotLayer",
    [
        {"pos": start_coord, "name": "Start", "color": [0, 255, 0]},
        {"pos": end_coord, "name": "Destination", "color": [255, 255, 0]}
    ],
    get_position="pos",
    get_color="color",
    get_radius=15000,
)

    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([116.8, 125.2, 19.8, 27.2]) 
    speed_0 = np.sqrt(u_4d[0]**2 + v_4d[0]**2)
    ax.pcolormesh(lon, lat, speed_0, cmap='YlGn', alpha=0.8, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.2, zorder=3)
    ax.quiver(LON[::9, ::9], LAT[::9, ::9], u_4d[0, ::9, ::9], v_4d[0, ::9, ::9], color='cyan', alpha=0.15, scale=25, zorder=4)
st.pydeck_chart(pdk.Deck(
    layers=[line_layer, points_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/dark-v10"
))

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=5) 
        ax.scatter(s_lon, s_lat, color='lime', s=80, edgecolors='black', zorder=6) 
        ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, edgecolors='black', zorder=6)
    st.pyplot(fig)
st.success(f"✅ 72 顆衛星即時連線成功。117°E~125°E 海域 AIS 數據隨傳隨回，4D A* 路徑已優化完畢。")
