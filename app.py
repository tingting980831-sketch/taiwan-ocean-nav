import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt
import plotly.graph_objects as go
import requests

# 設定頁面
st.set_page_config(layout="wide", page_title="HELIOS V6 智慧導航")
st.title("🛰️ HELIOS V6 智慧導航控制台")

# ===============================
# 1. 數據獲取與處理
# ===============================

@st.cache_data(ttl=3600)
def load_hycom_data():
    """獲取 HYCOM 海流數據"""
    # 這裡使用 2026 模擬路徑或實際可用 OpenDAP URL
    # 注意：實際部署時需確保 URL 有效，此處保留原邏輯
    try:
        url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs.get('time_origin', '2026-01-01'))
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        
        lat_slice = slice(21, 26)
        lon_slice = slice(118, 124)
        u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        
        lons = u_data['lon'].values
        lats = u_data['lat'].values
        u_val = np.nan_to_num(u_data.values)
        v_val = np.nan_to_num(v_data.values)
        land_mask = np.isnan(u_data.values)
        return lons, lats, u_val, v_val, land_mask, latest_time
    except Exception as e:
        st.error(f"數據加載失敗: {e}")
        return None, None, None, None, None, None

@st.cache_data(ttl=1800)
def get_marine_weather(lat, lon):
    """獲取 Open-Meteo 海象與風速"""
    # 預設值
    current_wave, wind_u, wind_v = 0.5, 0.0, 0.0
    try:
        # 波高
        wave_res = requests.get("https://marine-api.open-meteo.com/v1/marine", 
                                params={"latitude": lat, "longitude": lon, "hourly": ["wave_height"], "forecast_days": 1}).json()
        current_wave = float(wave_res.get('hourly', {}).get('wave_height', [0.5])[0])
        
        # 風速
        wind_res = requests.get("https://api.open-meteo.com/v1/forecast", 
                                params={"latitude": lat, "longitude": lon, "hourly": ["wind_speed_10m", "wind_direction_10m"], "forecast_days": 1}).json()
        ws = float(wind_res.get('hourly', {}).get('wind_speed_10m', [0])[0])
        wd = float(wind_res.get('hourly', {}).get('wind_direction_10m', [0])[0])
        wind_u = ws * np.sin(np.deg2rad(wd))
        wind_v = ws * np.cos(np.deg2rad(wd))
    except:
        pass
    return current_wave, wind_u, wind_v

# ===============================
# 2. 導航演算法 (A*)
# ===============================

def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()
    if not land_mask[lat_idx, lon_idx]:
        return lat_idx, lon_idx
    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]] - lat)**2 + (lons[ocean[1]] - lon)**2)
    i = dist.argmin()
    return ocean[0][i], ocean[1][i]

def astar_navigation(start, goal, u, v, land_mask, safety_map, ship_spd, wave, wind_mag):
    v_ship = ship_spd * 0.277  # km/h to m/s
    rows, cols = land_mask.shape
    pq = [(0, start)]
    cost = {start: 0}
    came_from = {}
    
    while pq:
        _, curr = heapq.heappop(pq)
        if curr == goal: break
        
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            ni, nj = curr[0] + di, curr[1] + dj
            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni, nj]:
                # 基礎距離 (約 8km 每格)
                dist_step = np.sqrt(di**2 + dj**2) * 8000
                
                # 海流影響
                flow_vec = np.array([u[curr], v[curr]])
                dir_vec = np.array([dj, di]) / np.sqrt(di**2 + dj**2)
                flow_eff = np.dot(flow_vec, dir_vec)
                
                v_ground = max(0.5, v_ship + flow_eff)
                # 基本時間 + 環境阻力
                step_cost = (dist_step / v_ground) + (wave * 200) + (wind_mag * 50)
                
                # 岸邊安全避讓
                if safety_map[ni, nj] < 3:
                    step_cost += 20000 / (safety_map[ni, nj] + 0.1)
                
                new_cost = cost[curr] + step_cost
                if (ni, nj) not in cost or new_cost < cost[(ni, nj)]:
                    cost[(ni, nj)] = new_cost
                    priority = new_cost + 1.2 * (np.sqrt((ni-goal[0])**2 + (nj-goal[1])**2) * 8000 / v_ship)
                    heapq.heappush(pq, (priority, (ni, nj)))
                    came_from[(ni, nj)] = curr
                    
    path = []
    curr = goal
    while curr in came_from:
        path.append(curr); curr = came_from[curr]
    return path[::-1]

# ===============================
# 3. 介面佈局
# ===============================

lons, lats, u_val, v_val, land_mask, obs_time = load_hycom_data()

with st.sidebar:
    st.header("⚙️ 航行參數")
    s_lon = st.number_input("起點經度", 118.0, 124.0, 120.3)
    s_lat = st.number_input("起點緯度", 21.0, 26.0, 22.6)
    e_lon = st.number_input("終點經度", 118.0, 124.0, 122.0)
    e_lat = st.number_input("終點緯度", 21.0, 26.0, 24.5)
    ship_speed = st.slider("巡航船速 (km/h)", 5, 60, 25)
    st.divider()
    st.info("系統將自動根據當前海象計算最優路徑。")

if lons is not None:
    # 預處理
    safety_map = distance_transform_edt(~land_mask)
    start_node = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
    goal_node = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)
    
    wave_h, wu, wv = get_marine_weather((s_lat+e_lat)/2, (s_lon+e_lon)/2)
    wind_mag = np.sqrt(wu**2 + wv**2)
    
    # 計算路徑
    path = astar_navigation(start_node, goal_node, u_val, v_val, land_mask, safety_map, ship_speed, wave_h, wind_mag)

    # 儀表板
    col1, col2, col3, col4 = st.columns(4)
    if path:
        dist_km = sum(np.sqrt((lats[path[i][0]]-lats[path[i+1][0]])**2 + (lons[path[i][1]]-lons[path[i+1][1]])**2) for i in range(len(path)-1)) * 111
        col1.metric("預估耗時", f"{dist_km/ship_speed:.1f} hr")
        col2.metric("總航程", f"{dist_km:.1f} km")
    col3.metric("當前波高", f"{wave_h:.1f} m")
    col4.metric("環境風速", f"{wind_mag:.1f} m/s")

    # --- 2D 地圖展示 ---
    st.subheader("🗺️ 航路規劃平面圖")
    fig2d = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118, 124, 21, 26])
    ax.add_feature(cfeature.LAND, facecolor='#dddddd')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    # 底圖顯示海流強度
    mag = np.sqrt(u_val**2 + v_val**2)
    im = ax.pcolormesh(lons, lats, mag, cmap='YlGnBu_r', alpha=0.7, shading='auto')
    plt.colorbar(im, label="海流流速 (m/s)", pad=0.02)
    
    if path:
        px = [lons[p[1]] for p in path]
        py = [lats[p[0]] for p in path]
        ax.plot(px, py, color='red', linewidth=2.5, label="最優航線")
    ax.scatter([s_lon, e_lon], [s_lat, e_lat], color=['green', 'gold'], s=100, edgecolors='black', zorder=5)
    st.pyplot(fig2d)

    # --- 3D 多層海象模型 (核心修改部分) ---
    st.divider()
    st.subheader("🌊 HELIOS 3D 分層動態模型")
    st.caption("視圖層次：頂層(風速) -> 中層(海流) -> 底層(波高)")

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 定義垂直分層基準高度
    Z_WAVE = 0
    Z_FLOW = 5
    Z_WIND = 10

    fig3d = go.Figure()

    # 1. 底層：波高曲面 (Surface)
    fig3d.add_trace(go.Surface(
        x=lon_grid, y=lat_grid, z=np.full_like(lon_grid, Z_WAVE) + (wave_h * 0.3),
        colorscale='Viridis', opacity=0.7, name="底層波高",
        colorbar=dict(title="波高", x=-0.1)
    ))

    # 2. 中層：海流 (曲面顯示強度 + 箭頭顯示流向)
    fig3d.add_trace(go.Surface(
        x=lon_grid, y=lat_grid, z=np.full_like(lon_grid, Z_FLOW) + mag,
        colorscale='Blues', opacity=0.5, showscale=False, name="中層海流"
    ))
    
    skip = 3
    fig3d.add_trace(go.Cone(
        x=lon_grid[::skip, ::skip].flatten(),
        y=lat_grid[::skip, ::skip].flatten(),
        z=np.full_like(lon_grid[::skip, ::skip], Z_FLOW + 1.5).flatten(),
        u=u_val[::skip, ::skip].flatten(),
        v=v_val[::skip, ::skip].flatten(),
        w=np.zeros_like(u_val[::skip, ::skip].flatten()),
        sizemode="scaled", sizeref=0.8, colorscale='Blues', showscale=False, name="流向"
    ))

    # 3. 頂層：風速箭頭 (Cone)
    fig3d.add_trace(go.Cone(
        x=lon_grid[::skip, ::skip].flatten(),
        y=lat_grid[::skip, ::skip].flatten(),
        z=np.full_like(lon_grid[::skip, ::skip], Z_WIND).flatten(),
        u=np.full_like(u_val[::skip, ::skip], wu).flatten(),
        v=np.full_like(v_val[::skip, ::skip], wv).flatten(),
        w=np.zeros_like(u_val[::skip, ::skip].flatten()),
        sizemode="scaled", sizeref=1.5, anchor="tail", colorscale='Reds', name="頂層風向"
    ))

    # 4. 航路投影 (懸浮在最頂層)
    if path:
        fig3d.add_trace(go.Scatter3d(
            x=[lons[p[1]] for p in path],
            y=[lats[p[0]] for p in path],
            z=np.full(len(path), Z_WIND + 2),
            mode='lines', line=dict(color='yellow', width=8), name="導航軌跡"
        ))

    fig3d.update_layout(
        scene=dict(
            xaxis_title="經度", yaxis_title="緯度", zaxis_title="分層等級",
            zaxis=dict(tickvals=[Z_WAVE, Z_FLOW, Z_WIND], ticktext=["波浪層", "海流層", "大氣層"]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
        ),
        height=800, margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig3d, use_container_width=True)

else:
    st.warning("暫時無法連結 HYCOM 伺服器，請檢查網路或稍後再試。")
