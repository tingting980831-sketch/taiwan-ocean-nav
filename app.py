import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt
import plotly.graph_objects as go
import requests

# ===============================
# 0. 系統設定
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V6 智慧導航")
st.title("🛰️ HELIOS V6 智慧導航控制台")

# ===============================
# 1. 數據獲取模組
# ===============================

@st.cache_data(ttl=3600)
def load_hycom_data():
    """獲取 HYCOM 海流數據"""
    try:
        url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
        ds = xr.open_dataset(url, decode_times=False)
        time_origin = pd.to_datetime(ds['time'].attrs.get('time_origin', '2026-01-01'))
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
        
        # 篩選台灣周邊海域
        lat_slice = slice(21, 26)
        lon_slice = slice(118, 124)
        
        # 獲取最新時段數據
        u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        
        lons = u_data['lon'].values
        lats = u_data['lat'].values
        
        # 核心：確保矩陣與 Grid 順序完全一致
        u_val = np.nan_to_num(u_data.values)
        v_val = np.nan_to_num(v_data.values)
        land_mask = np.isnan(u_data.values)
        
        return lons, lats, u_val, v_val, land_mask, latest_time
    except Exception as e:
        st.error(f"HYCOM 數據加載失敗: {e}")
        return None, None, None, None, None, None

@st.cache_data(ttl=1800)
def get_realtime_marine_data(lat, lon):
    """獲取 Open-Meteo 海象數據"""
    try:
        url = "https://marine-api.open-meteo.com/v1/marine"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ["wave_height", "wind_speed_10m", "wind_direction_10m"],
            "timezone": "Asia/Taipei", "forecast_days": 1
        }
        res = requests.get(url, params=params, timeout=5).json()
        wave_h = float(res['hourly']['wave_height'][0])
        wind_s = float(res['hourly']['wind_speed_10m'][0])
        wind_d = float(res['hourly']['wind_direction_10m'][0])
        
        # 轉換風向向量
        rad = np.deg2rad(270 - wind_d)
        wu = wind_s * np.cos(rad)
        wv = wind_s * np.sin(rad)
        return wave_h, wu, wv, wind_s
    except:
        return 1.2, 2.0, 3.0, 5.0

# ===============================
# 2. 導航與路徑 (保持 A*)
# ===============================
def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()
    if not land_mask[lat_idx, lon_idx]: return lat_idx, lon_idx
    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]] - lat)**2 + (lons[ocean[1]] - lon)**2)
    return ocean[0][dist.argmin()], ocean[1][dist.argmin()]

def astar(start, goal, u, v, land_mask, safety, ship_spd, wave_f, wind_f):
    v_ship = ship_spd * 0.277
    rows, cols = land_mask.shape
    pq = [(0, start)]
    cost = {start: 0}; came_from = {}
    while pq:
        _, curr = heapq.heappop(pq)
        if curr == goal: break
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            ni, nj = curr[0] + di, curr[1] + dj
            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni, nj]:
                dist_m = np.sqrt(di**2 + dj**2) * 8000
                flow_eff = (u[curr]*(dj/np.sqrt(di**2+dj**2)) + v[curr]*(di/np.sqrt(di**2+dj**2)))
                v_ground = max(0.5, v_ship + flow_eff)
                step = (dist_m / v_ground) + (wave_f * 150) + (wind_f * 50)
                if safety[ni, nj] < 4: step += 15000 / (safety[ni, nj] + 0.2)
                new_cost = cost[curr] + step
                if (ni, nj) not in cost or new_cost < cost[(ni, nj)]:
                    cost[(ni, nj)] = new_cost
                    priority = new_cost + 4 * (np.sqrt((ni-goal[0])**2 + (nj-goal[1])**2) * 8000 / v_ship)
                    heapq.heappush(pq, (priority, (ni, nj)))
                    came_from[(ni, nj)] = curr
    path = []
    curr = goal
    while curr in came_from:
        path.append(curr); curr = came_from[curr]
    return path[::-1]

# ===============================
# 3. 渲染主頁面
# ===============================
lons, lats, u_val, v_val, land_mask, obs_time = load_hycom_data()

with st.sidebar:
    st.header("📍 航線規劃")
    s_lon = st.number_input("起點經度", 118.0, 124.0, 120.3)
    s_lat = st.number_input("起點緯度", 21.0, 26.0, 22.6)
    e_lon = st.number_input("終點經度", 118.0, 124.0, 122.0)
    e_lat = st.number_input("終點緯度", 21.0, 26.0, 24.5)
    ship_speed = st.slider("船速 (km/h)", 5, 60, 25)

if lons is not None:
    safety_map = distance_transform_edt(~land_mask)
    start_node = nearest_ocean_cell(s_lon, s_lat, lons, lats, land_mask)
    goal_node = nearest_ocean_cell(e_lon, e_lat, lons, lats, land_mask)
    wave_h, wu, wv, ws = get_realtime_marine_data((s_lat+e_lat)/2, (s_lon+e_lon)/2)
    path = astar(start_node, goal_node, u_val, v_val, land_mask, safety_map, ship_speed, wave_h, ws)

    # 指標顯示
    m1, m2, m3, m4 = st.columns(4)
    if path:
        dist_km = sum(np.sqrt((lats[path[i][0]]-lats[path[i+1][0]])**2 + (lons[path[i][1]]-lons[path[i+1][1]])**2) for i in range(len(path)-1)) * 111
        m1.metric("預估時間", f"{dist_km/ship_speed:.1f} hr")
        m2.metric("總航程", f"{dist_km:.1f} km")
    m3.metric("波高", f"{wave_h:.1f} m")
    m4.metric("風速", f"{ws:.1f} m/s")

    # --- 3D 模型繪製 ---
    st.subheader("🌊 HELIOS 3D 分層動態海象模型")
    
    # 確保座標網格正確
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    current_mag = np.sqrt(u_val**2 + v_val**2)
    
    # Z 軸定義
    Z_WAVE = 0
    Z_FLOW = 10
    Z_WIND = 20

    fig3d = go.Figure()

    # 1. 底層：波浪 (依波高起伏)
    fig3d.add_trace(go.Surface(
        x=lon_grid, y=lat_grid, z=np.full_like(lon_grid, Z_WAVE) + (wave_h * np.cos(lon_grid*10)),
        colorscale='Viridis', opacity=0.8, name="波浪層",
        colorbar=dict(title="波高", x=-0.1)
    ))

    # 2. 中層：流場 (關鍵修正：直接使用 u_val, v_val 渲染)
    # 流速曲面
    fig3d.add_trace(go.Surface(
        x=lon_grid, y=lat_grid, z=np.full_like(lon_grid, Z_FLOW) + current_mag,
        colorscale='Blues', opacity=0.6, showscale=False, name="流速面"
    ))
    
    # 流向箭頭 (Cone)
    skip = 2 # 抽樣密度
    fig3d.add_trace(go.Cone(
        x=lon_grid[::skip, ::skip].flatten(),
        y=lat_grid[::skip, ::skip].flatten(),
        z=np.full_like(lon_grid[::skip, ::skip], Z_FLOW + 0.5).flatten(),
        u=u_val[::skip, ::skip].flatten(),
        v=v_val[::skip, ::skip].flatten(),
        w=np.zeros_like(u_val[::skip, ::skip].flatten()),
        sizemode="scaled", sizeref=1.5, colorscale='Blues', showscale=False, name="海流向量"
    ))

    # 3. 頂層：風速箭頭
    fig3d.add_trace(go.Cone(
        x=lon_grid[::skip, ::skip].flatten(),
        y=lat_grid[::skip, ::skip].flatten(),
        z=np.full_like(lon_grid[::skip, ::skip], Z_WIND).flatten(),
        u=np.full_like(lon_grid[::skip, ::skip], wu).flatten(),
        v=np.full_like(lon_grid[::skip, ::skip], wv).flatten(),
        w=np.zeros_like(lon_grid[::skip, ::skip].flatten()),
        sizemode="absolute", sizeref=0.3, anchor="tail", colorscale='Reds', name="風速向量",
        colorbar=dict(title="風速", x=1.1)
    ))

    # 4. 航線
    if path:
        fig3d.add_trace(go.Scatter3d(
            x=[lons[p[1]] for p in path], y=[lats[p[0]] for p in path],
            z=np.full(len(path), Z_WIND + 2),
            mode='lines', line=dict(color='yellow', width=8), name="最優航線"
        ))

    fig3d.update_layout(
        scene=dict(
            xaxis_title="經度", yaxis_title="緯度",
            zaxis=dict(title="分層", tickvals=[Z_WAVE, Z_FLOW, Z_WIND], ticktext=["波浪", "流場", "風速"]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        ),
        height=800, margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # 2D 參考
    st.divider()
    fig2d, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([118, 124, 21, 26])
    ax.add_feature(cfeature.LAND, facecolor='#dddddd')
    ax.add_feature(cfeature.COASTLINE)
    ax.quiver(lons[::2], lats[::2], u_val[::2, ::2], v_val[::2, ::2], color='blue', alpha=0.3)
    if path:
        ax.plot([lons[p[1]] for p in path], [lats[p[0]] for p in path], color='red', lw=2)
    st.pyplot(fig2d)
