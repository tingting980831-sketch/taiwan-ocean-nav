import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# ===============================
# 1. 核心自動數據引擎 (無時間手動選項)
# ===============================
def get_auto_t_step():
    """自動計算最接近當前 UTC 的預報時段 (t000-t021)"""
    # 檔案基準為 2024-03-06 12:00 UTC (您的資料夾日期)
    base_time = datetime(2024, 3, 6, 12, tzinfo=timezone.utc)
    # 這裡我們模擬現在的時間，或者使用 datetime.now(timezone.utc)
    # 為了讓您的 2024 資料能跑，我們計算相對小時差
    now = datetime.now(timezone.utc)
    diff_hours = (now - base_time).total_seconds() / 3600
    
    available = np.array([0, 3, 6, 9, 12, 15, 18, 21])
    # 限制在 t000-t021 範圍內
    nearest = available[np.argmin(np.abs(available - diff_hours))]
    return f"t{max(0, min(21, nearest)):03d}"

@st.cache_data
def load_hycom_auto():
    t_str = get_auto_t_step()
    base_dir = Path(r"C:\NODASS\HYCOM\2024\03")
    file_name = f"hycom_glby_930_2024030612_{t_str}_uv3z_subscene.nc"
    full_path = base_dir / file_name
    
    if not full_path.exists():
        return None, None, None, None, f"找不到對應時段檔案: {t_str}"

    try:
        ds = xr.open_dataset(full_path, engine="netcdf4")
        subset = ds.isel(depth=0).sel(lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)).load()
        u = np.nan_to_num(subset.water_u.values[0]).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values[0]).astype(np.float32)
        return subset.lat.values, subset.lon.values, u, v, t_str
    except:
        return None, None, None, None, "讀取錯誤"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 介面與動態 Session 設定
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V23 Full Auto")
st.title("🛰️ HELIOS 智慧導航模擬器 (全自動時間對接)")

if 'ship_pos' not in st.session_state:
    st.session_state.ship_pos = [22.35, 120.10] # [lat, lon]

# 自動載入資料 (不顯示時間選擇器)
lat, lon, u_2d, v_2d, current_t = load_hycom_auto()

with st.sidebar:
    st.header("📍 任務導航")
    # 顯示目前位置 (不可手動輸入，由模擬控制)
    st.write(f"當前座標: `{st.session_state.ship_pos[0]:.4f}, {st.session_state.ship_pos[1]:.4f}`")
    
    e_lat = st.number_input("終點緯度", value=25.20)
    e_lon = st.number_input("終點經度", value=122.00)
    base_speed = st.slider("航速 (kn)", 10, 30, 15)
    
    st.divider()
    col1, col2 = st.columns(2)
    run_plan = col1.button("🚀 規劃全路徑", use_container_width=True)
    next_sim = col2.button("⏭️ 模擬下一步", use_container_width=True)
    
    if st.button("🔄 重設起點", use_container_width=True):
        st.session_state.ship_pos = [22.35, 120.10]
        st.rerun()

# ===============================
# 3. 模擬下一步邏輯
# ===============================
if lat is not None:
    # 建立避讓遮罩 (大陸、台灣、澎湖)
    LON, LAT = np.meshgrid(lon, lat)
    forbidden = ((((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1) | \
                ((((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1) | \
                (LAT > (1.0 * LON - 93.5))

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))

    if next_sim:
        curr_lat, curr_lon = st.session_state.ship_pos
        i, j = get_idx(curr_lat, curr_lon)
        
        # 物理計算：下一步位置 = 當前位置 + (合成向量 * 時間步長)
        dt = 3600 * 3 # 模擬前進 3 小時
        ship_speed_ms = base_speed * 0.5144
        
        # 方向向量指向終點
        dy, dx = e_lat - curr_lat, e_lon - curr_lon
        mag = np.hypot(dx, dy)
        new_lat = curr_lat + ((dy/mag * ship_speed_ms + v_2d[i, j]) * dt) / 111000
        new_lon = curr_lon + ((dx/mag * ship_speed_ms + u_2d[i, j]) * dt) / 111000
        
        # 檢查是否撞陸地
        ni, nj = get_idx(new_lat, new_lon)
        if forbidden[ni, nj]:
            st.error("❌ 下一步將進入陸地或禁航區，模擬停止！")
        else:
            st.session_state.ship_pos = [new_lat, new_lon]
            st.rerun()

    # ===============================
    # 4. A* 路徑規劃
    # ===============================
    path = None
    if run_plan:
        s_idx = get_idx(st.session_state.ship_pos[0], st.session_state.ship_pos[1])
        g_idx = get_idx(e_lat, e_lon)
        
        open_set, came_from, g_score = [], {}, {s_idx: 0.0}
        heapq.heappush(open_set, (0, s_idx))
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == g_idx:
                path = []
                while current in came_from:
                    path.append(current); current = came_from[current]
                path = path[::-1]; break
            for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ni, nj = current[0]+d[0], current[1]+d[1]
                if 0 <= ni < len(lat) and 0 <= nj < len(lon) and not forbidden[ni, nj]:
                    step_d = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
                    cost = step_d * (1 - 0.7 * np.dot([lon[nj]-lon[current[1]], lat[ni]-lat[current[0]]], [u_2d[ni, nj], v_2d[ni, nj]]))
                    tg = g_score[current] + cost
                    if (ni, nj) not in g_score or tg < g_score[(ni, nj)]:
                        came_from[(ni, nj)] = current
                        g_score[(ni, nj)] = tg
                        heapq.heappush(open_set, (tg + np.hypot(ni-g_idx[0], nj-g_idx[1]), (ni, nj)))

    # ===============================
    # 5. 地圖展示
    # ===============================
    st.info(f"⚡ 自動同步時段: **{current_t.upper()}** | 位置更新時間: {datetime.now().strftime('%H:%M:%S')}")
    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8])
    
    # 繪製流速與陸地
    speed = np.sqrt(u_2d**2 + v_2d**2)
    ax.pcolormesh(lon, lat, speed, cmap='turbo', alpha=0.7, shading='auto')
    ax.add_feature(cfeature.LAND, facecolor='#121212', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=6)
    
    # 標記當前船隻位置
    ax.scatter(st.session_state.ship_pos[1], st.session_state.ship_pos[0], color='lime', s=150, edgecolors='black', zorder=10, label='當前船位')
    ax.scatter(e_lon, e_lat, color='red', marker='X', s=200, zorder=10, label='目的地')

    if path:
        py, px = [lat[p[0]] for p in path], [lon[p[1]] for p in path]
        ax.plot(px, py, color='#FF00FF', linewidth=3, zorder=9, label='最優路徑')
    
    ax.legend(loc='lower right')
    st.pyplot(fig)
else:
    st.error("無法加載資料，請檢查硬碟路徑。")
