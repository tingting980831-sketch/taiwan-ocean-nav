import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ===============================
# 1. 自動時間對接邏輯
# ===============================
def get_nearest_t_step():
    """根據現在 UTC 時間，計算對應的 HYCOM 預報時段 (0, 3, 6...21)"""
    # 假設檔案基準時間是 2024-03-06 12:00 UTC (根據您的檔名)
    base_time = datetime(2024, 3, 6, 12, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    
    # 計算小時差
    diff_hours = (now_utc - base_time).total_seconds() / 3600
    
    # 找出最接近的 3 小時間隔 (0, 3, 6, 9, 12, 15, 18, 21)
    available_steps = np.array([0, 3, 6, 9, 12, 15, 18, 21])
    nearest_val = available_steps[np.argmin(np.abs(available_steps - diff_hours))]
    
    return f"t{nearest_val:03d}"

@st.cache_data
def get_local_hycom_data(t_str):
    base_dir = Path(r"C:\NODASS\HYCOM\2024\03")
    file_name = f"hycom_glby_930_2024030612_{t_str}_uv3z_subscene.nc"
    full_path = base_dir / file_name
    
    if not full_path.exists():
        return None, None, None, None, None
    try:
        ds = xr.open_dataset(full_path, engine="netcdf4")
        subset = ds.isel(depth=0).sel(lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)).load()
        u_val = np.nan_to_num(subset.water_u.values[0]).astype(np.float32)
        v_val = np.nan_to_num(subset.water_v.values[0]).astype(np.float32)
        data_time = pd.to_datetime(subset.time.values[0]).strftime('%Y-%m-%d %H:%M')
        return subset.lat.values, subset.lon.values, u_val, v_val, data_time
    except:
        return None, None, None, None, None

# ===============================
# 2. 導航核心與模擬器
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V22 Dynamic Sim")
st.title("🛰️ HELIOS 智慧導航系統 (自動時間對接 + 動態模擬)")

# 初始化 Session State 用於模擬下一步
if 'current_step_idx' not in st.session_state:
    st.session_state.current_step_idx = 0
if 'sim_lat' not in st.session_state:
    st.session_state.sim_lat = 22.35
if 'sim_lon' not in st.session_state:
    st.session_state.sim_lon = 120.10

with st.sidebar:
    st.header("🕒 時間狀態")
    auto_t = get_nearest_t_step()
    # 允許手動覆蓋或使用自動
    use_auto = st.toggle("自動對接當前時段", value=True)
    t_step = auto_t if use_auto else st.selectbox("手動選擇時段", ["t000", "t003", "t006", "t009", "t012", "t015", "t018", "t021"])
    
    st.divider()
    st.header("📍 航行模擬控制")
    # 起點會隨模擬更新
    s_lat = st.number_input("當前緯度", value=st.session_state.sim_lat, format="%.4f")
    s_lon = st.number_input("當前經度", value=st.session_state.sim_lon, format="%.4f")
    e_lat = st.number_input("終點緯度", value=25.20, format="%.2f")
    e_lon = st.number_input("終點經度", value=122.00, format="%.2f")
    
    col_sim1, col_sim2 = st.columns(2)
    run_btn = col_sim1.button("🚀 規劃全航程")
    next_btn = col_sim2.button("⏭️ 模擬下一步")
    
    if st.button("🔄 重設航程"):
        st.session_state.sim_lat = 22.35
        st.session_state.sim_lon = 120.10
        st.rerun()

# 數據加載
lat, lon, u_2d, v_2d, data_time_label = get_local_hycom_data(t_step)

if lat is not None:
    # 陸地遮罩 (包含大陸與台灣)
    LON, LAT = np.meshgrid(lon, lat)
    forbidden = ((((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1) | \
                ((((LAT - 23.5) / 0.25) ** 2 + ((LON - 119.6) / 0.25) ** 2) < 1) | \
                (LAT > (1.0 * LON - 93.5))

    def get_idx(la, lo): return np.argmin(np.abs(lat-la)), np.argmin(np.abs(lon-lo))
    
    # 模擬下一步的邏輯
    if next_btn:
        # 取得當前位置的流速
        curr_i, curr_j = get_idx(st.session_state.sim_lat, st.session_state.sim_lon)
        u_f, v_f = u_2d[curr_i, curr_j], v_2d[curr_i, curr_j]
        
        # 假設前進 3 小時 (10800秒)，航速 15 節 (~7.7 m/s)
        # 簡單推算下一步位置 (度)
        dt = 10800 
        ship_speed_mps = 15 * 0.514
        # 朝向終點的方向向量
        dist_total = np.hypot(e_lon - s_lon, e_lat - s_lat)
        dir_lon, dir_lat = (e_lon - s_lon)/dist_total, (e_lat - s_lat)/dist_total
        
        # 合成速度 = 船速 + 海流
        new_lon = st.session_state.sim_lon + (dir_lon * ship_speed_mps + u_f) * dt / 111000
        new_lat = st.session_state.sim_lat + (dir_lat * ship_speed_mps + v_f) * dt / 111000
        
        st.session_state.sim_lat = new_lat
        st.session_state.sim_lon = new_lon
        st.toast(f"已前進至下一時段位置: {new_lat:.2f}, {new_lon:.2f}")
        st.rerun()

    # (這裡插入 A* 演算法與繪圖程式碼，同前一版本 V21.1)
    # ... A* 演算法 ...
    
    st.info(f"✅ 系統自動對接預報時段: **{t_step.upper()}** | 數據基準時間: {data_time_label}")
    # 顯示地圖與路徑
