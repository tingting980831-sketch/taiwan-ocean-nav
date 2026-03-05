import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime

# ===============================
# 1. 雲端數據擷取引擎 (OpenDAP)
# ===============================
@st.cache_data(show_spinner="正在從 HYCOM 伺服器下載即時流場...")
def fetch_hycom_online(t_step_idx):
    """
    直接連線至 HYCOM TDS 伺服器擷取特定時段數據
    """
    # HYCOM 2024-03-06 12:00 的 OpenDAP URL
    # 注意：HYCOM 檔案通常保留近期數據，若 2024 數據已移入 Archive，路徑會略有不同
    base_url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    
    try:
        ds = xr.open_dataset(base_url, decode_times=True)
        
        # 尋找 2024-03-06 12:00 UTC 的索引 (這裡簡化處理)
        # 加上 t_step_idx (0=t000, 1=t003...)
        time_idx = 0 + t_step_idx 
        
        # 切片：只抓取台灣周邊 (Lat: 20-27, Lon: 117-125)，深度 0
        subset = ds.isel(depth=0, time=time_idx).sel(
            lon=slice(117.0, 125.0), 
            lat=slice(20.0, 27.0)
        ).load()
        
        u = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values).astype(np.float32)
        data_time = str(subset.time.values)[:16]
        
        return subset.lat.values, subset.lon.values, u, v, data_time
    except Exception as e:
        st.error(f"連線 HYCOM 失敗: {e}")
        st.info("💡 提示：官方伺服器可能偶爾斷線，請檢查網路或稍後再試。")
        return None, None, None, None, None

# ===============================
# 2. 系統狀態與 UI
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Cloud Direct")
st.title("🛰️ HELIOS 雲端數據直連系統 (2024-03-06)")

if 'ship_pos' not in st.session_state:
    st.session_state.ship_pos = [22.35, 120.10]
if 'step_idx' not in st.session_state:
    st.session_state.step_idx = 0

T_LABELS = ["t000", "t003", "t006", "t009", "t012", "t015", "t018", "t021"]

# 執行線上抓取
lat, lon, u_2d, v_2d, obs_time = fetch_hycom_online(st.session_state.step_idx)

with st.sidebar:
    st.header("🚢 雲端航行監控")
    if obs_time:
        st.success(f"📡 數據時間: {obs_time}")
    st.write(f"當前時段: **{T_LABELS[st.session_state.step_idx]}**")
    st.write(f"座標: `{st.session_state.ship_pos[0]:.4f}, {st.session_state.ship_pos[1]:.4f}`")
    
    st.divider()
    e_lat = st.number_input("目的地緯度", value=25.20)
    e_lon = st.number_input("目的地經度", value=122.00)
    
    if st.button("⏭️ 模擬下一步 (3hr)", use_container_width=True):
        if u_2d is not None:
            # 物理位移邏輯 (同前版)
            curr_la, curr_lo = st.session_state.ship_pos
            idx_i = np.argmin(np.abs(lat - curr_la))
            idx_j = np.argmin(np.abs(lon - curr_lo))
            
            # 假設航速 15 節
            dt = 10800
            ship_ms = 15 * 0.5144
            dy, dx = e_lat - curr_la, e_lon - curr_lo
            dist = np.hypot(dx, dy)
            
            st.session_state.ship_pos[0] += ((dy/dist * ship_ms + v_2d[idx_i, idx_j]) * dt) / 111000
            st.session_state.ship_pos[1] += ((dx/dist * ship_ms + u_2d[idx_i, idx_j]) * dt) / 111000
            
            st.session_state.step_idx = min(st.session_state.step_idx + 1, 7)
            st.rerun()

# ===============================
# 3. 渲染與避讓遮罩
# ===============================
if u_2d is not None:
    LON, LAT = np.meshgrid(lon, lat)
    # 中國大陸沿岸 + 台灣遮罩
    forbidden = ((((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1) | \
                (LAT > (1.0 * LON - 93.5))

    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8])
    
    speed = np.sqrt(u_2d**2 + v_2d**2)
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', alpha=0.7)
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)')
    
    ax.add_feature(cfeature.LAND, facecolor='#151515')
    ax.add_feature(cfeature.COASTLINE, edgecolor='white')
    
    # 畫出遮罩邊界 (中國大陸沿岸)
    ax.contour(lon, lat, forbidden, levels=[0.5], colors='red', linewidths=1, linestyles='--')
    
    ax.scatter(st.session_state.ship_pos[1], st.session_state.ship_pos[0], color='lime', s=100)
    ax.scatter(e_lon, e_lat, color='red', marker='X', s=150)
    
    st.pyplot(fig)
