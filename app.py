import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
from pathlib import Path

# ===============================
# 1. 自動路徑與檔案偵測邏輯
# ===============================
# 取得程式目前的所在目錄 (無論是在你電腦還是 GitHub 網頁)
CURRENT_DIR = Path(__file__).parent 

# 搜尋優先順序：1. data/ 資料夾  2. 程式根目錄
if (CURRENT_DIR / "data").exists():
    BASE_PATH = CURRENT_DIR / "data"
else:
    BASE_PATH = CURRENT_DIR

FILE_PREFIX = "2024030612"
T_STEPS = ["t000", "t003", "t006", "t009", "t012", "t015", "t018", "t021"]

@st.cache_data(show_spinner="正在載入流場數據...")
def load_hycom_step(t_str):
    """從自動偵測的路徑讀取特定時段的表面流場"""
    # 搜尋包含 2024030612 與時段關鍵字 (如 t000) 的 .nc 檔案
    search_pattern = f"*{FILE_PREFIX}*{t_str}*.nc"
    files = list(BASE_PATH.glob(search_pattern))
    
    if not files:
        return None, None, None, None, None
    
    target_file = files[0]
    try:
        # 使用 netcdf4 引擎讀取
        ds = xr.open_dataset(target_file, engine="netcdf4")
        # 鎖定台灣周邊範圍
        subset = ds.isel(depth=0).sel(
            lon=slice(117.0, 125.0), 
            lat=slice(20.0, 27.0)
        ).load()
        
        u = np.nan_to_num(subset.water_u.values[0]).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values[0]).astype(np.float32)
        return subset.lat.values, subset.lon.values, u, v, target_file.name
    except Exception as e:
        st.error(f"讀取錯誤: {e}")
        return None, None, None, None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.asin(np.sqrt(a))

# ===============================
# 2. 系統狀態初始化 (Session State)
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS 20240306 Deployment")
st.title("🛰️ HELIOS 智慧導航系統 (GitHub 部署版)")

if 'ship_pos' not in st.session_state:
    st.session_state.ship_pos = [22.35, 120.10]
if 'step_idx' not in st.session_state:
    st.session_state.step_idx = 0

# 載入資料
current_t_label = T_STEPS[st.session_state.step_idx]
lat, lon, u_2d, v_2d, actual_file_name = load_hycom_step(current_t_label)

# ===============================
# 3. 側邊欄控制與模擬
# ===============================
with st.sidebar:
    st.header("🚢 實時航行監控")
    if actual_file_name:
        st.success(f"已連結數據: {actual_file_name}")
    else:
        st.error(f"⚠️ 找不到時段 {current_t_label} 的檔案")
        st.info(f"搜尋路徑: {BASE_PATH}")
    
    st.write(f"當前座標: `{st.session_state.ship_pos[0]:.4f}, {st.session_state.ship_pos[1]:.4f}`")
    
    st.divider()
    e_lat = st.number_input("目的地緯度", value=25.20)
    e_lon = st.number_input("目的地經度", value=122.00)
    base_speed = st.slider("巡航航速 (kn)", 10, 25, 15)
    
    # --- 模擬下一步邏輯 ---
    if st.button("⏭️ 執行模擬下一步 (3小時)", use_container_width=True):
        if u_2d is not None:
            curr_la, curr_lo = st.session_state.ship_pos
            idx_i = np.argmin(np.abs(lat - curr_la))
            idx_j = np.argmin(np.abs(lon - curr_lo))
            
            dt = 10800 # 3小時
            ship_ms = base_speed * 0.5144
            dy, dx = e_lat - curr_la, e_lon - curr_lo
            dist = np.hypot(dx, dy)
            
            # 合成向量位移
            new_la = curr_la + ((dy/dist * ship_ms + v_2d[idx_i, idx_j]) * dt) / 111000
            new_lo = curr_lo + ((dx/dist * ship_ms + u_2d[idx_i, idx_j]) * dt) / 111000
            
            st.session_state.ship_pos = [new_la, new_lo]
            st.session_state.step_idx = min(st.session_state.step_idx + 1, len(T_STEPS)-1)
            st.rerun()

    if st.button("🔄 重置航程", use_container_width=True):
        st.session_state.ship_pos = [22.35, 120.10]
        st.session_state.step_idx = 0
        st.rerun()

# ===============================
# 4. 地圖渲染
# ===============================
if u_2d is not None:
    # 建立陸地遮罩
    LON, LAT = np.meshgrid(lon, lat)
    forbidden = ((((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1) | \
                (LAT > (1.0 * LON - 93.5))

    fig, ax = plt.subplots(figsize=(11, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8])
    
    # 背景流速
    speed = np.sqrt(u_2d**2 + v_2d**2)
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', alpha=0.7, shading='auto')
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)', fraction=0.03, pad=0.04)
    
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=6)
    
    # 當前船位與終點
    ax.scatter(st.session_state.ship_pos[1], st.session_state.ship_pos[0], color='lime', s=150, edgecolors='black', zorder=10)
    ax.scatter(e_lon, e_lat, color='red', marker='X', s=200, zorder=10)
    
    st.pyplot(fig)
else:
    st.warning("等待數據加載中... 請確保 GitHub 上的 data 資料夾內有 .nc 檔案。")
