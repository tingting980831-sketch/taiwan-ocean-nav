import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
from pathlib import Path

# ===============================
# 1. 檔案與路徑配置 (固定 20240306)
# ===============================
BASE_PATH = Path(r"C:\NODASS\HYCOM\2024\03")
FILE_PREFIX = "hycom_glby_930_2024030612"
T_STEPS = ["t000", "t003", "t006", "t009", "t012", "t015", "t018", "t021"]

@st.cache_data
def load_hycom_step(t_str):
    """讀取特定時段的表面流場"""
    file_name = f"{FILE_PREFIX}_{t_str}_uv3z_subscene.nc"
    full_path = BASE_PATH / file_name
    
    if not full_path.exists():
        return None, None, None, None
    
    try:
        ds = xr.open_dataset(full_path, engine="netcdf4")
        # 鎖定台灣周邊海域
        subset = ds.isel(depth=0).sel(lon=slice(117.0, 125.0), lat=slice(20.0, 27.0)).load()
        u = np.nan_to_num(subset.water_u.values[0]).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values[0]).astype(np.float32)
        return subset.lat.values, subset.lon.values, u, v
    except:
        return None, None, None, None

# ===============================
# 2. 模擬器狀態初始化
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS 20240306-Final")
st.title("🛰️ HELIOS 智慧導航 (20240306 實戰版)")

if 'ship_pos' not in st.session_state:
    st.session_state.ship_pos = [22.35, 120.10] # 初始座標
if 'step_idx' not in st.session_state:
    st.session_state.step_idx = 0 # 起始時段 t000

# 載入當前時段數據
current_t = T_STEPS[st.session_state.step_idx]
lat, lon, u_2d, v_2d = load_hycom_step(current_t)

# ===============================
# 3. 側邊欄控制與物理引擎
# ===============================
with st.sidebar:
    st.header("🚢 航行監控")
    st.metric("當前檔案時段", current_t.upper())
    st.write(f"座標: `{st.session_state.ship_pos[0]:.4f}, {st.session_state.ship_pos[1]:.4f}`")
    
    st.divider()
    e_lat = st.number_input("目的地緯度", value=25.20)
    e_lon = st.number_input("目的地經度", value=122.00)
    base_speed = st.slider("航速 (kn)", 10, 25, 15)
    
    # --- 模擬下一步邏輯 ---
    if st.button("⏭️ 執行下一步模擬 (3hr)", use_container_width=True):
        if lat is not None:
            # 取得當前格點索引
            curr_la, curr_lo = st.session_state.ship_pos
            idx_i = np.argmin(np.abs(lat - curr_la))
            idx_j = np.argmin(np.abs(lon - curr_lo))
            
            # 物理位移計算 (船速 + 海流)
            dt = 10800 # 3小時 (秒)
            ms_to_deg = 1 / 111000 # 粗略轉換
            ship_ms = base_speed * 0.5144
            
            # 方向向量
            dy, dx = e_lat - curr_la, e_lon - curr_lo
            dist = np.hypot(dx, dy)
            
            # 新位置 = 舊位置 + (指向目的地的船速 + 該點海流) * 時間
            new_la = curr_la + ((dy/dist * ship_ms + v_2d[idx_i, idx_j]) * dt) * ms_to_deg
            new_lo = curr_lo + ((dx/dist * ship_ms + u_2d[idx_i, idx_j]) * dt) * ms_to_deg
            
            st.session_state.ship_pos = [new_la, new_lo]
            st.session_state.step_idx = min(st.session_state.step_idx + 1, len(T_STEPS)-1)
            st.rerun()

    if st.button("🔄 重置航程", use_container_width=True):
        st.session_state.ship_pos = [22.35, 120.10]
        st.session_state.step_idx = 0
        st.rerun()

# ===============================
# 4. 地圖渲染與遮罩
# ===============================
if lat is not None:
    LON, LAT = np.meshgrid(lon, lat)
    # 遮罩：台灣 + 中國大陸沿岸避讓
    forbidden = ((((LAT - 23.7) / 1.75) ** 2 + ((LON - 121.0) / 0.85) ** 2) < 1) | \
                (LAT > (1.0 * LON - 93.5))

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.5, 124.5, 20.5, 26.5])
    
    # 流場底圖
    speed = np.sqrt(u_2d**2 + v_2d**2)
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', alpha=0.7)
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)', fraction=0.03, pad=0.04)
    
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=6)
    
    # 標記位置
    ax.scatter(st.session_state.ship_pos[1], st.session_state.ship_pos[0], color='lime', s=150, zorder=10, label='當前位置')
    ax.scatter(e_lon, e_lat, color='red', marker='*', s=200, zorder=10, label='目的地')
    ax.legend()
    
    st.pyplot(fig)
else:
    st.error(f"❌ 找不到 2024030612_{current_t} 的檔案，請檢查 C:\NODASS\HYCOM\2024\03")
