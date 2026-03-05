import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from pathlib import Path

# ===============================
# 1. 自動定位檔案路徑
# ===============================
# 自動偵測：無論在 GitHub 還是電腦跑，都以 app.py 所在位置為基準
BASE_DIR = Path(__file__).parent

def find_nc_file(t_str):
    """
    自動在根目錄或 data 資料夾搜尋 20240306 的檔案
    """
    # 搜尋模式：包含 2024030612 且包含時段(如 t000) 的 .nc 檔
    pattern = f"*2024030612*{t_str}*.nc"
    
    # 優先找 data 資料夾，再找根目錄
    files = list((BASE_DIR / "data").glob(pattern)) + list(BASE_DIR.glob(pattern))
    
    if files:
        return files[0]
    return None

@st.cache_data(show_spinner="讀取海流數據中...")
def load_data(t_str):
    file_path = find_nc_file(t_str)
    if not file_path:
        return None, None, None, None
    
    try:
        # 指定 engine="netcdf4" 確保在 Streamlit Cloud 穩定運行
        ds = xr.open_dataset(file_path, engine="netcdf4")
        # 台灣周邊經緯度切片
        subset = ds.isel(depth=0).sel(
            lon=slice(117.0, 125.0), 
            lat=slice(20.0, 27.0)
        ).load()
        
        u = np.nan_to_num(subset.water_u.values[0]).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values[0]).astype(np.float32)
        return subset.lat.values, subset.lon.values, u, v
    except Exception as e:
        st.error(f"讀取錯誤: {e}")
        return None, None, None, None

# ===============================
# 2. UI 介面與狀態管理
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Nav-Sim")
st.title("🚢 智慧航行路徑模擬系統 (2024-03-06)")

# 模擬器狀態：位置、時段、路徑紀錄
if 'ship_pos' not in st.session_state:
    st.session_state.ship_pos = [22.35, 120.10]
if 't_idx' not in st.session_state:
    st.session_state.t_idx = 0
if 'track' not in st.session_state:
    st.session_state.track = [] # 紀錄走過的路徑

T_STEPS = ["t000", "t003", "t006", "t009", "t012", "t015", "t018", "t021"]
current_t = T_STEPS[st.session_state.t_idx]

# 加載數據
lat, lon, u_2d, v_2d = load_data(current_t)

# ===============================
# 3. 側邊欄控制與模擬邏輯
# ===============================
with st.sidebar:
    st.header("⚙️ 模擬控制")
    if u_2d is not None:
        st.success(f"✅ 已載入時段: {current_t}")
    else:
        st.error(f"❌ 找不到 {current_t} 的檔案")
        st.info("請確認 .nc 檔案已上傳至 GitHub 的 data 資料夾")

    st.write(f"當前位置: `{st.session_state.ship_pos[0]:.3f}, {st.session_state.ship_pos[1]:.3f}`")
    dest_lat = st.number_input("終點緯度", value=25.20, step=0.01)
    dest_lon = st.number_input("終點經度", value=122.00, step=0.01)
    ship_knots = st.slider("航速 (Knots)", 5, 30, 15)

    if st.button("⏭️ 執行下一步 (3小時)", use_container_width=True):
        if u_2d is not None:
            # 紀錄舊位置
            st.session_state.track.append(list(st.session_state.ship_pos))
            
            # 計算格點索引
            i = np.argmin(np.abs(lat - st.session_state.ship_pos[0]))
            j = np.argmin(np.abs(lon - st.session_state.ship_pos[1]))
            
            # 物理計算
            dt = 10800 # 3小時
            v_ship = ship_knots * 0.5144
            dy, dx = dest_lat - st.session_state.ship_pos[0], dest_lon - st.session_state.ship_pos[1]
            dist = np.hypot(dx, dy)
            
            # 更新位置 (加上流速)
            st.session_state.ship_pos[0] += ((dy/dist * v_ship + v_2d[i, j]) * dt) / 111000
            st.session_state.ship_pos[1] += ((dx/dist * v_ship + u_2d[i, j]) * dt) / 111000
            
            # 跳轉時段
            st.session_state.t_idx = min(st.session_state.t_idx + 1, 7)
            st.rerun()

    if st.button("🔄 重置航程", use_container_width=True):
        st.session_state.ship_pos = [22.35, 120.10]
        st.session_state.t_idx = 0
        st.session_state.track = []
        st.rerun()

# ===============================
# 4. 地圖繪製
# ===============================
if u_2d is not None:
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.5, 124.5, 20.5, 26.5])
    
    # 流速底圖
    speed = np.sqrt(u_2d**2 + v_2d**2)
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', alpha=0.6)
    plt.colorbar(mesh, ax=ax, label='Current Speed (m/s)', fraction=0.03, pad=0.04)
    
    # 陸地
    ax.add_feature(cfeature.LAND, facecolor='#202020', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=6)
    
    # 畫出走過的路徑
    if st.session_state.track:
        track_arr = np.array(st.session_state.track)
        ax.plot(track_arr[:,1], track_arr[:,0], 'y--', linewidth=1.5, label='History Track')

    # 船位與終點
    ax.scatter(st.session_state.ship_pos[1], st.session_state.ship_pos[0], color='lime', s=120, zorder=10, label='Ship')
    ax.scatter(dest_lon, dest_lat, color='red', marker='X', s=150, zorder=10, label='Goal')
    
    ax.legend(loc='lower right')
    st.pyplot(fig)
