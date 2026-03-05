import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from pathlib import Path
import os

# ===============================
# 1. 系統路徑與診斷邏輯
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS Nav-Sim")
st.title("🚢 智慧航行路徑模擬系統 (比賽專用版)")

# 定義搜尋路徑
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# --- 診斷區：這段會幫你在比賽展示前確認資料 ---
with st.expander("🔍 系統路徑診斷 (點擊展開)"):
    st.write(f"程式目錄: `{BASE_DIR}`")
    if DATA_DIR.exists():
        st.write("✅ 找到 data 資料夾")
        all_files = [f.name for f in DATA_DIR.glob("*.nc")]
        st.write(f"資料夾內 NC 檔案數量: {len(all_files)}")
        st.write("檔案清單:", all_files)
    else:
        st.error("❌ 找不到 data 資料夾，嘗試在根目錄尋找...")
        all_files = [f.name for f in BASE_DIR.glob("*.nc")]
        st.write("根目錄 NC 檔案:", all_files)

# ===============================
# 2. 數據加載引擎
# ===============================
@st.cache_data(show_spinner="讀取海流數據中...")
def load_hycom_data(t_str):
    """
    不管檔案在哪，只要名字裡有 20240306 和 t000 (或 t003...) 就讀取
    """
    # 建立所有可能的檔案路徑
    possible_files = list(DATA_DIR.glob("*.nc")) + list(BASE_DIR.glob("*.nc"))
    
    target_file = None
    for f in possible_files:
        # 這裡用最寬鬆的匹配，確保能抓到你上傳的檔案
        if "20240306" in f.name and t_str in f.name:
            target_file = f
            break
            
    if not target_file:
        return None, None, None, None, None

    try:
        ds = xr.open_dataset(target_file, engine="netcdf4")
        # 鎖定台灣海域
        subset = ds.isel(depth=0).sel(
            lon=slice(117.0, 125.0), 
            lat=slice(20.0, 27.0)
        ).load()
        
        lat_val = subset.lat.values
        lon_val = subset.lon.values
        u = np.nan_to_num(subset.water_u.values[0]).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values[0]).astype(np.float32)
        return lat_val, lon_val, u, v, target_file.name
    except Exception as e:
        st.error(f"解析檔案時出錯: {e}")
        return None, None, None, None, None

# ===============================
# 3. 模擬狀態管理
# ===============================
if 'ship_pos' not in st.session_state:
    st.session_state.ship_pos = [22.35, 120.10]
if 't_idx' not in st.session_state:
    st.session_state.t_idx = 0
if 'track' not in st.session_state:
    st.session_state.track = []

T_STEPS = ["t000", "t003", "t006", "t009", "t012", "t015", "t018", "t021"]
current_t = T_STEPS[st.session_state.t_idx]

# 執行讀取
lat, lon, u_2d, v_2d, fname = load_hycom_data(current_t)

# ===============================
# 4. 側邊欄與操作介面
# ===============================
with st.sidebar:
    st.header("⚙️ 導航模擬控制")
    if fname:
        st.success(f"✅ 已讀取檔案: {fname}")
    else:
        st.error(f"❌ 找不到包含 '20240306' 與 '{current_t}' 的 .nc 檔案")
        st.info("請檢查 GitHub 上傳的檔名是否正確")

    st.write(f"當前位置: `{st.session_state.ship_pos[0]:.4f}, {st.session_state.ship_pos[1]:.4f}`")
    dest_lat = st.number_input("目的地緯度", value=25.20)
    dest_lon = st.number_input("目的地經度", value=122.00)
    ship_kn = st.slider("航速 (Knots)", 5, 30, 15)

    if st.button("⏭️ 執行模擬下一步 (3小時)", use_container_width=True):
        if u_2d is not None:
            st.session_state.track.append(list(st.session_state.ship_pos))
            i = np.argmin(np.abs(lat - st.session_state.ship_pos[0]))
            j = np.argmin(np.abs(lon - st.session_state.ship_pos[1]))
            
            dt = 10800 # 3小時
            v_ship = ship_kn * 0.5144
            dy, dx = dest_lat - st.session_state.ship_pos[0], dest_lon - st.session_state.ship_pos[1]
            dist = np.hypot(dx, dy)
            
            st.session_state.ship_pos[0] += ((dy/dist * v_ship + v_2d[i, j]) * dt) / 111000
            st.session_state.ship_pos[1] += ((dx/dist * ship_kn + u_2d[i, j]) * dt) / 111000
            
            st.session_state.t_idx = min(st.session_state.t_idx + 1, 7)
            st.rerun()

    if st.button("🔄 重置航程", use_container_width=True):
        st.session_state.ship_pos = [22.35, 120.10]
        st.session_state.t_idx = 0
        st.session_state.track = []
        st.rerun()

# ===============================
# 5. 地圖繪製
# ===============================
if u_2d is not None:
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.2, 124.8, 20.2, 26.8])
    
    # 背景海流速
    speed = np.sqrt(u_2d**2 + v_2d**2)
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', alpha=0.6, shading='auto')
    plt.colorbar(mesh, ax=ax, label='Current Speed (m/s)', fraction=0.03, pad=0.04)
    
    ax.add_feature(cfeature.LAND, facecolor='#151515', zorder=5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1, zorder=6)
    
    # 畫出軌跡
    if st.session_state.track:
        t = np.array(st.session_state.track)
        ax.plot(t[:,1], t[:,0], 'y--', linewidth=1, label='Track')

    # 船隻與終點
    ax.scatter(st.session_state.ship_pos[1], st.session_state.ship_pos[0], color='lime', s=100, zorder=10)
    ax.scatter(dest_lon, dest_lat, color='red', marker='X', s=150, zorder=10)
    
    st.pyplot(fig)
else:
    st.info("📢 等待數據中... 請點擊上方診斷區檢查檔案路徑。")
