import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 系統初始化 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 23.184
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 121.739
if 'history_path' not in st.session_state: st.session_state.history_path = [] 
if 'planned_path' not in st.session_state: st.session_state.planned_path = []

# --- 2. 側邊欄控制 ---
st.sidebar.header("🧭 HELIOS 導航控制中心")
loc_mode = st.sidebar.radio("起始定位模式", ["立即定位 (GPS 模擬)", "手動輸入座標"])

if loc_mode == "立即定位 (GPS 模擬)":
    s_lat, s_lon = 23.184, 121.739
    st.sidebar.info(f"📍 GPS 定位: {s_lat}, {s_lon}")
else:
    s_lat = st.sidebar.number_input("起始緯度", value=23.184)
    s_lon = st.sidebar.number_input("起始經度", value=121.739)

d_lat = st.sidebar.number_input("目標緯度", value=25.500)
d_lon = st.sidebar.number_input("目標經度", value=121.800)

# --- 3. 避障與路徑規劃 ---
def plan_route(slat, slon, dlat, dlon):
    steps = 15
    lats = np.linspace(slat, dlat, steps)
    lons = np.linspace(slon, dlon, steps)
    path = []
    for la, lo in zip(lats, lons):
        if 120.0 < lo < 122.2 and 21.9 < la < 25.3: # 陸地避障
            lo = 122.6
        path.append((la, lo))
    return path

if st.sidebar.button("🚀 執行 AI 路徑分析"):
    st.session_state.ship_lat, st.session_state.ship_lon = s_lat, s_lon
    st.session_state.planned_path = plan_route(s_lat, s_lon, d_lat, d_lon)
    st.session_state.history_path = [(s_lat, s_lon)]

# --- 4. 獲取 HYCOM 海流數據 (包含底圖用的格子資料) ---
@st.cache_data(ttl=3600)
def get_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    return xr.open_dataset(url, decode_times=False)

try:
    ds = get_hycom_data()
    # 擷取台灣周邊局部區域數據，提升效能
    subset = ds.sel(lat=slice(20, 27), lon=slice(118, 126), depth=0).isel(time=-1).load()
    
    # 計算流速大小 (Speed) 作為底圖格子顏色
    speed = np.sqrt(subset.water_u**2 + subset.water_v**2)
    
    # 船隻當前位置插值
    curr_data = subset.interp(lat=st.session_state.ship_lat, lon=st.session_state.ship_lon)
    u_act, v_act = float(curr_data.water_u), float(curr_data.water_v)
    
    sog = 15.0 + (u_act * 1.94)
    fuel_gain = 25.4 if u_act > 0.4 else 12.0
except Exception as e:
    st.error(f"數據讀取失敗: {e}")
    u_act, v_act, sog, fuel_gain = 0, 0, 15, 0

# --- 5. 儀表板 ---
st.subheader("📊 HELIOS 即時導航監控儀表板")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 航速 (SOG)", f"{sog:.1f} kn")
c2.metric("⛽ 能源紅利", f"{fuel_gain}%")
c3.metric("📍 當前位置", f"{st.session_state.ship_lon:.2f}E, {st.session_state.ship_lat:.2f}N")
c4.metric("📡 衛星狀態", "LEO 900km Link")

# --- 6. 繪製海流格子圖與路徑 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# A. 繪製海流格子底圖 (Color Mesh)
mesh = ax.pcolormesh(subset.lon, subset.lat, speed, cmap='Blues', alpha=0.7, shading='auto', zorder=0)
plt.colorbar(mesh, ax=ax, label='Current Speed (m/s)', fraction=0.03, pad=0.04)

# B. 陸地與海岸線
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#333333', zorder=2)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', zorder=3)

# C. 預測路徑 (藍色虛線)
if st.session_state.planned_path:
    plon = [p[1] for p in st.session_state.planned_path]
    plat = [p[0] for p in st.session_state.planned_path]
    ax.plot(plon, plat, color='#00FFFF', linestyle='--', linewidth=1.5, label='Predicted (Future)')

# D. 實際路徑 (紅色實線)
if st.session_state.history_path:
    hlon = [p[1] for p in st.session_state.history_path]
    hlat = [p[0] for p in st.session_state.history_path]
    ax.plot(hlon, hlat, color='red', linestyle='-', linewidth=2.5, label='Actual (Verified)', zorder=4)

# E. 船隻當前流場向量 (紅色實線箭頭)
ax.quiver(st.session_state.ship_lon, st.session_state.ship_lat, u_act, v_act, 
          color='red', scale=5, width=0.01, zorder=5)

ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='white', s=80, edgecolors='red', zorder=6)
ax.scatter(d_lon, d_lat, color='yellow', marker='*', s=200, label='Goal', zorder=6)

ax.legend(loc='lower right')
st.pyplot(fig)

# --- 7. 移動控制 ---
if st.button("🚢 執行下一步移動"):
    if len(st.session_state.planned_path) > 1:
        next_pt = st.session_state.planned_path.pop(0)
        st.session_state.ship_lat, st.session_state.ship_lon = next_pt
        st.session_state.history_path.append(next_pt)
        st.rerun()
    else:
        st.success("✅ 已抵達目的地")
