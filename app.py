import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 抓取即時資料 (HYCOM 實時 OpenDAP) ---
@st.cache_data(ttl=3600)
def fetch_realtime_vectors():
    try:
        # HYCOM GLBy0.08 實時全球流場數據節點
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        # 鎖定台灣周邊海域 (深 0m, 最新時標)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()
        return subset
 Kurashio = fetch_realtime_vectors()

# --- 2. 地圖繪製邏輯 (加入 Quiver 箭頭) ---
def plot_navigation_v14(ship_pos, goal_pos, path):
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, facecolor='#2c2c2c', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1, zorder=3)
    
    if Kurashio is not None:
        lon, lat = Kurashio.lon.values, Kurashio.lat.values
        u, v = Kurashio.water_u.values, Kurashio.water_v.values
        speed = np.sqrt(u**2 + v**2)
        
        # 底色：流速熱圖
        im = ax.pcolormesh(lon, lat, speed, cmap='viridis', alpha=0.6, zorder=1)
        
        # --- 核心新增：繪製流向箭頭 ---
        # 為了美觀，我們每隔 4 個網格點畫一個箭頭 (Decimate)
        skip = (slice(None, None, 4), slice(None, None, 4))
        ax.quiver(lon[skip[1]], lat[skip[0]], u[skip], v[skip], 
                  color='white', alpha=0.8, scale=15, 
                  width=0.003, headwidth=3, zorder=4,
                  transform=ccrs.PlateCarree())
        
        plt.colorbar(im, label='Current Speed (m/s)', shrink=0.6)

    # 繪製航線與船隻
    if path:
        py, px = zip(*path)
        ax.plot(px, py, color='magenta', linewidth=2, label='AI Path', zorder=5)
        ax.scatter(ship_pos[1], ship_pos[0], color='red', s=100, edgecolors='white', zorder=6)
        ax.scatter(goal_pos[1], goal_pos[0], color='gold', marker='*', s=250, zorder=7)

    ax.set_extent([119.0, 125.0, 21.0, 26.0])
    return fig

# --- 3. Streamlit UI 儀表板 ---
st.title("🛰️ HELIOS V14 實時流向監控系統")
st.sidebar.markdown("### 🌊 實時流場參數")
st.sidebar.info("資料來源: HYCOM (NCODA Global 1/12°)")

# 模擬南下航點
ship_lat, ship_lon = 24.5, 122.2 # 起點在花蓮外海 (黑潮帶)
goal_lat, goal_lon = 21.5, 121.0 # 終點在南端

# 顯示地圖
with st.spinner('正在獲取最新衛星流場數據...'):
    fig = plot_navigation_v14([ship_lat, ship_lon], [goal_lat, goal_lon], None)
    st.pyplot(fig)

st.success("✅ 箭頭方向即為當前海流方向。如箭頭朝北(向上)且你要南下，請注意阻力。")
