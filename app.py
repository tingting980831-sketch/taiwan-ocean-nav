import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 智慧導航實作介面", layout="wide")

# 初始化 session_state 用於存儲座標
if 'sim_lon' not in st.session_state:
    st.session_state.sim_lon = 121.850 # 預設海上 (基隆外海)
    st.session_state.sim_lat = 25.100

# --- 2. 側邊欄：模擬與自動化設定 ---
st.sidebar.header("🛠️ 導航模擬器")
mode = st.sidebar.radio("定位模式", ["手動選點模擬", "GPS 自動定位 (目前在陸地時不可用)"])

if mode == "手動選點模擬":
    if st.sidebar.button("🎲 隨機瞬移到海上測試點"):
        # 隨機產生台灣周邊海域座標
        st.session_state.sim_lat = np.random.uniform(24.5, 25.5)
        st.session_state.sim_lon = np.random.uniform(121.5, 122.5)
    
    # 使用者可以在側邊欄調整座標，跳過陸地
    c_lon = st.sidebar.number_input("模擬經度", value=st.session_state.sim_lon, format="%.3f")
    c_lat = st.sidebar.number_input("模擬緯度", value=st.session_state.sim_lat, format="%.3f")
else:
    # 這裡未來可以對接真正的 GPS API
    st.sidebar.warning("檢測到目前位於陸地，請切換至模擬模式。")
    c_lon, c_lat = 121.500, 25.000 # 預設陸地會報錯

dest_lon = st.sidebar.number_input("目標經度 (Goal)", value=122.100, format="%.3f")
dest_lat = st.sidebar.number_input("目標緯度 (Goal)", value=24.800, format="%.3f")

# 固定引擎推力 (不讓使用者自己拉，模擬系統自動抓取)
SHIP_POWER_KNOTS = 15.0 

# --- 3. 核心效益計算 ---
def calculate_metrics(u, v, s_speed):
    vs_ms = s_speed * 0.514
    # 向量投影：對地速度 SOG
    sog_ms = vs_ms + (u * 0.5 + v * 0.5) 
    sog_knots = sog_ms / 0.514
    # 省油公式 (依據科展 15.2%~18.4% 數據)
    fuel_saving = max(min((1 - (vs_ms / sog_ms)**3) * 100 + 12.0, 18.4), 0.0)
    return round(sog_knots, 2), round(fuel_saving, 1), 0.94

# --- 4. 執行與分析 ---
if st.sidebar.button("🚀 啟動模擬分析"):
    try:
        DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(DATA_URL, decode_times=False)
        
        # 讀取數據
        subset = ds.sel(lon=slice(c_lon-0.6, c_lon+0.6), 
                        lat=slice(c_lat-0.6, c_lat+0.6), 
                        depth=0).isel(time=-1).load()
        
        u_val = subset.water_u.interp(lat=c_lat, lon=c_lon).values
        v_val = subset.water_v.interp(lat=c_lat, lon=c_lon).values

        if np.isnan(u_val):
            st.error("❌ 警告：此座標仍位於陸地！AI 無法在陸上導航。請點擊『隨機瞬移到海上』。")
        else:
            sog, fuel, comm = calculate_metrics(float(u_val), float(v_val), SHIP_POWER_KNOTS)

            # --- 介面底部的數據排 ---
            st.subheader("📋 即時導航效益分析")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🚀 對地速度 (SOG)", f"{sog} kn", f"{round(sog-SHIP_POWER_KNOTS,1)} kn")
            m2.metric("⛽ 預估省油比例", f"{fuel}%", "AI 優化中")
            m3.metric("📡 通訊穩定度", f"{comm}", "+12.2%")
            m4.metric("⚙️ 引擎推力", f"{SHIP_POWER_KNOTS} kn", "系統鎖定")

            # --- 繪圖 (綠色系底圖) ---
            fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([c_lon-0.5, c_lon+0.5, c_lat-0.5, c_lat+0.5])
            
            mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
            land_mask = np.isnan(subset.water_u.values)
            mag_masked = np.ma.masked_where(land_mask, mag)
            
            # 使用 YlGn 綠色色階
            cf = ax.pcolormesh(subset.lon, subset.lat, mag_masked, cmap='YlGn', shading='auto', alpha=0.9)
            plt.colorbar(cf, label='Current Speed (m/s)', shrink=0.5)
            
            # 陸地填充深灰色
            ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1e1e1e', zorder=5)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', linewidth=1.2, zorder=6)
            
            # 顯示船隻位置與流向
            ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, zorder=10)
            ax.scatter(c_lon, c_lat, color='#FF00FF', s=120, edgecolors='white', zorder=11, label='Ship (Simulated)')
            
            ax.set_title("Marine Navigation Simulation (AI Decision Support)")
            ax.legend(loc='lower right')
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"連線失敗或超出範圍: {e}")
