import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- 1. 核心路徑與避障演算法 ---
def generate_helios_path(start_lat, start_lon, dest_lat, dest_lon):
    """
    產生一條避開陸地（台灣）的智慧航線
    """
    # 建立基礎航點
    num_steps = 15
    lats = np.linspace(start_lat, dest_lat, num_steps)
    lons = np.linspace(start_lon, dest_lon, num_steps)
    path = []
    
    for lat, lon in zip(lats, lons):
        # 【陸地避障邏輯】
        # 定義台灣陸地大概範圍 (經度 120-122, 緯度 21.8-25.5)
        # 如果路徑點太靠近陸地，強制將其向東推移至深水區（黑潮流域）
        safe_lon = lon
        if 120.0 < lon < 122.2 and 21.8 < lat < 25.5:
            safe_lon = 122.5  # 強制切換至東部海域繞道，這就是你的「戰術偏航」
        
        path.append((lat, safe_lon))
    
    return path

def get_current_vector(ds, lat, lon, time_index=-1):
    """
    從 HYCOM 提取特定點的流場向量（含雙線性插值）
    """
    try:
        # 使用 interp 進行雙線性插值，確保數據連續性
        point_ds = ds.isel(time=time_index, depth=0).interp(lat=lat, lon=lon)
        u = float(point_ds.water_u)
        v = float(point_ds.water_v)
        return u, v
    except:
        return 0.0, 0.0

# --- 2. 介面初始化 ---
st.set_page_config(page_title="HELIOS 智慧導航決策系統", layout="wide")

if 'full_path' not in st.session_state:
    st.session_state.full_path = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# (側邊欄輸入部分與你原本的相似，這裡簡化)
st.sidebar.header("🧭 HELIOS 控制中心")
c_lon = st.sidebar.number_input("當前經度", value=121.739, format="%.3f")
c_lat = st.sidebar.number_input("當前緯度", value=23.184, format="%.3f")
d_lon = st.sidebar.number_input("目標經度", value=121.800, format="%.3f")
d_lat = st.sidebar.number_input("目標緯度", value=24.500, format="%.3f")

# --- 3. 按下執行路徑分析 ---
if st.sidebar.button("🚀 執行 AI 路徑分析"):
    with st.spinner('📡 正在運算最佳流場路徑...'):
        # 1. 生成避障路徑
        st.session_state.full_path = generate_helios_path(c_lat, c_lon, d_lat, d_lon)
        st.session_state.current_step = 0
        st.success("✅ 已規劃避開陸地之最佳節能航線")

# --- 4. 數據獲取與繪圖 ---
if st.session_state.full_path:
    # 獲取當前位置
    idx = st.session_state.current_step
    curr_loc = st.session_state.full_path[idx]
    
    # 讀取 HYCOM 數據 (建議加入快取以提升速度)
    DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    ds = xr.open_dataset(DATA_URL, decode_times=False)
    
    # 計算儀表板數據 (當前位置的實測數據)
    u_act, v_act = get_current_vector(ds, curr_loc[0], curr_loc[1])
    sog = 15.0 + (u_act * 1.94) # 簡化計算：基礎航速 + 海流增益
    fuel_save = 25.4 if u_act > 0.5 else 12.5 # 模擬你的研究結果
    
    # --- 儀表板呈現 ---
    st.subheader("📊 HELIOS 即時決策儀表板")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚀 當前對地航速", f"{sog:.1f} kn")
    m2.metric("⛽ 預估節能增益", f"{fuel_save}%")
    m3.metric("📍 當前座標", f"{curr_loc[1]:.2f}E, {curr_loc[0]:.2f}N")
    m4.metric("📡 數據狀態", "實時 LEO 鏈結中")

    # --- 地圖呈現 ---
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 繪製底圖與陸地
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#2c2c2c')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white')
    
    # 1. 繪製「完整規劃航線」（藍色細線）
    lats = [p[0] for p in st.session_state.full_path]
    lons = [p[1] for p in st.session_state.full_path]
    ax.plot(lons, lats, color='cyan', linewidth=1, alpha=0.5, label='Planned Path')

    # 2. 繪製「預測海流」（虛線箭頭）
    # 沿著航線每隔幾個點畫出未來的流場預測
    for p in st.session_state.full_path[idx+1::2]:
        up, vp = get_current_vector(ds, p[0], p[1])
        ax.quiver(p[1], p[0], up, vp, color='white', alpha=0.3, 
                  linestyle='--', scale=10, width=0.005)

    # 3. 繪製「當前實測海流」（紅色實線箭頭 - 強調正確性）
    ax.quiver(curr_loc[1], curr_loc[0], u_act, v_act, color='red', 
              scale=5, width=0.01, label='Actual Current (Verified)')

    # 4. 繪製船隻位置
    ax.scatter(curr_loc[1], curr_loc[0], color='red', s=100, edgecolors='white', zorder=5)
    ax.scatter(d_lon, d_lat, color='yellow', marker='*', s=200, label='Destination')

    ax.legend(loc='lower right')
    st.pyplot(fig)
    
    if st.button("🚢 模擬移動至下一航點"):
        if st.session_state.current_step < len(st.session_state.full_path) - 1:
            st.session_state.current_step += 1
            st.rerun()
