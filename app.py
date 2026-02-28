import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import make_interp_spline # 用於路徑平滑化

# --- 修正後的路徑生成與儀表板邏輯 ---

# 1. 產生平滑路徑 (解決路徑怪怪的問題)
def generate_smooth_path(slat, slon, dlat, dlon):
    steps = 25
    lats = np.linspace(slat, dlat, steps)
    lons = np.linspace(slon, dlon, steps)
    
    path = []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        # 避障修正：台灣陸地範圍 (21.9N-25.3N, 120E-122E)
        if 21.9 < la < 25.4 and 120.0 < lo < 122.2:
            lo = 122.5 # 向東偏移至黑潮區
        path.append((la, lo))
    
    # 簡單平滑處理：避免直角轉彎
    smooth_path = []
    for i in range(len(path)):
        if i == 0 or i == len(path)-1:
            smooth_path.append(path[i])
        else:
            # 取前後點的平均，讓轉折處變圓滑
            avg_la = (path[i-1][0] + path[i][0] + path[i+1][0]) / 3
            avg_lo = (path[i-1][1] + path[i][1] + path[i+1][1]) / 3
            smooth_path.append((avg_la, avg_lo))
    return smooth_path

# 2. 儀表板數值計算 (解決總距離/時間為 0 的問題)
if st.session_state.real_p:
    idx = st.session_state.step_idx
    
    # 假設每一步代表航行了 0.5 小時 (你可以根據需求調整這個比例)
    time_step = 0.5 
    st.session_state.total_time = idx * time_step
    
    # 距離 = 速度 * 時間 (SOG 來自你截圖的 15.7 kn)
    current_sog = 15.7 
    st.session_state.total_dist = st.session_state.total_time * current_sog
    
    # 剩餘距離估算
    rem_dist = 139.0 - st.session_state.total_dist
    if rem_dist < 0: rem_dist = 0

    # --- 顯示儀表板 ---
    st.subheader("📊 HELIOS 衛星導航即時儀表板")
    r1, r2, r3 = st.columns(3)
    r1.metric("🚀 航速 (SOG)", f"{current_sog} kn")
    # 這裡就是修正語法錯誤的地方：
    r1.metric("📡 衛星接收", f"穩定 ({LEO_STABILITY*100:.1f}%)", "LEO-Link")
    
    r2.metric("⛽ 能源紅利", "25.4%", "Optimal")
    r2.metric("📏 航行總距離", f"{st.session_state.total_dist:.1f} nmi")
    
    r3.metric("🎯 剩餘距離", f"{rem_dist:.1f} nmi")
    r3.metric("🕒 航行總時間", f"{st.session_state.total_time:.2f} hrs")
# --- 3. 執行分析時的邏輯 ---
if st.sidebar.button("🚀 執行 AI 路徑分析"):
    with st.spinner('📡 正在運算 HELIOS 向量合成場...'):
        # 生成兩條對比路徑
        st.session_state.real_p = generate_advanced_path(s_lat, s_lon, d_lat, d_lon)
        # 預測路徑(虛線)模擬預報誤差，稍微偏西
        st.session_state.pred_p = [(la, lo - 0.15) for la, lo in st.session_state.real_p]
        
        st.session_state.step_idx = 0
        st.session_state.total_dist = 0.0
        st.session_state.total_time = 0.0
        st.rerun()

# --- 4. 儀表板更新邏輯 (放置於繪圖前) ---
if st.session_state.real_p:
    idx = st.session_state.step_idx
    curr_loc = st.session_state.real_p[idx]
    
    # 模擬計算：根據研究報告之合速度公式 
    # V_sog = sqrt(Ve^2 + Vc^2 + 2*Ve*Vc*cos(theta))
    # 這裡我們直接帶入你截圖中的 SOG 15.7 kn 作為基準
    current_sog = 15.7 
    
    # 更新累計數據
    if idx > 0:
        # 簡單估算：每步代表 0.5 小時
        dt = 0.5 
        st.session_state.total_time = idx * dt
        # 距離 = 速度 * 時間
        st.session_state.total_dist = st.session_state.total_time * current_sog

    # 顯示儀表板 (包含你要求的所有欄位)
    st.subheader("📊 HELIOS 衛星決策儀表板")
    r1, r2, r3 = st.columns(3)
    r1.metric("🚀 航速 (SOG)", f"{current_sog} kn")
    r1.metric("📡 衛星接收", f"穩定 ({LEO_STABILITY*100:.1)%", "LEO-Link")
    
    r2.metric("⛽ 能源紅利", f"{FUEL_GAIN_AVG}%", "Optimal")
    r2.metric("📏 航行總距離", f"{st.session_state.total_dist:.1f} nmi")
    
    r3.metric("🎯 剩餘距離", f"{139.0 - st.session_state.total_dist:.1f} nmi")
    r3.metric("🕒 航行總時間", f"{st.session_state.total_time:.2f} hrs")
