import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import make_interp_spline # 用於路徑平滑化

# --- 1. 核心物理常數 (來自研究報告) ---
LEO_STABILITY = 0.982 # 衛星接收穩定度 98.2% [cite: 22, 107]
FUEL_GAIN_AVG = 25.4  # 平均節能 25.4% [cite: 36, 99]

# --- 2. 路徑平滑與避障演算法 ---
def generate_advanced_path(slat, slon, dlat, dlon):
    # 建立多個控制點以實踐「戰術偏航」
    mid_lat = (slat + dlat) / 2
    # 根據研究，強制將路徑向東(黑潮流域)偏移 [cite: 33]
    ctrl_lon = 122.6 if slon < 122.0 else slon + 0.5
    
    nodes = np.array([
        [slat, slon],
        [mid_lat, ctrl_lon], # 誘導轉折點：捕獲流軸動能
        [dlat, dlon]
    ])
    
    # 使用 B-Spline 產生 30 個平滑航點
    t = np.linspace(0, 1, 3)
    t_smooth = np.linspace(0, 1, 30)
    
    # 分別對緯度與經度進行平滑插值
    spl_lat = make_interp_spline(t, nodes[:, 0], k=2)(t_smooth)
    spl_lon = make_interp_spline(t, nodes[:, 1], k=2)(t_smooth)
    
    # 確保不會撞上台灣本島 (緯度 22-25.3, 經度 < 122.1)
    safe_path = []
    for la, lo in zip(spl_lat, spl_lon):
        if 21.9 < la < 25.4 and lo < 122.2:
            lo = 122.5 # 強制推向深水區
        safe_path.append((la, lo))
    return safe_path

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
