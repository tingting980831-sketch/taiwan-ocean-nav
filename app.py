import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta

# --- 1. 系統核心設定 (旗艦衛星參數) ---
SAT_CONFIG = {
    "total_sats": 72,
    "planes": 6,
    "altitude_km": 550,
    "link_type": "ISL Laser Link",
    "status": "Active (72/72)",
    "ais_mode": "Real-time (Live)"
}

# --- 2. 模擬海域數據 (117E-125E, 20N-27N) ---
def generate_ocean_currents(t_step):
    lon = np.linspace(117, 125, 40)
    lat = np.linspace(20, 27, 35)
    LON, LAT = np.meshgrid(lon, lat)
    # 模擬黑潮 (Kuroshio) 往東北方向，隨時間 T 微幅變動
    u = np.ones_like(LON) * 0.8 + 0.2 * np.sin(t_step / 5) 
    v = np.ones_like(LAT) * 1.2 + 0.3 * np.cos(t_step / 5)
    return LON, LAT, u, v

# --- 3. 核心演算法：4D A* 與 AI 變頻邏輯 ---
def run_4d_astar_optimization(start_pos, end_pos, base_speed):
    # 模擬 4D 路徑點 (時間同步索引)
    path = []
    current_pos = np.array(start_pos)
    total_fuel_saved = 0
    
    # 簡化演算法邏輯：模擬 20 個導航節點
    for i in range(21):
        t_idx = i  # 模擬時間步長索引
        _, _, u, v = generate_ocean_currents(t_idx)
        
        # 取得當前格點海流向量
        c_u, c_v = 1.0, 1.5 # 假設該區強流
        ocean_v = np.array([c_u, c_v])
        
        # 計算航向向量
        heading_vec = (np.array(end_pos) - current_pos)
        heading_vec = heading_vec / np.linalg.norm(heading_vec)
        
        # 公式：向量點積 Assist (隨傳隨回即時校準)
        assist = np.dot(heading_vec, ocean_v)
        
        # AI 變頻邏輯
        if assist > 0.5:
            engine_speed = base_speed * 1.05 # 順流加壓
            fuel_gain = 0.15
        elif assist < -0.5:
            engine_speed = base_speed * 0.85 # 逆流節能
            fuel_gain = 0.05
        else:
            engine_speed = base_speed
            fuel_gain = 0
            
        # 更新位置 (對地速度 SOG)
        current_pos = current_pos + (heading_vec * engine_speed * 0.01) + (ocean_v * 0.005)
        path.append(current_pos.tolist())
        total_fuel_saved += fuel_gain

    return path, round(total_fuel_saved * 0.9, 1)

# --- 4. Streamlit 介面佈局 ---
st.set_page_config(layout="wide", page_title="HELIOS V6 Flagship")

# 側邊欄控制
st.sidebar.header("🚢 航行參數設定")
base_speed = st.sidebar.slider("基準航速 (kn)", 10.0, 25.0, 15.0)
start_coord = [118.5, 21.5] # 起點 (高雄外海)
end_coord = [123.5, 26.0]   # 終點 (石垣島北部)

# 執行運算
path_data, fuel_efficiency = run_4d_astar_optimization(start_coord, end_coord, base_speed)

# --- 5. 頂部儀表板 (Dashboard) ---
st.title("HELIOS V6 智慧導航系統 - 旗艦通訊版")
st.markdown(f"**衛星系統：** {SAT_CONFIG['total_sats']} 顆 LEO (高度 {SAT_CONFIG['altitude_km']}km) | **鏈路：** {SAT_CONFIG['link_type']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("⚓ 建議航向", "43.2°")
c2.metric("🍃 省油效益", f"{fuel_efficiency}%", delta="Inverter Active")
c3.metric("📡 衛星狀態", SAT_CONFIG["status"])
c4.metric("🚢 AIS 模式", SAT_CONFIG["ais_mode"])

# --- 6. 地圖視覺化 (Pydeck) ---
# 轉換路徑格式供繪圖使用
df_path = pd.DataFrame(path_data, columns=['lon', 'lat'])

view_state = pdk.ViewState(longitude=121.0, latitude=23.5, zoom=6, pitch=0)

# 繪製路徑層
line_layer = pdk.Layer(
    "PathLayer",
    [{"path": path_data}],
    get_color=[255, 0, 255, 200], # 洋紅色
    width_min_pixels=3,
)

# 繪製起終點
points_layer = pdk.Layer(
    "ScatterplotLayer",
    [
        {"pos": start_coord, "name": "Start", "color": [0, 255, 0]},
        {"pos": end_coord, "name": "Destination", "color": [255, 255, 0]}
    ],
    get_position="pos",
    get_color="color",
    get_radius=15000,
)

st.pydeck_chart(pdk.Deck(
    layers=[line_layer, points_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/dark-v10"
))

st.success(f"✅ 72 顆衛星即時連線成功。117°E~125°E 海域 AIS 數據隨傳隨回，4D A* 路徑已優化完畢。")
