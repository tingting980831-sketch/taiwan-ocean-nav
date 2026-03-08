import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata

# =============================
# 1️⃣ Streamlit UI 設定
# =============================
st.set_page_config(page_title="智慧海上導航", layout="wide")
st.title("⚓ 智慧海象導航系統")

# 左側選單
layer_option = st.sidebar.selectbox("選擇底圖", ["海流流向", "風場", "波高"])

# =============================
# 2️⃣ 讀取禁航區與離岸風場
# =============================
try:
    no_go_area = gpd.read_file("no_go_area.geojson")
except:
    st.warning("禁航區資料缺失，暫時不顯示")
    no_go_area = None

try:
    offshore_wind = gpd.read_file("offshore_wind_area.geojson")
except:
    st.warning("離岸風場資料缺失，暫時不顯示")
    offshore_wind = None

# =============================
# 3️⃣ 抓取 CWB 即時海象資料
# =============================
# 這裡示範用 HTML 表格抓取龍洞測站資料
url = "https://www.cwb.gov.tw/V8/C/W/OBS_Ocean/ObsOcean_Station.html"
tables = pd.read_html(url)
ocean_table = tables[0]
ocean_table = ocean_table.rename(columns=lambda x: x.strip())
df = ocean_table[['測站', '緯度', '經度', '浪高(m)', '風力 (m/s) (級)', '海流流向', '流速 (m/s) (節)']]
df = df.replace('-', np.nan).dropna(subset=['緯度', '經度'])

# 轉風速風向為 u,v
def wind_dir_to_uv(direction, speed):
    dir_map = {
        "北": 0, "北北東": 22.5, "北東": 45, "東北東": 67.5, "東": 90, "東南東": 112.5,
        "南東": 135, "南南東": 157.5, "南": 180, "南南西": 202.5, "南西": 225, "西南西": 247.5,
        "西": 270, "西北西": 292.5, "北西": 315, "北北西": 337.5
    }
    angle = dir_map.get(direction, 0) * np.pi / 180
    u = speed * np.sin(angle)
    v = speed * np.cos(angle)
    return u, v

df['u_wind'], df['v_wind'] = zip(*df.apply(lambda row: wind_dir_to_uv('北', float(row['風力 (m/s) (級)'])), axis=1))

# =============================
# 4️⃣ 插值到網格
# =============================
lon_grid = np.linspace(118, 124, 300)
lat_grid = np.linspace(21, 26, 250)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

u_grid = griddata(df[['經度','緯度']].values, df['u_wind'].values, (lon_mesh, lat_mesh), method='linear', fill_value=0)
v_grid = griddata(df[['經度','緯度']].values, df['v_wind'].values, (lon_mesh, lat_mesh), method='linear', fill_value=0)
wind_speed_grid = np.sqrt(u_grid**2 + v_grid**2)
wave_grid = griddata(df[['經度','緯度']].values, df['浪高(m)'].values, (lon_mesh, lat_mesh), method='linear', fill_value=0)

# =============================
# 5️⃣ 計算成本函數
# =============================
alpha = 1.0
beta = 2.0
cost_grid = 1 + alpha*wind_speed_grid + beta*wave_grid

# =============================
# 6️⃣ 畫圖
# =============================
fig, ax = plt.subplots(figsize=(12,8))

if layer_option == "風場":
    speed = wind_speed_grid
    ax.quiver(lon_mesh, lat_mesh, u_grid, v_grid, speed, scale=50, cmap='autumn', alpha=0.5)
elif layer_option == "波高":
    ax.contourf(lon_mesh, lat_mesh, wave_grid, cmap='Blues', alpha=0.6)
else:  # 海流流向
    ax.quiver(lon_mesh, lat_mesh, u_grid, v_grid, np.sqrt(u_grid**2+v_grid**2), scale=50, cmap='Blues', alpha=0.5)

# 畫禁航區和離岸風場
if no_go_area is not None:
    no_go_area.plot(ax=ax, facecolor="red", alpha=0.3, edgecolor='darkred')
if offshore_wind is not None:
    offshore_wind.plot(ax=ax, facecolor="yellow", alpha=0.3, edgecolor='goldenrod')

ax.set_xlim(118,124)
ax.set_ylim(21,26)
ax.set_xlabel("經度")
ax.set_ylabel("緯度")
st.pyplot(fig)

st.success("海象底圖顯示完成！")
