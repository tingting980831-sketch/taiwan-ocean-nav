import requests
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

# ==============================
# 1️⃣ 取得中央氣象署即時海況資料
# ==============================
# 這裡以「龍洞潮位站」為例，你可以抓全台灣所有站資料
url = "https://www.cwb.gov.tw/V8/C/W/OBS_Ocean/ObsOcean_Station.html"  # 假設可抓到 HTML 表格
tables = pd.read_html(url)

# 找出包含風速、浪高、海流的表格（根據你抓到的網站，index 可能不同）
ocean_table = tables[0]  # 例如第一個表格
ocean_table = ocean_table.rename(columns=lambda x: x.strip())  # 去掉空白

# 只保留需要欄位
df = ocean_table[['測站', '緯度', '經度', '浪高(m)', '風力 (m/s) (級)', '海流流向', '流速 (m/s) (節)']]

# 將空值補 NaN，方便運算
df = df.replace('-', np.nan)
df = df.dropna(subset=['緯度', '經度'])

# ==============================
# 2️⃣ 將方向速度轉換成矢量
# ==============================
def wind_dir_to_uv(direction, speed):
    # 16方位轉角度
    dir_map = {
        "北": 0, "北北東": 22.5, "北東": 45, "東北東": 67.5, "東": 90, "東南東": 112.5,
        "南東": 135, "南南東": 157.5, "南": 180, "南南西": 202.5, "南西": 225, "西南西": 247.5,
        "西": 270, "西北西": 292.5, "北西": 315, "北北西": 337.5
    }
    angle = dir_map.get(direction, 0) * np.pi / 180
    u = speed * np.sin(angle)  # 東向
    v = speed * np.cos(angle)  # 北向
    return u, v

# 假設風向為北北東，風速欄位是 float
# df['風速(m/s)'] = df['風力 (m/s) (級)'].astype(float)
# 這裡先假設風向北
df['u_wind'], df['v_wind'] = zip(*df.apply(lambda row: wind_dir_to_uv('北', float(row['風力 (m/s) (級)'])), axis=1))

# 海流速度也可類似處理
# df['u_current'], df['v_current'] = zip(*df.apply(lambda row: wind_dir_to_uv(row['海流流向'], float(row['流速 (m/s) (節)'])*0.51444), axis=1))
# 1節 ≈ 0.51444 m/s

# ==============================
# 3️⃣ 插值到格點 (與 A* 演算法網格一致)
# ==============================
lon_grid = np.linspace(118, 124, 300)  # 你的網格範圍
lat_grid = np.linspace(21, 26, 250)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# 插值風速
u_grid = griddata(df[['經度','緯度']].values, df['u_wind'].values, (lon_mesh, lat_mesh), method='linear', fill_value=0)
v_grid = griddata(df[['經度','緯度']].values, df['v_wind'].values, (lon_mesh, lat_mesh), method='linear', fill_value=0)

wind_speed_grid = np.sqrt(u_grid**2 + v_grid**2)

# 同理可插值波高
wave_grid = griddata(df[['經度','緯度']].values, df['浪高(m)'].values, (lon_mesh, lat_mesh), method='linear', fill_value=0)

# ==============================
# 4️⃣ 將風速、波高加入成本函數
# ==============================
# 假設原成本是海流阻力 cost = base_cost
# 我們可以簡單加權：
alpha = 1.0  # 風速權重
beta = 2.0   # 波高權重

cost_grid = 1 + alpha*wind_speed_grid + beta*wave_grid  # cost 越高表示越難走

# ==============================
# 5️⃣ 在 A* 演算法中使用 cost_grid
# ==============================
# path = a_star_search(start, goal, cost_grid)
