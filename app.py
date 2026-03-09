import requests
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.ndimage import distance_transform_edt

# ----------------------------
# 1️⃣ 抓取即時風浪資料
# ----------------------------
def get_stable_marine_data(lat, lon):
    wave_url = "https://marine-api.open-meteo.com/v1/marine"
    wave_params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["wave_height", "wave_direction"],
        "timezone": "Asia/Taipei", "forecast_days": 1
    }
    wind_url = "https://api.open-meteo.com/v1/forecast"
    wind_params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["wind_speed_10m", "wind_direction_10m"],
        "wind_speed_unit": "kn",
        "timezone": "Asia/Taipei", "forecast_days": 1
    }
    try:
        w_res = requests.get(wave_url, params=wave_params).json()
        f_res = requests.get(wind_url, params=wind_params).json()
        df_wave = pd.DataFrame(w_res['hourly'])
        df_wind = pd.DataFrame(f_res['hourly'])
        df = pd.merge(df_wave, df_wind, on="time")
        df['time'] = pd.to_datetime(df['time'])
        current_time = pd.Timestamp.now().floor('h') 
        current_data = df[df['time'] == current_time]
        return df, current_data
    except Exception as e:
        print(f"錯誤: {e}")
        return None, None

# ----------------------------
# 2️⃣ 抓取最新 HYCOM 流場
# ----------------------------
url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
ds = xr.open_dataset(url, decode_times=False)
time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
time_values = time_origin + pd.to_timedelta(ds['time'].values, unit='h')
lat_slice = slice(21,26)
lon_slice = slice(118,123)
ssu = ds['ssu'].sel(lat=lat_slice, lon=lon_slice)
ssv = ds['ssv'].sel(lat=lat_slice, lon=lon_slice)
latest_index = -1
ssu_latest = ssu.isel(time=latest_index)
ssv_latest = ssv.isel(time=latest_index)
latest_time = time_values[latest_index]
flow_speed = np.sqrt(ssu_latest**2 + ssv_latest**2)
flow_dir = np.arctan2(ssv_latest, ssu_latest) * 180 / np.pi

# ----------------------------
# 3️⃣ 避障距離權重
# ----------------------------
# 建立陸地掩膜 (coastline 為 1，其餘為 0, 可自行調整)
# 這裡假設簡單: lat<21.5 或 lat>25.5 或 lon<118.5 或 lon>122.5 視為陸地
land_mask = np.zeros(ssu_latest.shape)
land_mask[(ssu_latest.lat<21.5) | (ssu_latest.lat>25.5) | (ssu_latest.lon<118.5) | (ssu_latest.lon>122.5)] = 1
distance = distance_transform_edt(1-land_mask)
buffer_cost = np.exp(-distance/2)  # 指數衰減

# ----------------------------
# 4️⃣ 即時風浪加入成本
# ----------------------------
lat_center = 24.2
lon_center = 120.4
_, now_data = get_stable_marine_data(lat_center, lon_center)
if not now_data.empty:
    wave_height = now_data['wave_height'].values[0]
    wind_speed = now_data['wind_speed_10m'].values[0]
else:
    wave_height = 0
    wind_speed = 0

# ----------------------------
# 5️⃣ 定義成本函數
# ----------------------------
alpha, beta, gamma = 1.0, 1.5, 1.0  # 權重，可調整
cost_grid = 1 + alpha*flow_speed + beta*wave_height + gamma*wind_speed + buffer_cost

# ----------------------------
# 6️⃣ A* 演算法
# ----------------------------
def astar(start, goal, cost_grid):
    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    h, w = cost_grid.shape
    open_set = [(0, start)]
    came_from = {}
    gscore = {start:0}
    fscore = {start:np.linalg.norm(np.array(start)-np.array(goal))}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx,dy in neighbors:
            nx, ny = current[0]+dx, current[1]+dy
            if 0<=nx<h and 0<=ny<w:
                tentative_g = gscore[current] + cost_grid[nx,ny]
                neighbor = (nx,ny)
                if neighbor not in gscore or tentative_g<gscore[neighbor]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + np.linalg.norm(np.array(neighbor)-np.array(goal))
                    heapq.heappush(open_set, (fscore[neighbor], neighbor))
    return []

# ----------------------------
# 7️⃣ 測試航線規劃
# ----------------------------
h, w = cost_grid.shape
start = (0,0)
goal = (h-1, w-1)
path = astar(start, goal, cost_grid)

# ----------------------------
# 8️⃣ 可視化結果
# ----------------------------
lons = ds['lon'].sel(lon=lon_slice)
lats = ds['lat'].sel(lat=lat_slice)
plt.figure(figsize=(8,6))
plt.quiver(lons, lats, ssu_latest, ssv_latest, flow_speed, scale=3, cmap='viridis')
if path:
    px, py = zip(*path)
    plt.plot(lons[py], lats[px], color='red', linewidth=2, label='航線')
plt.colorbar(label='流速 (m/s)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'航線規劃 + 流場\n波高: {wave_height} m, 風速: {wind_speed} kn')
plt.legend()
plt.show()
