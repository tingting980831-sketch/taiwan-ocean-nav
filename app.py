import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

# ==================== 1️⃣ 讀取 HYCOM ====================
def load_hycom():
    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)

    origin = pd.to_datetime(ds['time'].attrs['time_origin'])
    times = origin + pd.to_timedelta(ds['time'].values, unit='h')

    lat_slice = slice(20,26)
    lon_slice = slice(119,123)

    ssu = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
    ssv = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)

    u = ssu.values
    v = ssv.values

    lats = ssu['lat'].values
    lons = ssu['lon'].values

    land_mask = np.isnan(u)

    return lons, lats, u, v, land_mask

# ==================== 2️⃣ 最近海格點 ====================
def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()

    if not land_mask[lat_idx, lon_idx]:
        return (lat_idx, lon_idx)

    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]] - lat)**2 + (lons[ocean[1]] - lon)**2)
    i = dist.argmin()
    return (ocean[0][i], ocean[1][i])

# ==================== 3️⃣ EDT 安全距離 ====================
def compute_safety(land_mask):
    ocean = ~land_mask
    dist = distance_transform_edt(ocean)
    return dist

# ==================== 4️⃣ A* ====================
def astar(start, goal, u, v, land_mask, safety):
    rows, cols = land_mask.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    pq = []
    heapq.heappush(pq, (0, start))
    came_from = {}
    cost = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            break
        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if ni<0 or nj<0 or ni>=rows or nj>=cols:
                continue
            if land_mask[ni,nj]:
                continue
            dist_step = np.sqrt(d[0]**2 + d[1]**2)
            flow = u[cur]*d[1] + v[cur]*d[0]
            land_penalty = 1/(safety[ni,nj]+1)
            new_cost = cost[cur] + dist_step + land_penalty - 0.5*flow
            if (ni,nj) not in cost or new_cost < cost[(ni,nj)]:
                cost[(ni,nj)] = new_cost
                heapq.heappush(pq, (new_cost, (ni,nj)))
                came_from[(ni,nj)] = cur

    if goal not in came_from:
        print("找不到路徑")
        return []

    path=[]
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path

# ==================== 5️⃣ 畫圖 ====================
def plot_navigation(lons, lats, u, v, path, start_lon, start_lat, goal_lon, goal_lat):
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([119,123,20,26], crs=ccrs.PlateCarree())

    # 台灣底圖
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True)

    # 海流背景顏色 (流速強度)
    speed = np.sqrt(u**2 + v**2)
    im = ax.pcolormesh(lons, lats, speed, cmap='Blues', shading='auto', alpha=0.6, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label='流速 (m/s)')

    # 海流箭頭
    ax.quiver(lons, lats, u, v, scale=5, transform=ccrs.PlateCarree(), color='black')

    # 航線
    if len(path)>0:
        path_lats = [lats[p[0]] for p in path]
        path_lons = [lons[p[1]] for p in path]
        ax.plot(path_lons, path_lats, color='red', linewidth=2, transform=ccrs.PlateCarree(), label='AI航線')

    # 起點終點
    ax.scatter(start_lon, start_lat, color='green', s=80, transform=ccrs.PlateCarree(), label='起點')
    ax.scatter(goal_lon, goal_lat, color='blue', s=80, transform=ccrs.PlateCarree(), label='終點')

    plt.legend()
    plt.title('智慧海流導航系統')
    plt.show()

# ==================== 6️⃣ 主程式 ====================
print("讀取 HYCOM 資料...")
lons, lats, u, v, land_mask = load_hycom()
print("計算安全距離...")
safety = compute_safety(land_mask)

# 起點 / 終點設定
start_lon, start_lat = 120.3, 22.6   # 高雄
goal_lon, goal_lat = 122.8, 24.5     # 外海測試

start = nearest_ocean_cell(start_lon, start_lat, lons, lats, land_mask)
goal = nearest_ocean_cell(goal_lon, goal_lat, lons, lats, land_mask)

print("計算路徑...")
path = astar(start, goal, u, v, land_mask, safety)

plot_navigation(lons, lats, u, v, path, start_lon, start_lat, goal_lon, goal_lat)
