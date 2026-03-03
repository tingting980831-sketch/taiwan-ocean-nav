# ===============================
# AI 海象導航系統（Stable Final）
# ===============================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq
import time

# -------------------------------
# Streamlit 基本設定
# -------------------------------
st.set_page_config(layout="wide")
st.title("⚓ AI 海象導航系統")

# -------------------------------
# 建立模擬海流場 (可替換 HYCOM)
# -------------------------------
lat = np.linspace(20, 27, 250)
lon = np.linspace(118, 124, 250)
LON, LAT = np.meshgrid(lon, lat)

# 模擬流場
u = 0.6 * np.sin(LAT / 2)
v = 0.4 * np.cos(LON / 2)

# -------------------------------
# 台灣陸地遮罩（核心修正）
# -------------------------------

# 粗略台灣形狀（橢圓）
center_lat, center_lon = 23.7, 121.0
a, b = 1.6, 0.8

land_mask = (
    ((LAT - center_lat) / a) ** 2
    + ((LON - center_lon) / b) ** 2
) < 1

# --- 外擴 5 km ---
grid_km = 111 * (lat[1] - lat[0])
expand_cells = int(5 / grid_km)

dist = distance_transform_edt(~land_mask)
buffer_mask = dist <= expand_cells

# 最終禁止區
forbidden = land_mask | buffer_mask

# -------------------------------
# A* 搜尋
# -------------------------------

def heuristic(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def flow_cost(i, j, ni, nj):

    dx = lon[nj] - lon[j]
    dy = lat[ni] - lat[i]

    move_vec = np.array([dx, dy])
    move_vec = move_vec / (np.linalg.norm(move_vec)+1e-6)

    flow_vec = np.array([u[i,j], v[i,j]])

    assist = np.dot(move_vec, flow_vec)

    base = np.hypot(dx, dy)

    return base * (1 - 0.6*assist)


def astar(start, goal):

    open_set = []
    heapq.heappush(open_set, (0, start))

    came = {}
    g = {start: 0}

    dirs = [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(-1,1),(1,-1),(1,1)
    ]

    while open_set:

        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came:
                path.append(current)
                current = came[current]
            path.append(start)
            return path[::-1]

        for d in dirs:

            ni = current[0] + d[0]
            nj = current[1] + d[1]

            if ni<0 or nj<0 or ni>=len(lat) or nj>=len(lon):
                continue

            # 🚨 永久禁止進入陸地
            if forbidden[ni, nj]:
                continue

            new_cost = g[current] + flow_cost(
                current[0],current[1],ni,nj
            )

            nxt = (ni,nj)

            if nxt not in g or new_cost < g[nxt]:
                g[nxt] = new_cost
                f = new_cost + heuristic(nxt, goal)
                heapq.heappush(open_set,(f,nxt))
                came[nxt] = current

    return None


# -------------------------------
# UI 輸入
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    start_lat = st.number_input("起點緯度", value=22.3)
    start_lon = st.number_input("起點經度", value=120.0)

with col2:
    end_lat = st.number_input("終點緯度", value=25.2)
    end_lon = st.number_input("終點經度", value=122.0)

# -------------------------------
# 轉格點
# -------------------------------
def nearest_idx(lat0, lon0):
    i = np.argmin(np.abs(lat-lat0))
    j = np.argmin(np.abs(lon-lon0))
    return i,j

start = nearest_idx(start_lat, start_lon)
goal  = nearest_idx(end_lat, end_lon)

# 若落在禁止區 → 自動推到最近海域
def move_to_sea(pt):
    i,j = pt
    if not forbidden[i,j]:
        return pt

    sea_dist = distance_transform_edt(forbidden)
    idx = np.unravel_index(np.argmax(sea_dist), sea_dist.shape)
    return idx

start = move_to_sea(start)
goal  = move_to_sea(goal)

# -------------------------------
# 計算路徑（自動更新）
# -------------------------------
t0 = time.time()
path = astar(start, goal)
elapsed = time.time() - t0

st.write(f"計算時間：{elapsed:.2f} 秒")

# -------------------------------
# 畫圖
# -------------------------------
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,20,27])
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=0)

# 海流
ax.quiver(
    LON[::8,::8],
    LAT[::8,::8],
    u[::8,::8],
    v[::8,::8],
    scale=20
)

if path:
    py = [lat[p[0]] for p in path]
    px = [lon[p[1]] for p in path]

    ax.plot(px, py, linewidth=3)

ax.scatter(lon[start[1]], lat[start[0]], s=80)
ax.scatter(lon[goal[1]], lat[goal[0]], s=80)

st.pyplot(fig)
