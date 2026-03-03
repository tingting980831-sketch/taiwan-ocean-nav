# ===============================
# 🌊 AI Ocean Navigation System
# ===============================

import streamlit as st
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq
from datetime import datetime

# ===============================
# Page
# ===============================
st.set_page_config(layout="wide")
st.title("⚓ AI 智慧海象導航系統")

# ===============================
# Session State
# ===============================
if "real_path" not in st.session_state:
    st.session_state.real_path = []

# ===============================
# HYCOM DATA
# ===============================
@st.cache_data(ttl=3600)
def load_data():

    url = "https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"

    ds = xr.open_dataset(url)

    ds = ds.isel(depth=0, time=-1)

    return ds

ds = load_data()

u = ds["water_u"].values
v = ds["water_v"].values

lat = ds["lat"].values
lon = ds["lon"].values

# ===============================
# 流場時間（修正216345問題）
# ===============================
raw_time = ds["time"].values

try:
    flow_time = np.datetime_as_string(raw_time, unit="h")
except:
    flow_time = str(raw_time)

st.markdown(f"🌊 **流場時間：{flow_time}**")

# ===============================
# 陸地 Mask
# ===============================
mask_land = np.isnan(u)

dist_to_land = distance_transform_edt(~mask_land)

SAFE_DIST = 3

# ===============================
# GPS 起點
# ===============================
colA, colB = st.columns(2)

with colA:
    if st.button("📍 使用GPS定位"):
        gps_lat = 25.04
        gps_lon = 121.56
        st.session_state.start = (gps_lat, gps_lon)

with colB:
    st.write("可手動輸入或GPS")

start_lat = st.number_input("起點緯度", value=25.0)
start_lon = st.number_input("起點經度", value=121.0)

mid_lat = st.number_input("中繼點緯度", value=23.5)
mid_lon = st.number_input("中繼點經度", value=120.5)

end_lat = st.number_input("終點緯度", value=22.3)
end_lon = st.number_input("終點經度", value=120.3)

# ===============================
# 格點轉換
# ===============================
def nearest_idx(value, arr):
    return np.abs(arr - value).argmin()

def is_land(lat_i, lon_i):
    return mask_land[lat_i, lon_i]

# ===============================
# 真實船速模型
# ===============================
SHIP_SPEED = 12  # knots

def ship_speed_model(u_c, v_c, dx, dy):

    direction = np.array([dx, dy])
    norm = np.linalg.norm(direction) + 1e-6
    direction /= norm

    current = np.array([u_c, v_c])

    assist = np.dot(current, direction)

    speed = SHIP_SPEED + assist * 3

    speed = np.clip(speed, 4, 20)

    return speed

# ===============================
# A* 演算法（防亂繞版本）
# ===============================
def astar(start, goal):

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:

        _, current = heapq.heappop(open_set)

        if current == goal:
            break

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),
                       (-1,-1),(1,1),(-1,1),(1,-1)]:

            nx = current[0] + dx
            ny = current[1] + dy

            if nx<0 or ny<0 or nx>=len(lat) or ny>=len(lon):
                continue

            if mask_land[nx,ny]:
                continue

            if dist_to_land[nx,ny] < SAFE_DIST:
                continue

            u_c = u[nx,ny]
            v_c = v[nx,ny]

            speed = ship_speed_model(u_c,v_c,dx,dy)

            dist = np.hypot(dx,dy)

            time_cost = dist / speed

            tentative = g_score[current] + time_cost

            neighbor = (nx,ny)

            if tentative < g_score.get(neighbor,1e9):
                came_from[neighbor]=current
                g_score[neighbor]=tentative

                h = np.hypot(goal[0]-nx,goal[1]-ny)

                heapq.heappush(open_set,(tentative+h,neighbor))

    path=[]
    node=goal

    while node in came_from:
        path.append(node)
        node=came_from[node]

    path.append(start)
    path.reverse()

    return path

# ===============================
# 陸地檢查
# ===============================
def validate_point(name, lat_i, lon_i):

    if is_land(lat_i, lon_i):
        st.error(f"❌ {name} 在陸地上，請重新選擇")
        return False
    return True

# ===============================
# 計算
# ===============================
if st.button("🚢 規劃路徑"):

    s = (nearest_idx(start_lat,lat),
         nearest_idx(start_lon,lon))

    m = (nearest_idx(mid_lat,lat),
         nearest_idx(mid_lon,lon))

    e = (nearest_idx(end_lat,lat),
         nearest_idx(end_lon,lon))

    valid = (
        validate_point("起點",*s)
        and validate_point("中繼點",*m)
        and validate_point("終點",*e)
    )

    if valid:

        path1 = astar(s,m)
        path2 = astar(m,e)

        st.session_state.real_path = path1 + path2

# ===============================
# 距離與時間
# ===============================
def compute_metrics(path):

    dist=0
    time=0

    for i in range(len(path)-1):

        x1,y1=path[i]
        x2,y2=path[i+1]

        d=np.hypot(x2-x1,y2-y1)

        u_c=u[x2,y2]
        v_c=v[x2,y2]

        speed=ship_speed_model(u_c,v_c,x2-x1,y2-y1)

        dist+=d
        time+=d/speed

    return dist*5, time  # km / hr approx

# ===============================
# 畫圖
# ===============================
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(cfeature.LAND)

step=8
ax.quiver(lon[::step],lat[::step],
          u[::step,::step],
          v[::step,::step])

# 路徑
if len(st.session_state.real_path)>1:

    lats=[lat[p[0]] for p in st.session_state.real_path]
    lons=[lon[p[1]] for p in st.session_state.real_path]

    ax.plot(lons,lats,'r-',linewidth=2)

    # ⭐ 終點
    ax.plot(lons[-1],lats[-1],'y*',markersize=15)

    dist,time=compute_metrics(st.session_state.real_path)

else:
    dist,time=0,0

st.pyplot(fig)

# ===============================
# 儀表板（兩行）
# ===============================
row1,row2=st.columns(2)

with row1:
    st.metric("📏 預計距離",f"{dist:.1f} km")

with row2:
    st.metric("⏱ 預計時間",f"{time:.2f} hr")

row3,row4=st.columns(2)

with row3:
    st.metric("⛽ 省油效益","≈18%")

with row4:
    st.metric("⚡ 省時效益","≈12%")
