import streamlit as st
import numpy as np
import xarray as xr
import heapq
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

st.set_page_config(layout="wide")
st.title("🌊 AI 智慧海流導航系統（HYCOM 真實流場）")

# =====================================================
# HYCOM 真實海流
# =====================================================

@st.cache_data(ttl=3600)
def load_hycom():

    url = (
        "https://ncss.hycom.org/thredds/dodsC/"
        "ESPC-D-V02/GLBy0.08/latest"
    )

    ds = xr.open_dataset(url, decode_times=False)

    ds = ds.sel(
        lat=slice(18, 28),
        lon=slice(117, 125)
    )

    u = ds["water_u"].isel(time=0, depth=0).values
    v = ds["water_v"].isel(time=0, depth=0).values

    lat = ds.lat.values
    lon = ds.lon.values

    return u, v, lat, lon


with st.spinner("🌊 載入真實 HYCOM 海流..."):
    u, v, lat, lon = load_hycom()

LON, LAT = np.meshgrid(lon, lat)

# =====================================================
# 建立陸地遮罩（嚴格海上限制）
# =====================================================

def build_land_mask():

    fig = plt.figure(figsize=(4,4))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="black")
    ax.set_extent([117,125,18,28])

    fig.canvas.draw()

    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    gray = img[:,:,0]
    mask = gray < 50  # land=True

    # 向外擴張約5km
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask, iterations=2)

    return mask

land_mask = build_land_mask()

# resize mask to grid
land_mask = np.resize(land_mask, u.shape)

# =====================================================
# 使用者輸入
# =====================================================

st.sidebar.header("📍 航線設定")

start_lat = st.sidebar.slider("起點緯度",18.5,27.5,22.0)
start_lon = st.sidebar.slider("起點經度",117.5,124.5,120.0)

end_lat = st.sidebar.slider("終點緯度",18.5,27.5,25.0)
end_lon = st.sidebar.slider("終點經度",117.5,124.5,122.0)

ship_speed = st.sidebar.slider("船速 (knots)",5.0,25.0,12.0)

ship_speed_ms = ship_speed * 0.5144

# =====================================================
# 找最近格點
# =====================================================

def nearest(lat0, lon0):
    i = np.abs(lat-lat0).argmin()
    j = np.abs(lon-lon0).argmin()
    return i, j

start = nearest(start_lat, start_lon)
goal = nearest(end_lat, end_lon)

# =====================================================
# A* 成本（真實流場）
# =====================================================

def flow_cost(i,j,ni,nj):

    if land_mask[ni,nj]:
        return 1e12

    dx = lon[nj]-lon[j]
    dy = lat[ni]-lat[i]

    move = np.array([dx,dy])
    move /= (np.linalg.norm(move)+1e-9)

    flow = np.array([u[i,j], v[i,j]])

    assist = np.dot(move, flow)

    eff_speed = ship_speed_ms + assist
    eff_speed = max(eff_speed,1)

    distance = np.hypot(dx,dy)*111000

    return distance/eff_speed


def heuristic(a,b):
    return np.hypot(a[0]-b[0],a[1]-b[1])


# =====================================================
# A* 搜尋
# =====================================================

def astar(start,goal):

    moves = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]

    pq=[]
    heapq.heappush(pq,(0,start))

    came={}
    cost={start:0}

    while pq:

        _,cur = heapq.heappop(pq)

        if cur==goal:
            break

        for di,dj in moves:

            ni = cur[0]+di
            nj = cur[1]+dj

            if not (0<=ni<u.shape[0] and 0<=nj<u.shape[1]):
                continue

            c = cost[cur] + flow_cost(cur[0],cur[1],ni,nj)

            nxt=(ni,nj)

            if nxt not in cost or c<cost[nxt]:
                cost[nxt]=c
                priority=c+heuristic(goal,nxt)
                heapq.heappush(pq,(priority,nxt))
                came[nxt]=cur

    # reconstruct
    path=[]
    cur=goal
    while cur in came:
        path.append(cur)
        cur=came[cur]

    path.append(start)
    path.reverse()

    return path


# =====================================================
# 計算航線
# =====================================================

with st.spinner("🧠 AI 計算最佳航線..."):
    path = astar(start,goal)

route_lat = [lat[i] for i,j in path]
route_lon = [lon[j] for i,j in path]

# =====================================================
# 距離與時間
# =====================================================

def route_stats():

    dist=0
    for k in range(len(path)-1):
        i,j=path[k]
        ni,nj=path[k+1]

        dx=lon[nj]-lon[j]
        dy=lat[ni]-lat[i]
        dist+=np.hypot(dx,dy)*111

    eta = dist/(ship_speed*1.852)

    return dist, eta

distance, eta = route_stats()

# =====================================================
# 儀表板（兩行）
# =====================================================

c1,c2 = st.columns(2)
c1.metric("📏 預計距離 (km)",f"{distance:.1f}")
c2.metric("⏱ 預計時間 (hr)",f"{eta:.1f}")

# =====================================================
# 地圖
# =====================================================

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([117,125,18,28])

ax.add_feature(cfeature.LAND,color="lightgray")
ax.add_feature(cfeature.COASTLINE)

skip=4
ax.quiver(
    LON[::skip,::skip],
    LAT[::skip,::skip],
    u[::skip,::skip],
    v[::skip,::skip],
    scale=15
)

ax.plot(route_lon,route_lat,'r-',linewidth=3)

ax.plot(start_lon,start_lat,'go',markersize=10)
ax.plot(end_lon,end_lat,'y*',markersize=15)

st.pyplot(fig)
