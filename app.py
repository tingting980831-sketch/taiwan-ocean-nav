import streamlit as st
import xarray as xr
import numpy as np
import heapq
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =============================
# PAGE
# =============================
st.set_page_config(layout="wide", page_title="AI 海象導航系統")

# =============================
# SIDEBAR (你的選單)
# =============================
with st.sidebar:

    st.title("⚓ 航行設定")

    start_lat = st.number_input("起點緯度", 21.5)
    start_lon = st.number_input("起點經度", 120.0)

    end_lat = st.number_input("終點緯度", 25.0)
    end_lon = st.number_input("終點經度", 122.0)

    ship_speed_kn = st.slider("船速 (knots)", 5.0,20.0,10.0)

    buffer_km = st.slider("離岸安全距離 (km)",1,15,5)

# knots → m/s
ship_speed = ship_speed_kn * 0.5144

# =============================
# HYCOM LOAD (只載一次)
# =============================
@st.cache_resource
def load_hycom():

    url = "https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"

    ds = xr.open_dataset(url, engine="netcdf4")

    subset = ds.sel(
        lon=slice(118,123),
        lat=slice(21,26)
    )

    u = subset["water_u"].isel(time=0, depth=0)
    v = subset["water_v"].isel(time=0, depth=0)

    return subset, u.values, v.values

subset,u,v = load_hycom()

lon = subset.lon.values
lat = subset.lat.values

ny,nx = u.shape

# =============================
# 最近格點
# =============================
def nearest(lat0,lon0):
    iy = np.abs(lat-lat0).argmin()
    ix = np.abs(lon-lon0).argmin()
    return (iy,ix)

start = nearest(start_lat,start_lon)
goal  = nearest(end_lat,end_lon)

# =============================
# A* COST (含流場)
# =============================
neighbors=[(-1,0),(1,0),(0,-1),(0,1),
           (-1,-1),(-1,1),(1,-1),(1,1)]

def heuristic(a,b):
    return np.hypot(a[0]-b[0],a[1]-b[1])

def current_cost(a,b):

    dy=b[0]-a[0]
    dx=b[1]-a[1]

    dist=np.hypot(dx,dy)

    direction=np.array([dy,dx])/(dist+1e-6)

    cu=u[a]
    cv=v[a]

    assist=np.dot([cv,cu],direction)

    v_eff=max(0.5, ship_speed+assist)

    return dist/v_eff

# =============================
# A*
# =============================
def astar(start,goal):

    open_set=[]
    heapq.heappush(open_set,(0,start))

    came={}
    g={start:0}

    while open_set:

        _,cur=heapq.heappop(open_set)

        if cur==goal:
            break

        for dy,dx in neighbors:

            nyy=cur[0]+dy
            nxx=cur[1]+dx

            if not(0<=nyy<ny and 0<=nxx<nx):
                continue

            nxt=(nyy,nxx)

            tentative=g[cur]+current_cost(cur,nxt)

            if nxt not in g or tentative<g[nxt]:

                g[nxt]=tentative
                f=tentative+heuristic(nxt,goal)

                heapq.heappush(open_set,(f,nxt))
                came[nxt]=cur

    path=[]
    node=goal

    while node in came:
        path.append(node)
        node=came[node]

    path.append(start)
    path.reverse()

    return path,g[goal]

# =============================
# RUN PATH
# =============================
t0=time.time()
path,total_cost=astar(start,goal)
runtime=time.time()-t0

# =============================
# 計算航行資訊
# =============================
distance_km=len(path)*4.5   # HYCOM格距約4.5km
hours=distance_km/(ship_speed_kn*1.852)

eta=datetime.now()+timedelta(hours=hours)

# 建議航向
if len(path)>2:
    dy=path[2][0]-path[0][0]
    dx=path[2][1]-path[0][1]
    heading=np.degrees(np.arctan2(dx,dy))%360
else:
    heading=0

# =============================
# DASHBOARD
# =============================
st.title("🌊 AI 海象智慧導航儀表板")

c1,c2,c3,c4,c5=st.columns(5)

c1.metric("🚀 即時航速",f"{ship_speed_kn:.1f} kn")
c2.metric("📏 航行距離",f"{distance_km:.1f} km")
c3.metric("⏱ 預估時間",f"{hours:.2f} hr")
c4.metric("🧭 建議航向",f"{heading:.0f}°")
c5.metric("🕒 抵達時間",eta.strftime("%Y-%m-%d %H:%M"))

# =============================
# MAP
# =============================
speed=np.sqrt(u**2+v**2)

fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,123,21,26])
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)

plt.pcolormesh(lon,lat,speed,shading="auto")

step=6
plt.quiver(
    lon[::step],
    lat[::step],
    u[::step,::step],
    v[::step,::step]
)

py=[lat[p[0]] for p in path]
px=[lon[p[1]] for p in path]

plt.plot(px,py,'r',linewidth=3)

st.pyplot(fig)

st.caption(f"A* 計算時間 {runtime:.2f}s")
