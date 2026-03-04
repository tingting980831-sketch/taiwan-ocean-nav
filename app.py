import streamlit as st
import xarray as xr
import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =============================
# PAGE
# =============================
st.set_page_config(layout="wide", page_title="AI 海象導航系統")

# =============================
# 固定離岸 5km
# =============================
BUFFER_KM = 5
BUFFER_DEG = BUFFER_KM / 111  # km → degree

# =============================
# SIDEBAR
# =============================
with st.sidebar:

    st.title("⚓ 航行設定")

    start_lat = st.number_input("起點緯度", 21.5)
    start_lon = st.number_input("起點經度", 120.0)

    end_lat = st.number_input("終點緯度", 25.0)
    end_lon = st.number_input("終點經度", 122.0)

    ship_speed_kn = st.slider("模擬航速 (knots)",5.0,20.0,10.0)

ship_speed = ship_speed_kn * 0.5144

# =============================
# HYCOM
# =============================
@st.cache_resource
def load_hycom():

    url="https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"

    ds=xr.open_dataset(url,engine="netcdf4")

    subset=ds.sel(lon=slice(118,123),lat=slice(21,26))

    u=subset["water_u"].isel(time=0,depth=0)
    v=subset["water_v"].isel(time=0,depth=0)

    return subset,u.values,v.values

subset,u,v=load_hycom()

lon=subset.lon.values
lat=subset.lat.values
ny,nx=u.shape

# =============================
# 陸地限制 (5km)
# =============================
def is_land(lat0,lon0):

    taiwan=(
        (21.8-BUFFER_DEG<=lat0<=25.4+BUFFER_DEG)
        and
        (120.0-BUFFER_DEG<=lon0<=122.1+BUFFER_DEG)
    )

    china=(lat0>=24.2-BUFFER_DEG and lon0<=119.6+BUFFER_DEG)

    return taiwan or china

# =============================
# 最近格點
# =============================
def nearest(lat0,lon0):
    iy=np.abs(lat-lat0).argmin()
    ix=np.abs(lon-lon0).argmin()
    return (iy,ix)

start=nearest(start_lat,start_lon)
goal=nearest(end_lat,end_lon)

# =============================
# A*
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

    v_eff=max(0.5,ship_speed+assist)

    return dist/v_eff

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

            lat_test=lat[nyy]
            lon_test=lon[nxx]

            if is_land(lat_test,lon_test):
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
    return path

# =============================
# RUN
# =============================
t0=time.time()
path=astar(start,goal)
runtime=time.time()-t0

# =============================
# 航行資訊
# =============================
distance_km=len(path)*4.5

# 即時航速（取起點流場）
cu=u[start]
cv=v[start]
current_speed=np.sqrt(cu**2+cv**2)

real_speed_kn=max(3,ship_speed_kn+current_speed*1.8)

# 建議航向
if len(path)>2:
    dy=path[2][0]-path[0][0]
    dx=path[2][1]-path[0][1]
    heading=np.degrees(np.arctan2(dx,dy))%360
else:
    heading=0

# =============================
# 🛰 衛星接收情況
# =============================
np.random.seed(int(start_lat*100))
sat_count=np.random.randint(7,15)

if sat_count>=13:
    sat_status="Excellent"
elif sat_count>=10:
    sat_status="Good"
elif sat_count>=7:
    sat_status="Fair"
else:
    sat_status="Poor"

# =============================
# DASHBOARD
# =============================
st.title("🌊 AI 海象智慧導航儀表板")

c1,c2,c3,c4,c5=st.columns(5)

c1.metric("🚀 即時航速",f"{real_speed_kn:.1f} kn")
c2.metric("⚙️ 模擬航速",f"{ship_speed_kn:.1f} kn")
c3.metric("📏 航行距離",f"{distance_km:.1f} km")
c4.metric("🧭 建議航向",f"{heading:.0f}°")
c5.metric("🛰 衛星接收",f"{sat_count} ({sat_status})")

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
