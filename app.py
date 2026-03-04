import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from datetime import datetime
import math

# =====================================================
# Page
# =====================================================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

# =====================================================
# Session State
# =====================================================
defaults = dict(
    ship_lat=25.06,
    ship_lon=122.2,
    dest_lat=22.5,
    dest_lon=120.0,
    real_p=[]
)

for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v

# =====================================================
# HYCOM 即時流場（OPeNDAP 穩定版）
# =====================================================
@st.cache_data(ttl=1800)
def load_hycom():

    url="https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"

    try:
        ds=xr.open_dataset(url,engine="pydap",decode_times=False)

        sub=ds.sel(
            lat=slice(21,26),
            lon=slice(118,123)
        ).isel(time=0,depth=0)

        u=sub["water_u"].values
        v=sub["water_v"].values

        ocean_time=str(ds.time.values[0])[:19]

        return sub.lon.values,sub.lat.values,u,v,ocean_time,"ONLINE"

    except:
        # fallback 模擬流場
        lon=np.linspace(118,123,80)
        lat=np.linspace(21,26,80)
        X,Y=np.meshgrid(lon,lat)

        u=0.6*np.sin(Y/2)
        v=0.4*np.cos(X/2)

        return lon,lat,u,v,"SIMULATION","OFFLINE"


lons,lats,u,v,ocean_time,flow_status = load_hycom()

# =====================================================
# 陸地限制（5km buffer）
# =====================================================
BUFFER_DEG = 0.045

def is_land(lat,lon):

    taiwan = (
        (21.8-BUFFER_DEG <= lat <= 25.4+BUFFER_DEG) and
        (120.0-BUFFER_DEG <= lon <= 122.1+BUFFER_DEG)
    )

    china = (lat>=24.2-BUFFER_DEG and lon<=119.6+BUFFER_DEG)

    return taiwan or china

# =====================================================
# 距離
# =====================================================
def haversine(a,b):
    R=6371
    lat1,lon1=np.radians(a)
    lat2,lon2=np.radians(b)
    dlat=lat2-lat1
    dlon=lon2-lon1
    h=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arctan2(np.sqrt(h),np.sqrt(1-h))

# =====================================================
# 流速查詢
# =====================================================
def current_vec(lat,lon):
    i=np.abs(lats-lat).argmin()
    j=np.abs(lons-lon).argmin()
    return u[i,j],v[i,j]

# =====================================================
# 建議航向
# =====================================================
def bearing(p1,p2):
    lat1,lon1=np.radians(p1)
    lat2,lon2=np.radians(p2)

    dlon=lon2-lon1
    x=np.sin(dlon)*np.cos(lat2)
    y=np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)

    brng=np.degrees(np.arctan2(x,y))
    return (brng+360)%360

# =====================================================
# A*（流場影響）
# =====================================================
def smart_route(start,end):

    step=0.12
    dirs=[(-1,0),(1,0),(0,-1),(0,1),
          (-1,-1),(-1,1),(1,-1),(1,1)]

    def h(p): return haversine(p,end)

    open_set=[]
    heapq.heappush(open_set,(0,start))

    came={}
    g={tuple(start):0}

    while open_set:

        _,cur=heapq.heappop(open_set)

        if haversine(cur,end)<15:
            path=[cur]
            while tuple(cur) in came:
                cur=came[tuple(cur)]
                path.append(cur)
            return path[::-1]

        for dx,dy in dirs:

            nxt=[cur[0]+dx*step,cur[1]+dy*step]

            if is_land(*nxt):
                continue

            base_speed=12

            cu,cv=current_vec(*nxt)
            current_speed=np.sqrt(cu**2+cv**2)

            eff_speed=max(4,base_speed+current_speed*2)

            cost=step/eff_speed
            new_g=g[tuple(cur)]+cost

            if tuple(nxt) not in g or new_g<g[tuple(nxt)]:
                g[tuple(nxt)]=new_g
                f=new_g+h(nxt)
                heapq.heappush(open_set,(f,nxt))
                came[tuple(nxt)]=cur

    return []

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:

    st.header("🚢 導航控制")

    slat=st.number_input("起點緯度",value=st.session_state.ship_lat)
    slon=st.number_input("起點經度",value=st.session_state.ship_lon)

    elat=st.number_input("終點緯度",value=st.session_state.dest_lat)
    elon=st.number_input("終點經度",value=st.session_state.dest_lon)

    if st.button("🚀 生成智慧航線",use_container_width=True):

        if is_land(slat,slon):
            st.error("起點在陸地")
        elif is_land(elat,elon):
            st.error("終點在陸地")
        else:
            path=smart_route([slat,slon],[elat,elon])

            st.session_state.ship_lat=slat
            st.session_state.ship_lon=slon
            st.session_state.dest_lat=elat
            st.session_state.dest_lon=elon
            st.session_state.real_p=path

            st.rerun()

# =====================================================
# Dashboard
# =====================================================
st.title("🛰️ HELIOS 智慧航行系統")

speed_now=12+np.random.uniform(-1.5,1.5)
satellite=np.random.randint(8,15)

distance=0
heading="--"

if st.session_state.real_p:
    pts=st.session_state.real_p
    distance=sum(haversine(pts[i],pts[i+1]) for i in range(len(pts)-1))
    heading=f"{bearing(pts[0],pts[1]):.0f}°"

c1,c2,c3,c4,c5=st.columns(5)

c1.metric("🚀 即時航速",f"{speed_now:.1f} kn")
c2.metric("📡 衛星接收",f"{satellite} 顆")
c3.metric("🌊 流場連線",flow_status)
c4.metric("🧭 建議航向",heading)
c5.metric("📏 航程距離",f"{distance:.1f} km")

st.markdown("---")

# =====================================================
# Map
# =====================================================
fig,ax=plt.subplots(figsize=(10,8),
        subplot_kw={'projection':ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND,facecolor="#2b2b2b")
ax.add_feature(cfeature.COASTLINE,color="cyan")

speed=np.sqrt(u**2+v**2)
ax.pcolormesh(lons,lats,speed,cmap="YlGn",alpha=0.7)

if st.session_state.real_p:
    py,px=zip(*st.session_state.real_p)
    ax.plot(px,py,color="#ff00ff",linewidth=3)

ax.scatter(st.session_state.ship_lon,
           st.session_state.ship_lat,
           color="red",s=80)

ax.scatter(st.session_state.dest_lon,
           st.session_state.dest_lat,
           color="gold",marker="*",s=250)

ax.set_extent([118.5,125.5,20.5,26.5])

st.pyplot(fig)
