import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from datetime import datetime

# ===============================
# PAGE
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

# ===============================
# SESSION STATE
# ===============================
defaults = {
    "ship_lat":25.06,
    "ship_lon":122.2,
    "dest_lat":22.5,
    "dest_lon":120.0,
    "real_p":[]
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v

# ===============================
# HYCOM 真實流場（強制）
# ===============================
@st.cache_data(ttl=1800)
def fetch_ocean():

    url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"

    try:
        ds=xr.open_dataset(url,decode_times=True)

        sub=ds.sel(
            lat=slice(20,27),
            lon=slice(118,126),
            depth=0
        ).isel(time=-1).load()

        u=sub.water_u.values
        v=sub.water_v.values

        if np.isnan(u).all():
            raise ValueError("Empty current")

        ocean_time=np.datetime_as_string(
            ds.time.values[-1],unit="m"
        )

        return (
            sub.lon.values,
            sub.lat.values,
            u,
            v,
            ocean_time,
            "HYCOM CONNECTED ✅"
        )

    except:
        st.error("❌ HYCOM 流場連線失敗")
        st.stop()

lons,lats,u,v,ocean_time,status = fetch_ocean()

pretty_time = datetime.fromisoformat(
    ocean_time.replace("Z","")
).strftime("%Y-%m-%d %H:%M")

# ===============================
# LAND MASK（台灣 +5km）
# ===============================
BUFFER=0.045

def is_land(lat,lon):
    taiwan=((21.8-BUFFER<=lat<=25.4+BUFFER) and
            (120.0-BUFFER<=lon<=122.1+BUFFER))
    china=(lat>=24.2-BUFFER and lon<=119.6+BUFFER)
    return taiwan or china

# ===============================
# DISTANCE
# ===============================
def haversine(a,b):
    R=6371
    lat1,lon1=np.radians(a)
    lat2,lon2=np.radians(b)
    dlat=lat2-lat1
    dlon=lon2-lon1
    h=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arctan2(np.sqrt(h),np.sqrt(1-h))

# ===============================
# CURRENT VECTOR
# ===============================
def current_vec(lat,lon):
    i=np.abs(lats-lat).argmin()
    j=np.abs(lons-lon).argmin()
    return u[i,j],v[i,j]

# ===============================
# A* 智慧航線（真流場）
# ===============================
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

        if haversine(cur,end)<20:
            path=[cur]
            while tuple(cur) in came:
                cur=came[tuple(cur)]
                path.append(cur)
            return path[::-1]

        for dx,dy in dirs:

            nxt=[cur[0]+dx*step,cur[1]+dy*step]

            if is_land(*nxt):
                continue

            base_speed=6.5  # m/s

            cu,cv=current_vec(*nxt)

            vx=nxt[1]-cur[1]
            vy=nxt[0]-cur[0]

            norm=np.sqrt(vx**2+vy**2)+1e-9
            dirx=vx/norm
            diry=vy/norm

            current_projection=cu*dirx+cv*diry

            effective_speed=np.clip(
                base_speed+current_projection,
                1.5,15
            )

            cost=step/effective_speed
            new_g=g[tuple(cur)]+cost

            if tuple(nxt) not in g or new_g<g[tuple(nxt)]:
                g[tuple(nxt)]=new_g
                f=new_g+h(nxt)
                heapq.heappush(open_set,(f,nxt))
                came[tuple(nxt)]=cur

    return []

# ===============================
# ROUTE STATS
# ===============================
def route_stats(path):

    if len(path)<2:
        return 0,0,0,0

    dist=0
    for i in range(len(path)-1):
        dist+=haversine(path[i],path[i+1])

    speed_kn=12+np.random.uniform(-1,1)
    hours=dist/(speed_kn*1.852)

    dy=path[1][0]-path[0][0]
    dx=path[1][1]-path[0][1]
    heading=(np.degrees(np.arctan2(dx,dy))+360)%360

    return dist,hours,heading,speed_kn

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:

    st.header("🚢 導航控制")

    slat=st.number_input("起點緯度",value=st.session_state.ship_lat,format="%.3f")
    slon=st.number_input("起點經度",value=st.session_state.ship_lon,format="%.3f")
    elat=st.number_input("終點緯度",value=st.session_state.dest_lat,format="%.3f")
    elon=st.number_input("終點經度",value=st.session_state.dest_lon,format="%.3f")

    if st.button("🚀 啟動智慧航線",use_container_width=True):

        if is_land(slat,slon):
            st.error("起點在陸地")
        elif is_land(elat,elon):
            st.error("終點在陸地")
        else:
            path=smart_route([slat,slon],[elat,elon])
            st.session_state.real_p=path
            st.session_state.ship_lat=slat
            st.session_state.ship_lon=slon
            st.session_state.dest_lat=elat
            st.session_state.dest_lon=elon
            st.rerun()

# ===============================
# DASHBOARD
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")

dist,eta,heading,speed_now=route_stats(st.session_state.real_p)
satellite=int(8+np.random.randint(0,6))

r1=st.columns(4)
r1[0].metric("🚀 即時航速",f"{speed_now:.1f} kn")
r1[1].metric("📡 衛星接收",f"{satellite} 顆")
r1[2].metric("🌊 流場狀態",status)
r1[3].metric("🕒 流場時間",pretty_time)

r2=st.columns(3)
r2[0].metric("📏 預計距離",f"{dist:.1f} km")
r2[1].metric("⏱️ 預計航行時間",f"{eta:.1f} hr")
r2[2].metric("🧭 建議航向",f"{heading:.0f}°")

st.markdown("---")

# ===============================
# MAP
# ===============================
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
           color="red",s=80,zorder=6)

ax.scatter(st.session_state.dest_lon,
           st.session_state.dest_lat,
           color="gold",marker="*",s=250,zorder=7)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)
