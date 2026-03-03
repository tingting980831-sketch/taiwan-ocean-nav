import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq
import math

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
    "route":[]
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v


# ===============================
# HYCOM REAL DATA (Cloud Safe)
# ===============================
@st.cache_data(ttl=3600)
def load_hycom():

    url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest"

    try:
        ds=xr.open_dataset(
            url,
            engine="pydap",
            decode_times=False
        )

        ds=ds.sel(lat=slice(18,28),
                  lon=slice(117,125)
        ).isel(time=-1,depth=0)

        u=ds["water_u"].values
        v=ds["water_v"].values
        lat=ds.lat.values
        lon=ds.lon.values
        status="HYCOM ONLINE"

    except:
        lat=np.linspace(18,28,80)
        lon=np.linspace(117,125,80)
        LON,LAT=np.meshgrid(lon,lat)
        u=0.5*np.sin(LAT/3)
        v=0.4*np.cos(LON/3)
        status="SIMULATION"

    return u,v,lat,lon,status


u,v,lats,lons,status=load_hycom()

# ===============================
# LAND MASK (台灣 +5km)
# ===============================
speed=np.sqrt(u**2+v**2)
ocean_mask=~np.isnan(speed)

dist=distance_transform_edt(ocean_mask)
km_per_pixel=9
safe_mask=dist>=(5/km_per_pixel)


def latlon_to_idx(lat,lon):
    i=np.abs(lats-lat).argmin()
    j=np.abs(lons-lon).argmin()
    return i,j


def is_sea(lat,lon):
    i,j=latlon_to_idx(lat,lon)
    return safe_mask[i,j]


# ===============================
# SHIP MODEL
# ===============================
BASE_SPEED=12  # knots


def ship_speed(i,j,ni,nj):
    dx=nj-j
    dy=ni-i
    mag=math.sqrt(dx*dx+dy*dy)+1e-6

    dirx=dx/mag
    diry=dy/mag

    current=u[i,j]*dirx+v[i,j]*diry
    return max(5,BASE_SPEED+current*3)


# ===============================
# A* FLOW ROUTING
# ===============================
def astar(start,end):

    si,sj=latlon_to_idx(*start)
    ei,ej=latlon_to_idx(*end)

    moves=[(-1,0),(1,0),(0,-1),(0,1),
           (-1,-1),(-1,1),(1,-1),(1,1)]

    open_set=[]
    heapq.heappush(open_set,(0,(si,sj)))

    came={}
    g={(si,sj):0}

    while open_set:

        _,(i,j)=heapq.heappop(open_set)

        if (i,j)==(ei,ej):
            break

        for di,dj in moves:

            ni,nj=i+di,j+dj

            if not(0<=ni<len(lats) and 0<=nj<len(lons)):
                continue

            if not safe_mask[ni,nj]:
                continue

            sp=ship_speed(i,j,ni,nj)

            cost=1/sp
            ng=g[(i,j)]+cost

            if (ni,nj) not in g or ng<g[(ni,nj)]:
                g[(ni,nj)]=ng

                h=np.hypot(ei-ni,ej-nj)/BASE_SPEED
                f=ng+h

                heapq.heappush(open_set,(f,(ni,nj)))
                came[(ni,nj)]=(i,j)

    path=[]
    node=(ei,ej)

    while node in came:
        path.append(node)
        node=came[node]

    path.append((si,sj))
    path.reverse()

    return [[lats[i],lons[j]] for i,j in path]


# ===============================
# SIDEBAR
# ===============================
with st.sidebar:

    st.header("🚢 導航控制")

    slat=st.number_input("起點緯度",value=st.session_state.ship_lat)
    slon=st.number_input("起點經度",value=st.session_state.ship_lon)
    elat=st.number_input("終點緯度",value=st.session_state.dest_lat)
    elon=st.number_input("終點經度",value=st.session_state.dest_lon)

    if st.button("🚀 生成智慧航線",use_container_width=True):

        if not is_sea(slat,slon):
            st.error("起點在陸地或過近海岸")
        elif not is_sea(elat,elon):
            st.error("終點在陸地或過近海岸")
        else:
            with st.spinner("AI 正在分析流場..."):
                route=astar((slat,slon),(elat,elon))

            st.session_state.route=route
            st.session_state.ship_lat=slat
            st.session_state.ship_lon=slon
            st.session_state.dest_lat=elat
            st.session_state.dest_lon=elon


# ===============================
# DASHBOARD
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")

r1=st.columns(4)
r1[0].metric("🚀 基準航速","12 kn")
r1[1].metric("⛽ 省油模式","ON")
r1[2].metric("📡 衛星連線",f"{np.random.randint(8,18)} pcs")
r1[3].metric("🌊 流場狀態",status)

st.markdown("---")

# ===============================
# MAP
# ===============================
fig,ax=plt.subplots(figsize=(10,8),
                    subplot_kw={'projection':ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND,facecolor="black",zorder=2)
ax.add_feature(cfeature.COASTLINE,edgecolor="cyan",linewidth=1)

spd=np.sqrt(u**2+v**2)
ax.pcolormesh(lons,lats,spd,cmap="YlGn",
              shading="auto",alpha=0.7)

skip=(slice(None,None,5),slice(None,None,5))
ax.quiver(lons[skip[1]],lats[skip[0]],
          u[skip],v[skip],
          color="white",scale=30,alpha=0.5)

# ROUTE
if st.session_state.route:
    py,px=zip(*st.session_state.route)
    ax.plot(px,py,color="magenta",linewidth=2.5,zorder=5)

ax.scatter(st.session_state.ship_lon,
           st.session_state.ship_lat,
           color="red",s=80,zorder=6)

ax.scatter(st.session_state.dest_lon,
           st.session_state.dest_lat,
           color="gold",marker="*",s=220,zorder=7)

ax.set_extent([118,125,20,27])

st.pyplot(fig)
