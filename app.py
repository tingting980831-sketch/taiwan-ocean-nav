import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import heapq

st.set_page_config(layout="wide", page_title="HELIOS V9")
st.title("HELIOS V9 智慧海洋導航系統")

# ===============================
# HYCOM 海流
# ===============================

@st.cache_data(ttl=3600)
def load_current():

    try:

        url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"

        ds=xr.open_dataset(url,decode_times=False)

        lat_slice=slice(21,26)
        lon_slice=slice(118,124)

        u=ds["ssu"].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)
        v=ds["ssv"].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)

        speed=np.sqrt(u**2+v**2)

        lons=u.lon.values
        lats=u.lat.values

        return lons,lats,u.values,v.values,speed.values

    except Exception as e:

        st.error(f"HYCOM讀取失敗 {e}")
        return None,None,None,None,None


# ===============================
# GFS 風速
# ===============================

@st.cache_data(ttl=3600)
def load_wind():

    try:

        url="https://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs"

        ds=xr.open_dataset(url)

        lat_slice=slice(21,26)
        lon_slice=slice(118,124)

        u=ds["ugrd10m"].sel(lat=lat_slice,lon=lon_slice).isel(time=0)
        v=ds["vgrd10m"].sel(lat=lat_slice,lon=lon_slice).isel(time=0)

        wind=np.sqrt(u**2+v**2)

        lons=u.lon.values
        lats=u.lat.values

        return lons,lats,wind.values,u.values,v.values

    except Exception as e:

        st.warning(f"風速讀取失敗 {e}")
        return None,None,None,None,None


# ===============================
# WaveWatch 波高
# ===============================

@st.cache_data(ttl=3600)
def load_wave():

    try:

        url="https://nomads.ncep.noaa.gov:9090/dods/wave/gfswave/global"

        ds=xr.open_dataset(url)

        lat_slice=slice(21,26)
        lon_slice=slice(118,124)

        wave=ds["htsgwsfc"].sel(lat=lat_slice,lon=lon_slice).isel(time=0)

        lons=wave.lon.values
        lats=wave.lat.values

        return lons,lats,wave.values

    except Exception as e:

        st.warning(f"波高讀取失敗 {e}")
        return None,None,None


# ===============================
# 讀取限制區
# ===============================

def load_zones():

    try:
        no_go=gpd.read_file("no_go_area.geojson")
    except:
        no_go=gpd.GeoDataFrame()

    try:
        windfarm=gpd.read_file("offshore_wind.geojson")
    except:
        windfarm=gpd.GeoDataFrame()

    return no_go,windfarm


# ===============================
# A* 路徑
# ===============================

def astar(cost,start,goal):

    rows,cols=cost.shape

    open_set=[]
    heapq.heappush(open_set,(0,start))

    came={}
    g={start:0}

    dirs=[(1,0),(-1,0),(0,1),(0,-1),
          (1,1),(1,-1),(-1,1),(-1,-1)]

    while open_set:

        _,cur=heapq.heappop(open_set)

        if cur==goal:

            path=[]

            while cur in came:
                path.append(cur)
                cur=came[cur]

            path.append(start)
            path.reverse()

            return path

        for dx,dy in dirs:

            nx=cur[0]+dx
            ny=cur[1]+dy

            if nx<0 or ny<0 or nx>=rows or ny>=cols:
                continue

            n=(nx,ny)

            t=g[cur]+cost[nx,ny]

            if n not in g or t<g[n]:

                came[n]=cur
                g[n]=t

                heapq.heappush(open_set,(t,n))

    return None


# ===============================
# Cost map
# ===============================

def build_cost(current,wind,wave):

    base=np.ones_like(current)

    cost=base

    cost+=wave*4
    cost+=wind*1.5
    cost+=current*0.5

    return cost


# ===============================
# 載入資料
# ===============================

hlons,hlats,cu,cv,current=load_current()
flons,flats,wind,wu,wv=load_wind()
wlons,wlats,wave=load_wave()

no_go,windfarm=load_zones()


# ===============================
# UI
# ===============================

layer=st.sidebar.radio(
"底圖",
["海流","風場","波高"]
)

start=(10,10)
goal=(80,80)


# ===============================
# 地圖
# ===============================

fig=plt.figure(figsize=(10,8))

ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])

ax.add_feature(cfeature.LAND,facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE)

gl=ax.gridlines(draw_labels=True,alpha=0.3)

gl.top_labels=False
gl.right_labels=False


# ===============================
# 海流
# ===============================

if layer=="海流":

    im=ax.pcolormesh(
        hlons,
        hlats,
        current,
        cmap="Blues",
        shading="auto",
        alpha=0.8
    )

    ax.quiver(
        hlons[::2],
        hlats[::2],
        cu[::2,::2],
        cv[::2,::2],
        color="white",
        alpha=0.5,
        scale=10
    )

    plt.colorbar(im,label="流速 m/s")


# ===============================
# 風
# ===============================

elif layer=="風場":

    im=ax.pcolormesh(
        flons,
        flats,
        wind,
        cmap="Oranges",
        shading="auto",
        alpha=0.8
    )

    ax.quiver(
        flons[::2],
        flats[::2],
        wu[::2,::2],
        wv[::2,::2],
        color="black",
        alpha=0.4
    )

    plt.colorbar(im,label="風速 m/s")


# ===============================
# 波
# ===============================

elif layer=="波高":

    im=ax.pcolormesh(
        wlons,
        wlats,
        wave,
        cmap="coolwarm",
        shading="auto",
        alpha=0.8
    )

    plt.colorbar(im,label="波高 m")


# ===============================
# 禁航區
# ===============================

if not no_go.empty:

    no_go.plot(
        ax=ax,
        facecolor="red",
        alpha=0.3,
        edgecolor="darkred"
    )


# ===============================
# 離岸風場
# ===============================

if not windfarm.empty:

    windfarm.plot(
        ax=ax,
        facecolor="yellow",
        alpha=0.3,
        edgecolor="gold"
    )


# ===============================
# 路徑
# ===============================

cost=build_cost(current,wind,wave)

path=astar(cost,start,goal)

if path:

    xs=[hlons[p[1]] for p in path]
    ys=[hlats[p[0]] for p in path]

    ax.plot(xs,ys,color="lime",linewidth=3)


plt.title("HELIOS V9 智慧航線")

st.pyplot(fig)
