import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt
import requests
import re
from matplotlib.path import Path

st.set_page_config(layout="wide", page_title="HELIOS V8")
st.title("🛰️ HELIOS V8 智慧海象導航系統")

# =====================================================
# ⭐ 禁航區自動解析器（核心升級）
# =====================================================

RAW_RESTRICTED_TEXT = """
A[22.953536,120.171678],G[22.934628,120.175472],
H[22.933136,120.170942],B[22.957810,120.160780]
[22.943956,120.172358],[22.939717,120.173944],
[22.928353,120.157372],[22.936636,120.153547]
[23.280833,119.5],[23.280833,119.509722],
[23.274444,119.509722],[23.274444,119.5]
[24.831074,120.914995],[24.831032,120.915097],
[24.822774,120.909618],[24.818237,120.907696]
[25.0936,121.4439],[25.1109,121.4150],
[25.1257,121.4223],[25.1212,121.4534]
"""

def parse_restricted(text):

    pairs = re.findall(
        r"\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]", text
    )

    coords = [(float(lat), float(lon)) for lat,lon in pairs]

    zones=[]
    temp=[]

    for p in coords:
        temp.append(p)
        if len(temp)>=4:
            zones.append(temp.copy())
            temp=[]

    return zones

RESTRICTED_ZONES = parse_restricted(RAW_RESTRICTED_TEXT)

def in_restricted(lat,lon):
    for zone in RESTRICTED_ZONES:
        poly = Path([(p[1],p[0]) for p in zone])
        if poly.contains_point((lon,lat)):
            return True
    return False

# =====================================================
# HYCOM
# =====================================================

@st.cache_data(ttl=3600)
def load_hycom():

    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds=xr.open_dataset(url,decode_times=False)

    origin=pd.to_datetime(ds['time'].attrs['time_origin'])
    obs_time=origin+pd.to_timedelta(ds['time'].values[-1],unit='h')

    lat_slice=slice(21,26)
    lon_slice=slice(118,124)

    u=ds['ssu'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)
    v=ds['ssv'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)

    lons=u.lon.values
    lats=u.lat.values

    u_val=np.nan_to_num(u.values)
    v_val=np.nan_to_num(v.values)

    land_mask=np.isnan(u.values)

    return lons,lats,u_val,v_val,land_mask,obs_time

lons,lats,u,v,land_mask,obs_time=load_hycom()

# =====================================================
# 風速
# =====================================================

@st.cache_data(ttl=1800)
def get_wind(lat,lon):

    url="https://api.open-meteo.com/v1/forecast"

    params={
        "latitude":lat,
        "longitude":lon,
        "hourly":"wind_speed_10m,wind_direction_10m",
        "timezone":"Asia/Taipei"
    }

    data=requests.get(url,params=params).json()

    try:
        ws=float(data["hourly"]["wind_speed_10m"][0])
        wd=float(data["hourly"]["wind_direction_10m"][0])
    except:
        ws,wd=None,None

    return ws,wd

# =====================================================
# 最近海洋格點
# =====================================================

def nearest_cell(lon,lat):

    lon_i=np.abs(lons-lon).argmin()
    lat_i=np.abs(lats-lat).argmin()

    if not land_mask[lat_i,lon_i]:
        return lat_i,lon_i

    ocean=np.where(~land_mask)

    dist=np.sqrt(
        (lats[ocean[0]]-lat)**2+
        (lons[ocean[1]]-lon)**2
    )

    i=dist.argmin()
    return ocean[0][i],ocean[1][i]

# =====================================================
# A*
# =====================================================

def astar(start,goal,safety,ship_speed):

    v_ship=ship_speed*0.277
    rows,cols=land_mask.shape

    dirs=[(1,0),(-1,0),(0,1),(0,-1),
          (1,1),(1,-1),(-1,1),(-1,-1)]

    pq=[(0,start)]
    cost={start:0}
    came={}

    while pq:

        _,cur=heapq.heappop(pq)

        if cur==goal:
            break

        for d in dirs:

            ni=cur[0]+d[0]
            nj=cur[1]+d[1]

            if not(0<=ni<rows and 0<=nj<cols):
                continue

            if land_mask[ni,nj]:
                continue

            lat=lats[ni]
            lon=lons[nj]

            if in_restricted(lat,lon):
                continue

            dist_m=np.sqrt(d[0]**2+d[1]**2)*8000

            flow=(u[cur]*(d[1])+
                  v[cur]*(d[0]))

            v_ground=max(0.5,v_ship+flow)

            step=dist_m/v_ground

            if safety[ni,nj]<4:
                step+=8000

            new=cost[cur]+step

            if (ni,nj) not in cost or new<cost[(ni,nj)]:
                cost[(ni,nj)]=new
                priority=new+np.hypot(
                    ni-goal[0],nj-goal[1]
                )

                heapq.heappush(pq,(priority,(ni,nj)))
                came[(ni,nj)]=cur

    path=[]
    c=goal
    while c in came:
        path.append(c)
        c=came[c]

    if path:
        path.append(start)

    return path[::-1]

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    s_lon=st.number_input("起點經度",118.0,124.0,120.3)
    s_lat=st.number_input("起點緯度",21.0,26.0,22.6)

    e_lon=st.number_input("終點經度",118.0,124.0,122.0)
    e_lat=st.number_input("終點緯度",21.0,26.0,24.5)

    ship_speed=st.number_input("船速 km/h",1.0,60.0,20.0)

# =====================================================
# 計算
# =====================================================

safety=distance_transform_edt(~land_mask)

start=nearest_cell(s_lon,s_lat)
goal=nearest_cell(e_lon,e_lat)

path=astar(start,goal,safety,ship_speed)

# =====================================================
# 2D MAP（原配色完全保留）
# =====================================================

colors=[
"#E5F0FF","#CCE0FF","#99C2FF","#66A3FF",
"#3385FF","#0066FF","#0052CC","#003D99",
"#002966","#001433","#000E24"
]

cmap=mcolors.LinearSegmentedColormap.from_list("flow",colors)

fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])

ax.add_feature(cfeature.LAND,facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)

flow=np.sqrt(u**2+v**2)

im=ax.pcolormesh(lons,lats,flow,
                 cmap=cmap,
                 shading='auto',
                 alpha=0.8)

plt.colorbar(im,ax=ax,label="海流強度")

# ⭐ 半透明禁航區
for zone in RESTRICTED_ZONES:
    xs=[p[1] for p in zone]
    ys=[p[0] for p in zone]
    ax.fill(xs,ys,
            color="red",
            alpha=0.25,
            transform=ccrs.PlateCarree())

# 航線
if path:
    ax.plot([lons[p[1]] for p in path],
            [lats[p[0]] for p in path],
            color='red',linewidth=2)

ax.scatter(s_lon,s_lat,color='green',s=120,edgecolors='black')
ax.scatter(e_lon,e_lat,color='yellow',marker='*',s=200,edgecolors='black')

plt.title("HELIOS V8 Navigation")

st.pyplot(fig)
