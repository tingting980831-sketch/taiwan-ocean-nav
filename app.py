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
from matplotlib.path import Path
import requests

st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("🛰️ HELIOS V7 智慧海象導航系統")

# =====================================================
# 模擬衛星
# =====================================================
def get_visible_sats():
    return np.random.randint(8,12)

# =====================================================
# HYCOM 海流
# =====================================================
@st.cache_data(ttl=3600)
def load_hycom_data():

    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds=xr.open_dataset(url,decode_times=False)

    time_origin=pd.to_datetime(ds['time'].attrs['time_origin'])
    latest_time=time_origin+pd.to_timedelta(ds['time'].values[-1],unit='h')

    lat_slice=slice(21,26)
    lon_slice=slice(118,124)

    u_data=ds['ssu'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)
    v_data=ds['ssv'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)

    lons=u_data['lon'].values
    lats=u_data['lat'].values

    u=np.nan_to_num(u_data.values)
    v=np.nan_to_num(v_data.values)
    land_mask=np.isnan(u_data.values)

    return lons,lats,u,v,land_mask,latest_time

lons,lats,u,v,land_mask,obs_time=load_hycom_data()

# =====================================================
# 🌬️ 風速（保留原本可用版本）
# =====================================================
@st.cache_data(ttl=1800)
def get_wind(lat,lon):

    url="https://api.open-meteo.com/v1/forecast"
    params={
        "latitude":lat,
        "longitude":lon,
        "hourly":["wind_speed_10m","wind_direction_10m"],
        "timezone":"Asia/Taipei",
        "forecast_days":1
    }

    data=requests.get(url,params=params).json()

    try:
        speed=float(data["hourly"]["wind_speed_10m"][0])
        direction=float(data["hourly"]["wind_direction_10m"][0])
    except:
        speed=None
        direction=None

    return speed,direction

# =====================================================
# 🚫 完全禁止航行區
# =====================================================
RESTRICTED_ZONES=[

[
(22.953536,120.171678),
(22.934628,120.175472),
(22.933136,120.170942),
(22.957810,120.160780)
],

[
(22.943956,120.172358),
(22.939717,120.173944),
(22.928353,120.157372),
(22.936636,120.153547)
],

[
(22.88462,120.17547),
(22.91242,120.17696),
(22.91178,120.17214),
(22.933136,120.170942)
],

[
(23.280833,119.5),
(23.280833,119.509722),
(23.274444,119.509722),
(23.274444,119.5)
],

[
(23.3,119.492222),
(23.3,119.689444),
(23.225,119.689444),
(23.225,119.49222)
],

[
(24.38367,120.58407),
(24.383464,120.5842898),
(24.384821,120.5849261)
],

[
(25.231417,121.648863),
(25.226151,121.651505)
],

[
(25.233410,121.642090),
(25.242200,121.634560)
],

[
(24.583736,121.872265),
(24.585287,121.874346),
(24.583521,121.872801),
(24.584590,121.874067)
],

[
(24.849621,120.928948),
(24.848034,120.929797),
(24.847862,120.930568),
(24.850101,120.930086)
]
]

def in_restricted_zone(lat,lon):
    for zone in RESTRICTED_ZONES:
        poly=Path([(p[1],p[0]) for p in zone])
        if poly.contains_point((lon,lat)):
            return True
    return False

# =====================================================
# 最近海洋格點
# =====================================================
def nearest_ocean_cell(lon,lat,lons,lats,land_mask):

    lon_idx=np.abs(lons-lon).argmin()
    lat_idx=np.abs(lats-lat).argmin()

    if not land_mask[lat_idx,lon_idx]:
        return lat_idx,lon_idx

    ocean=np.where(~land_mask)
    dist=np.sqrt((lats[ocean[0]]-lat)**2+(lons[ocean[1]]-lon)**2)
    i=dist.argmin()
    return ocean[0][i],ocean[1][i]

# =====================================================
# ⭐ A* 避障航線
# =====================================================
def astar(start,goal,u,v,land_mask,safety,ship_spd,wind_factor=0):

    v_ship=ship_spd*0.277
    rows,cols=land_mask.shape

    dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    pq=[(0,start)]
    cost={start:0}
    came={}

    while pq:
        _,cur=heapq.heappop(pq)
        if cur==goal:
            break

        for d in dirs:

            ni,nj=cur[0]+d[0],cur[1]+d[1]

            if 0<=ni<rows and 0<=nj<cols:

                if land_mask[ni,nj]:
                    continue

                lat=lats[ni]
                lon=lons[nj]

                if in_restricted_zone(lat,lon):
                    continue

                dist=8000*np.sqrt(d[0]**2+d[1]**2)
                step=dist/v_ship + wind_factor*20

                new=cost[cur]+step

                if (ni,nj) not in cost or new<cost[(ni,nj)]:
                    cost[(ni,nj)]=new
                    heapq.heappush(pq,(new,(ni,nj)))
                    came[(ni,nj)]=cur

    path=[]
    curr=goal
    while curr in came:
        path.append(curr)
        curr=came[curr]
    path.append(start)

    return path[::-1]

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:

    st.header("航點設定")

    s_lon=st.number_input("起點經度",118.0,124.0,120.3)
    s_lat=st.number_input("起點緯度",21.0,26.0,22.6)

    e_lon=st.number_input("終點經度",118.0,124.0,122.0)
    e_lat=st.number_input("終點緯度",21.0,26.0,24.5)

    ship_speed=st.number_input("船速 km/h",1.0,60.0,20.0)

# =====================================================
# 計算
# =====================================================
safety=distance_transform_edt(~land_mask)

start=nearest_ocean_cell(s_lon,s_lat,lons,lats,land_mask)
goal=nearest_ocean_cell(e_lon,e_lat,lons,lats,land_mask)

wind_speed,wind_dir=get_wind((s_lat+e_lat)/2,(s_lon+e_lon)/2)

path=astar(start,goal,u,v,land_mask,safety,ship_speed,
           wind_factor=wind_speed if wind_speed else 0)

# =====================================================
# 儀表板
# =====================================================
c1,c2,c3=st.columns(3)

dist_km=sum(np.sqrt(
(lats[path[i][0]]-lats[path[i+1][0]])**2+
(lons[path[i][1]]-lons[path[i+1][1]])**2
) for i in range(len(path)-1))*111

c1.metric("航行時間",f"{dist_km/ship_speed:.1f} hr")
c2.metric("航行距離",f"{dist_km:.1f} km")
c3.metric("衛星數",f"{get_visible_sats()} SATS")

wind_status="OK" if wind_speed else "未接到"
st.caption(f"HYCOM資料時間 {obs_time} | 風速資料: {wind_status}")

# =====================================================
# 地圖
# =====================================================
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND,facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)

flow=np.sqrt(u**2+v**2)

im=ax.pcolormesh(lons,lats,flow,shading='auto',alpha=0.8)
plt.colorbar(im,ax=ax,label="流速")

# 禁航區畫出
for zone in RESTRICTED_ZONES:
    xs=[p[1] for p in zone]
    ys=[p[0] for p in zone]
    ax.fill(xs,ys,color="red",alpha=0.3)

# 航線
path_lons=[lons[p[1]] for p in path]
path_lats=[lats[p[0]] for p in path]

ax.plot(path_lons,path_lats,color='yellow',linewidth=2)

ax.scatter(s_lon,s_lat,color='green',s=120,edgecolors='black')
ax.scatter(e_lon,e_lat,color='red',marker='*',s=200)

plt.title("HELIOS V7 Navigation")
st.pyplot(fig)
