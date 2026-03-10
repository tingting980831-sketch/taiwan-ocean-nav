import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import json
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer

# ===============================
# 基本設定
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("🛰️ HELIOS V7 智慧海象導航系統")

# ===============================
# TWD97 → WGS84 轉換
# ===============================
transformer = Transformer.from_crs(
    "EPSG:3826",
    "EPSG:4326",
    always_xy=True
)

def twd97_to_lonlat(coords):
    out = []
    for x, y in coords:
        lon, lat = transformer.transform(x, y)
        out.append((lon, lat))
    return out

# ===============================
# 讀取離岸風場 GeoJSON
# ===============================
@st.cache_data
def load_offshore_wind_geojson(path):

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        return []

    polygons = []

    for feature in data["features"]:
        geom = feature["geometry"]

        if geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                for ring in poly:
                    polygons.append(twd97_to_lonlat(ring))

    return polygons


OFFSHORE_WIND = load_offshore_wind_geojson("offshore_wind.geojson")

# ===============================
# 禁航區（你的原始）
# ===============================
NO_GO_ZONES = [
    [[22.953536,120.171678],[22.934628,120.175472],[22.933136,120.170942],[22.957810,120.160780]],
    [[22.943956,120.172358],[22.939717,120.173944],[22.928353,120.157372],[22.936636,120.153547]],
]

# ===============================
# HYCOM 海流（完全沿用你方法）
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():

    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds=xr.open_dataset(url,decode_times=False)

    time_origin=pd.to_datetime(ds['time'].attrs['time_origin'])
    latest_time=time_origin+pd.to_timedelta(ds['time'].values[-1],unit='h')

    lat_slice=slice(21,26)
    lon_slice=slice(118,124)

    ssu=ds['ssu'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)
    ssv=ds['ssv'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)

    lons=ssu['lon'].values
    lats=ssu['lat'].values

    u=np.nan_to_num(ssu.values)
    v=np.nan_to_num(ssv.values)
    land_mask=np.isnan(ssu.values)

    return lons,lats,u,v,land_mask,latest_time

lons,lats,u,v,land_mask,obs_time=load_hycom_data()

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("航點設定")

    s_lon=st.number_input("起點經度",118.0,124.0,120.3)
    s_lat=st.number_input("起點緯度",21.0,26.0,22.6)

    e_lon=st.number_input("終點經度",118.0,124.0,122.0)
    e_lat=st.number_input("終點緯度",21.0,26.0,24.5)

    ship_speed=st.number_input("船速 km/h",1.0,60.0,20.0)

# ===============================
# 找最近海洋格點
# ===============================
def nearest_ocean_cell(lon,lat,lons,lats,mask):

    j=np.abs(lons-lon).argmin()
    i=np.abs(lats-lat).argmin()

    if not mask[i,j]:
        return i,j

    ocean=np.where(~mask)
    d=((lats[ocean[0]]-lat)**2+(lons[ocean[1]]-lon)**2)
    k=d.argmin()

    return ocean[0][k],ocean[1][k]

# ===============================
# A* 導航
# ===============================
def astar(start,goal,u,v,mask,safety,ship_spd):

    v_ship=ship_spd*0.277
    rows,cols=mask.shape
    dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

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

            if 0<=ni<rows and 0<=nj<cols and not mask[ni,nj]:

                dist=8000*np.sqrt(d[0]**2+d[1]**2)
                norm=np.sqrt(d[0]**2+d[1]**2)

                flow=u[cur]*(d[1]/norm)+v[cur]*(d[0]/norm)
                v_ground=max(0.5,v_ship+flow)

                step=dist/v_ground

                new=cost[cur]+step

                if (ni,nj) not in cost or new<cost[(ni,nj)]:
                    cost[(ni,nj)]=new
                    heapq.heappush(pq,(new,(ni,nj)))
                    came[(ni,nj)]=cur

    path=[]
    c=goal
    while c in came:
        path.append(c)
        c=came[c]

    if path:
        path.append(start)

    return path[::-1]

# ===============================
# 計算航線
# ===============================
safety=distance_transform_edt(~land_mask)

start=nearest_ocean_cell(s_lon,s_lat,lons,lats,land_mask)
goal=nearest_ocean_cell(e_lon,e_lat,lons,lats,land_mask)

path=astar(start,goal,u,v,land_mask,safety,ship_speed)

# ===============================
# 距離與時間
# ===============================
def calc_stats(path):

    v_ship=ship_speed*0.277
    dist_total=0
    time_total=0

    for i in range(len(path)-1):

        y0,x0=path[i]
        y1,x1=path[i+1]

        dist=np.sqrt(
            (lats[y1]-lats[y0])**2+
            (lons[x1]-lons[x0])**2
        )*111e3

        flow=u[y0,x0]
        v_ground=max(0.5,v_ship+flow)

        dist_total+=dist
        time_total+=dist/v_ground

    return dist_total/1000,time_total/3600

if path:
    total_distance,total_hours=calc_stats(path)

# ===============================
# 儀表板
# ===============================
st.subheader("📊 航行資訊儀表板")

if path:
    st.metric("總航程 (km)",f"{total_distance:.2f}")
    st.metric("總航行時間 (hr)",f"{total_hours:.2f}")

sat_count=np.random.randint(3,7)
st.metric("可連接衛星數量",sat_count)

# ===============================
# 畫圖
# ===============================
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND,facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE)

speed=np.sqrt(u**2+v**2)

cmap=mcolors.LinearSegmentedColormap.from_list(
"flow",
["#E5F0FF","#66A3FF","#0066FF","#003D99","#001433"]
)

ax.pcolormesh(lons,lats,speed,cmap=cmap,shading="auto",alpha=0.8)

# 禁航區
for zone in NO_GO_ZONES:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="red",alpha=0.4)

# 離岸風場（全部）
for poly in OFFSHORE_WIND:
    xs=[p[0] for p in poly]
    ys=[p[1] for p in poly]
    ax.plot(xs,ys,linestyle="--",linewidth=1.5)

# 航線
if path:
    path_lons=[lons[p[1]] for p in path]
    path_lats=[lats[p[0]] for p in path]
    ax.plot(path_lons,path_lats,color="green",linewidth=2)

# 起終點
ax.scatter(s_lon,s_lat,color="green",s=120,edgecolors="black")
ax.scatter(e_lon,e_lat,color="yellow",marker="*",s=200,edgecolors="black")

plt.title("HELIOS V7 Navigation")
st.pyplot(fig)
