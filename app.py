import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer
import heapq
import json

# ===============================
# UI
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("🛰️ HELIOS V7 智慧海象導航系統")

# ===============================
# 禁航區（示例，可持續擴充）
# ===============================
NO_GO_ZONES = [
    [[22.953536,120.171678],[22.934628,120.175472],
     [22.933136,120.170942],[22.957810,120.160780]],
]

# ===============================
# 讀取離岸風場 GeoJSON
# ===============================
@st.cache_data
def load_offshore_wind_geojson(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transformer = Transformer.from_crs(
        "EPSG:3826", "EPSG:4326", always_xy=True
    )

    wind_polygons = []

    for feature in data["features"]:
        geom = feature["geometry"]

        if geom["type"] == "MultiPolygon":
            for polygon in geom["coordinates"]:
                for ring in polygon:

                    converted = []
                    for x, y in ring:
                        lon, lat = transformer.transform(x, y)
                        converted.append([lat, lon])

                    wind_polygons.append(converted)

    return wind_polygons


OFFSHORE_WIND = load_offshore_wind_geojson(
    "offshore_wind.geojson"
)

# ===============================
# HYCOM 海流（你的原始成功方法）
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():

    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)

    if 'time_origin' in ds['time'].attrs:
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(
            ds['time'].values[-1], unit='h')
    else:
        latest_time = pd.Timestamp.now()

    lat_slice = slice(21,26)
    lon_slice = slice(118,124)

    u = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
    v = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)

    lons = u['lon'].values
    lats = u['lat'].values

    u_val = np.nan_to_num(u.values)
    v_val = np.nan_to_num(v.values)
    land_mask = np.isnan(u.values)

    return lons, lats, u_val, v_val, land_mask, latest_time


lons, lats, u, v, land_mask, obs_time = load_hycom_data()

# ===============================
# 側邊欄
# ===============================
with st.sidebar:
    st.header("航點設定")

    s_lon = st.number_input("起點經度",118.0,124.0,120.3)
    s_lat = st.number_input("起點緯度",21.0,26.0,22.6)

    e_lon = st.number_input("終點經度",118.0,124.0,122.0)
    e_lat = st.number_input("終點緯度",21.0,26.0,24.5)

    ship_speed = st.number_input("船速 km/h",1.0,60.0,20.0)

# ===============================
# 最近海洋格點
# ===============================
def nearest_ocean_cell(lon, lat):
    lon_idx = np.abs(lons-lon).argmin()
    lat_idx = np.abs(lats-lat).argmin()

    if not land_mask[lat_idx,lon_idx]:
        return lat_idx,lon_idx

    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]]-lat)**2 +
                   (lons[ocean[1]]-lon)**2)
    i = dist.argmin()
    return ocean[0][i], ocean[1][i]

# ===============================
# 判斷是否進入風場
# ===============================
def inside_windfarm(lat, lon):

    for zone in OFFSHORE_WIND:
        poly = np.array(zone)

        if (lat > poly[:,0].min() and
            lat < poly[:,0].max() and
            lon > poly[:,1].min() and
            lon < poly[:,1].max()):
            return True
    return False

# ===============================
# A* 導航
# ===============================
def astar(start, goal):

    v_ship = ship_speed*0.277
    safety = distance_transform_edt(~land_mask)

    rows,cols = land_mask.shape
    dirs=[(1,0),(-1,0),(0,1),(0,-1),
          (1,1),(1,-1),(-1,1),(-1,-1)]

    pq=[(0,start)]
    cost={start:0}
    came={}

    while pq:
        _,cur = heapq.heappop(pq)

        if cur==goal:
            break

        for d in dirs:
            ni,nj=cur[0]+d[0],cur[1]+d[1]

            if not(0<=ni<rows and 0<=nj<cols):
                continue
            if land_mask[ni,nj]:
                continue

            dist=8000*np.sqrt(d[0]**2+d[1]**2)

            norm=np.sqrt(d[0]**2+d[1]**2)
            flow=(u[cur]*(d[1]/norm)+v[cur]*(d[0]/norm))

            v_ground=max(0.5,v_ship+flow)
            step=dist/v_ground

            # ⭐ 風場避讓
            lat=lats[ni]
            lon=lons[nj]
            if inside_windfarm(lat,lon):
                step+=15000

            if safety[ni,nj]<4:
                step+=12000/(safety[ni,nj]+0.2)**2

            new=cost[cur]+step

            if (ni,nj) not in cost or new<cost[(ni,nj)]:
                cost[(ni,nj)]=new
                priority=new+np.hypot(
                    ni-goal[0],nj-goal[1])*8000/v_ship

                heapq.heappush(pq,(priority,(ni,nj)))
                came[(ni,nj)]=cur

    path=[]
    c=goal
    while c in came:
        path.append(c)
        c=came[c]
    path.append(start)

    return path[::-1]


start = nearest_ocean_cell(s_lon,s_lat)
goal  = nearest_ocean_cell(e_lon,e_lat)

path = astar(start,goal)

# ===============================
# 儀表板
# ===============================
def calc_distance_time(path):

    v_ship=ship_speed*0.277
    dist_sum=0
    time_sum=0

    for i in range(len(path)-1):
        y0,x0=path[i]
        y1,x1=path[i+1]

        dist=np.sqrt(
            (lats[y1]-lats[y0])**2+
            (lons[x1]-lons[x0])**2)*111e3

        dist_sum+=dist
        time_sum+=dist/v_ship

    return dist_sum/1000,time_sum/3600

total_km,total_hr = calc_distance_time(path)

st.subheader("📊 航行資訊")
c1,c2,c3=st.columns(3)
c1.metric("總航程 (km)",f"{total_km:.2f}")
c2.metric("預估時間 (hr)",f"{total_hr:.2f}")
c3.metric("可連線衛星",np.random.randint(3,7))

# ===============================
# 地圖
# ===============================
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND,facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)

speed=np.sqrt(u**2+v**2)

cmap=mcolors.LinearSegmentedColormap.from_list(
"flow",
["#E5F0FF","#66A3FF","#0066FF","#002966","#000E24"]
)

ax.pcolormesh(lons,lats,speed,cmap=cmap,
              shading='auto',alpha=0.8)

# 禁航區
for zone in NO_GO_ZONES:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color='red',alpha=0.4)

# 離岸風場
for zone in OFFSHORE_WIND:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color='yellow',alpha=0.35)

# 航線
path_lons=[lons[p[1]] for p in path]
path_lats=[lats[p[0]] for p in path]
ax.plot(path_lons,path_lats,color='lime',linewidth=2)

# 起終點
ax.scatter(s_lon,s_lat,color='green',s=120,edgecolors='black')
ax.scatter(e_lon,e_lat,color='yellow',
           marker='*',s=220,edgecolors='black')

plt.title("HELIOS V7 Navigation")
st.pyplot(fig)
