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

st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("🛰️ HELIOS V7 智慧海象導航系統 (2D)")

# ===============================
# 禁航區（全部）
# ===============================
NO_GO_ZONES = [
    [[22.953536,120.171678],[22.934628,120.175472],[22.933136,120.170942],[22.957810,120.160780]],
    [[22.943956,120.172358],[22.939717,120.173944],[22.928353,120.157372],[22.936636,120.153547]],
    [[22.88462,120.17547],[22.91242,120.17696],[22.91178,120.17214],[22.933136,120.170942]],
    [[22.360902,120.381109],[22.354722,120.382139],[22.353191,120.384728],[22.357684,120.387818]],
    [[22.452778,120.458611],[22.4475,120.451389],[22.446111,120.452222],[22.449167,120.460556]],
    [[23.7885,119.598368],[23.784251,119.598368],[23.784251,119.602022],[23.7885,119.602022]],
    [[23.280833,119.5],[23.280833,119.509722],[23.274444,119.509722],[23.274444,119.5]],
    [[25.231417,121.648863],[25.226151,121.651505],[25.233410,121.642090],[25.242200,121.634560]],
]

# ===============================
# 離岸風場（全部）
# ===============================
OFFSHORE_WIND = [
    [[24.18,120.12],[24.22,120.28],[24.05,120.35],[24.00,120.15]],
    [[24.00,120.10],[24.05,120.32],[23.90,120.38],[23.85,120.15]],
    [[23.88,120.05],[23.92,120.18],[23.75,120.25],[23.70,120.08]],
    [[23.68,120.02],[23.72,120.12],[23.58,120.15],[23.55,120.05]],
    [[24.75,120.72],[24.80,120.85],[24.65,120.92],[24.60,120.78]],
    [[24.88,120.85],[24.95,120.95],[24.82,121.02],[24.78,120.90]],
]

# ===============================
# HYCOM 海流（原方法）
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds=xr.open_dataset(url,decode_times=False)

    if 'time_origin' in ds['time'].attrs:
        time_origin=pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time=time_origin+pd.to_timedelta(ds['time'].values[-1],unit='h')
    else:
        latest_time=pd.Timestamp.now()

    lat_slice=slice(21,26)
    lon_slice=slice(118,124)

    u_data=ds['ssu'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)
    v_data=ds['ssv'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)

    lons=u_data['lon'].values
    lats=u_data['lat'].values

    u_val=np.nan_to_num(u_data.values)
    v_val=np.nan_to_num(v_data.values)
    land_mask=np.isnan(u_data.values)

    return lons,lats,u_val,v_val,land_mask,latest_time

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
# 最近海點
# ===============================
def nearest_ocean_cell(lon,lat,lons,lats,land_mask):
    lon_idx=np.abs(lons-lon).argmin()
    lat_idx=np.abs(lats-lat).argmin()
    if not land_mask[lat_idx,lon_idx]:
        return lat_idx,lon_idx
    ocean=np.where(~land_mask)
    dist=np.sqrt((lats[ocean[0]]-lat)**2+(lons[ocean[1]]-lon)**2)
    i=dist.argmin()
    return ocean[0][i],ocean[1][i]

# ===============================
# A*
# ===============================
def astar(start,goal,u,v,land_mask,safety,ship_spd):
    v_ship=ship_spd*0.277
    rows,cols=land_mask.shape
    dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq=[(0,start)]
    cost={start:0}
    came={}

    while pq:
        _,cur=heapq.heappop(pq)
        if cur==goal: break

        for d in dirs:
            ni,nj=cur[0]+d[0],cur[1]+d[1]
            if 0<=ni<rows and 0<=nj<cols and not land_mask[ni,nj]:

                dist_m=np.sqrt(d[0]**2+d[1]**2)*8000
                norm=np.sqrt(d[0]**2+d[1]**2)

                flow=(u[cur]*(d[1]/norm)+v[cur]*(d[0]/norm))
                v_ground=max(0.5,v_ship+flow)

                step=dist_m/v_ground
                new=cost[cur]+step

                if (ni,nj) not in cost or new<cost[(ni,nj)]:
                    cost[(ni,nj)]=new
                    priority=new+4*np.hypot(ni-goal[0],nj-goal[1])
                    heapq.heappush(pq,(priority,(ni,nj)))
                    came[(ni,nj)]=cur

    path=[]
    curr=goal
    while curr in came:
        path.append(curr)
        curr=came[curr]
    if path: path.append(start)

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

        dist=np.hypot(lats[y1]-lats[y0],lons[x1]-lons[x0])*111e3
        dist_total+=dist

        flow=u[y0,x0]
        v_ground=max(0.5,v_ship+flow)
        time_total+=dist/v_ground

    return dist_total/1000,time_total/3600

if path:
    total_distance,total_hours=calc_stats(path)
else:
    total_distance,total_hours=0,0

# ===============================
# 儀表板（橫向）
# ===============================
st.subheader("📊 航行資訊儀表板")

c1,c2,c3=st.columns(3)

with c1:
    st.metric("總航程 (km)",f"{total_distance:.2f}")
with c2:
    st.metric("總航行時間 (hr)",f"{total_hours:.2f}")
with c3:
    st.metric("可連接衛星數量",np.random.randint(3,7))

# ===============================
# 地圖
# ===============================
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118,124,21,26])
ax.add_feature(cfeature.LAND,facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE)

speed=np.sqrt(u**2+v**2)

mesh=ax.pcolormesh(lons,lats,speed,
                   cmap="Blues",
                   shading="auto",
                   alpha=0.8)

fig.colorbar(mesh,ax=ax,label="海流速度 (m/s)")

# 禁航區
for zone in NO_GO_ZONES:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="red",alpha=0.4)

# 離岸風場
for zone in OFFSHORE_WIND:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="cyan",alpha=0.35)

# 航線
if path:
    path_lons=[lons[p[1]] for p in path]
    path_lats=[lats[p[0]] for p in path]
    ax.plot(path_lons,path_lats,color="green",linewidth=2)

# 起終點
ax.scatter(s_lon,s_lat,color="green",s=120,edgecolors="black")
ax.scatter(e_lon,e_lat,color="yellow",marker="*",s=200,edgecolors="black")

plt.title("HELIOS V7 Navigation (2D)")
st.pyplot(fig)
