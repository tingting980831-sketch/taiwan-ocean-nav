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
import plotly.graph_objects as go
import requests

st.set_page_config(layout="wide", page_title="HELIOS V6")
st.title("🛰 HELIOS V6 智慧導航控制台")

# ---------------------------
# 模擬衛星
# ---------------------------
def get_visible_sats():
    return np.random.randint(2,5)

# ---------------------------
# HYCOM 海流
# ---------------------------
@st.cache_data(ttl=3600)
def load_hycom():
    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds=xr.open_dataset(url,decode_times=False)
    lat_slice=slice(21,26)
    lon_slice=slice(118,124)
    u=ds['ssu'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)
    v=ds['ssv'].sel(lat=lat_slice,lon=lon_slice).isel(time=-1)
    lons=u['lon'].values
    lats=u['lat'].values
    u_val=np.nan_to_num(u.values)
    v_val=np.nan_to_num(v.values)
    land_mask=np.isnan(u.values)
    return lons,lats,u_val,v_val,land_mask

lons,lats,u,v,land_mask=load_hycom()

# ---------------------------
# Open-Meteo 波高與風
# ---------------------------
@st.cache_data(ttl=1800)
def get_marine(lat,lon):
    try:
        url="https://marine-api.open-meteo.com/v1/marine"
        params={
            "latitude":lat,
            "longitude":lon,
            "hourly":["wave_height","wind_speed_10m","wind_direction_10m"],
            "timezone":"Asia/Taipei"
        }
        r=requests.get(url,params=params).json()
        df=pd.DataFrame(r["hourly"])
        df["time"]=pd.to_datetime(df["time"])
        now=pd.Timestamp.now().floor("h")
        row=df[df["time"]==now]
        if row.empty:
            return 0,0,0
        wave=float(row["wave_height"].iloc[0])
        wind=float(row["wind_speed_10m"].iloc[0])
        wdir=float(row["wind_direction_10m"].iloc[0])
        return wave,wind,wdir
    except:
        return 0,0,0

# ---------------------------
# A* 導航
# ---------------------------
def nearest_ocean(lon,lat):
    lon_i=np.abs(lons-lon).argmin()
    lat_i=np.abs(lats-lat).argmin()
    if not land_mask[lat_i,lon_i]:
        return lat_i,lon_i
    ocean=np.where(~land_mask)
    d=np.sqrt((lats[ocean[0]]-lat)**2+(lons[ocean[1]]-lon)**2)
    i=d.argmin()
    return ocean[0][i],ocean[1][i]

def astar(start,goal,ship_speed):
    v_ship=ship_speed*0.277
    rows,cols=land_mask.shape
    dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    pq=[(0,start)]
    cost={start:0}
    came={}
    safety=distance_transform_edt(~land_mask)
    while pq:
        _,cur=heapq.heappop(pq)
        if cur==goal:
            break
        for d in dirs:
            ni=cur[0]+d[0]
            nj=cur[1]+d[1]
            if 0<=ni<rows and 0<=nj<cols and not land_mask[ni,nj]:
                dist=np.sqrt(d[0]**2+d[1]**2)*8000
                flow=(u[cur]*d[1]+v[cur]*d[0])
                v_ground=max(0.5,v_ship+flow)
                step=dist/v_ground
                if safety[ni,nj]<4:
                    step+=10000/(safety[ni,nj]+0.2)**2
                new=cost[cur]+step
                if (ni,nj) not in cost or new<cost[(ni,nj)]:
                    cost[(ni,nj)]=new
                    h=np.sqrt((ni-goal[0])**2+(nj-goal[1])**2)
                    heapq.heappush(pq,(new+h,(ni,nj)))
                    came[(ni,nj)]=cur
    path=[]
    c=goal
    while c in came:
        path.append(c)
        c=came[c]
    if path:
        path.append(start)
    return path[::-1]

# ---------------------------
# 側邊欄
# ---------------------------
with st.sidebar:
    st.header("航線設定")
    s_lon=st.number_input("起點經度",118.0,124.0,120.3)
    s_lat=st.number_input("起點緯度",21.0,26.0,22.6)
    e_lon=st.number_input("終點經度",118.0,124.0,122.0)
    e_lat=st.number_input("終點緯度",21.0,26.0,24.5)
    ship_speed=st.number_input("船速 km/h",1.0,60.0,20.0)

# ---------------------------
# 海象資訊
# ---------------------------
wave,wind_speed,wind_dir=get_marine((s_lat+e_lat)/2,(s_lon+e_lon)/2)
wind_u=wind_speed*np.sin(np.deg2rad(wind_dir))
wind_v=wind_speed*np.cos(np.deg2rad(wind_dir))

# ---------------------------
# 計算航線
# ---------------------------
start=nearest_ocean(s_lon,s_lat)
goal=nearest_ocean(e_lon,e_lat)
path=astar(start,goal,ship_speed)

# ---------------------------
# 儀表板
# ---------------------------
c1,c2,c3,c4=st.columns(4)
c1.metric("波高",f"{wave:.2f} m")
c2.metric("風速",f"{wind_speed:.1f} m/s")
c3.metric("衛星",get_visible_sats())
c4.metric("航線節點",len(path))

# ---------------------------
# 2D 海圖
# ---------------------------
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
speed=np.sqrt(u**2+v**2)

# 流速底圖（使用你的漸層配色）
im=ax.pcolormesh(lons,lats,speed,cmap=cmap,alpha=0.8)

# 流向箭頭
ax.quiver(lons[::2],lats[::2],u[::2,::2],v[::2,::2],color="white")

# 風速箭頭修正
lon_w, lat_w = np.meshgrid(lons[::4], lats[::4])
wind_u_grid = np.full_like(lon_w, wind_u)
wind_v_grid = np.full_like(lat_w, wind_v)
ax.quiver(lon_w, lat_w, wind_u_grid, wind_v_grid, color="red", scale=50)

# 波高透明層
ax.contourf(lons,lats,np.full_like(speed,wave),alpha=0.25,cmap="cool")

# 航線
if path:
    plon=[lons[p[1]] for p in path]
    plat=[lats[p[0]] for p in path]
    ax.plot(plon,plat,color="yellow",linewidth=3)

st.pyplot(fig)

# ---------------------------
# 3D 三明治模型
# ---------------------------
st.subheader("🌊 3D 海象模型")
lon_grid,lat_grid=np.meshgrid(lons,lats)
wave_surface=np.full_like(lon_grid,wave)

fig3d=go.Figure()

# 波高 surface
fig3d.add_trace(go.Surface(
    x=lon_grid,
    y=lat_grid,
    z=wave_surface+0.5,
    colorscale="Viridis",
    opacity=0.8,
    showscale=False
))

# 海流箭頭
fig3d.add_trace(go.Cone(
    x=lon_grid.flatten(),y=lat_grid.flatten(),z=wave_surface.flatten()+0.2,
    u=u.flatten(),v=v.flatten(),w=np.zeros_like(u.flatten()),sizeref=3,showscale=False
))

# 風速箭頭
fig3d.add_trace(go.Cone(
    x=lon_grid.flatten(),y=lat_grid.flatten(),z=wave_surface.flatten()+0.4,
    u=np.full_like(u.flatten(),wind_u),
    v=np.full_like(v.flatten(),wind_v),
    w=np.zeros_like(u.flatten()),
    sizeref=4,
    colorscale="Reds"
))

# 航線
if path:
    plon=[lons[p[1]] for p in path]
    plat=[lats[p[0]] for p in path]
    fig3d.add_trace(go.Scatter3d(
        x=plon,y=plat,z=np.full(len(plon),wave+0.6),
        mode="lines",line=dict(color="yellow",width=6)
    ))

fig3d.update_layout(
    height=700,
    scene=dict(
        xaxis_title="經度",
        yaxis_title="緯度",
        zaxis_title="高度"
    )
)
st.plotly_chart(fig3d,use_container_width=True)
