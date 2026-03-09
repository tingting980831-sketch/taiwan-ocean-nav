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
st.title("🛰️ HELIOS V6 智慧導航控制台")

# ===============================
# 模擬衛星
# ===============================
def get_visible_sats():
    return np.random.randint(2,5)

# ===============================
# 讀取 HYCOM 海流
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)
    time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
    latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
    lat_slice = slice(21,26)
    lon_slice = slice(118,124)
    u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
    v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
    lons = u_data['lon'].values
    lats = u_data['lat'].values
    u_val = np.nan_to_num(u_data.values)
    v_val = np.nan_to_num(v_data.values)
    land_mask = np.isnan(u_data.values)
    return lons,lats,u_val,v_val,land_mask,latest_time

lons,lats,u,v,land_mask,obs_time = load_hycom_data()

# ===============================
# 讀取 Open-Meteo 波高與風速
# ===============================
@st.cache_data(ttl=900)
def get_marine(lat,lon):
    try:
        wave_res = requests.get("https://marine-api.open-meteo.com/v1/marine", params={
            "latitude": lat, "longitude": lon,
            "hourly":["wave_height"], "forecast_days":1, "timezone":"Asia/Taipei"
        }).json()
        wind_res = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "hourly":["wind_speed_10m","wind_direction_10m"],
            "wind_speed_unit":"m/s", "forecast_days":1, "timezone":"Asia/Taipei"
        }).json()
        df_wave = pd.DataFrame(wave_res['hourly'])
        df_wind = pd.DataFrame(wind_res['hourly'])
        df = pd.merge(df_wave, df_wind, on="time")
        current_time = pd.Timestamp.now().floor('h')
        row = df[df['time']==current_time].iloc[0]
        wave = float(row["wave_height"])
        wind_speed = float(row["wind_speed_10m"])
        wind_dir = float(row["wind_direction_10m"])
        # 風向轉成 u/v
        rad = np.deg2rad(wind_dir)
        wind_u = wind_speed*np.sin(rad)
        wind_v = wind_speed*np.cos(rad)
        return wave, wind_u, wind_v
    except:
        return 1.0, np.zeros_like(u), np.zeros_like(v)

# ===============================
# 導航函數
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

def astar(start,goal,u,v,land_mask,safety,ship_spd,wave_factor=1.0,wind_factor=1.0):
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
            ni=cur[0]+d[0]
            nj=cur[1]+d[1]
            if 0<=ni<rows and 0<=nj<cols and not land_mask[ni,nj]:
                dist_m=np.sqrt(d[0]**2+d[1]**2)*8000
                norm=np.sqrt(d[0]**2+d[1]**2)
                flow=(u[cur]*(d[1]/norm)+v[cur]*(d[0]/norm))
                v_ground=max(0.5,v_ship+flow)
                step=(dist_m/v_ground)+(-flow*(dist_m/v_ship)*1.5)
                # 加波高與風阻力
                step += wave_factor*dist_m*0.05 + wind_factor*dist_m*0.03
                if safety[ni,nj]<4:
                    step+=12000/(safety[ni,nj]+0.2)**2
                new=cost[cur]+step
                if (ni,nj) not in cost or new<cost[(ni,nj)]:
                    cost[(ni,nj)]=new
                    priority=new+4*np.sqrt((ni-goal[0])**2+(nj-goal[1])**2)*8000/v_ship
                    heapq.heappush(pq,(priority,(ni,nj)))
                    came[(ni,nj)]=cur
    path=[]
    curr=goal
    while curr in came:
        path.append(curr)
        curr=came[curr]
    if path:
        path.append(start)
    return path[::-1]

# ===============================
# 側邊欄
# ===============================
with st.sidebar:
    st.header("航點")
    s_lon=st.number_input("起點經度",118.0,124.0,120.3)
    s_lat=st.number_input("起點緯度",21.0,26.0,22.6)
    e_lon=st.number_input("終點經度",118.0,124.0,122.0)
    e_lat=st.number_input("終點緯度",21.0,26.0,24.5)
    ship_speed=st.number_input("船速 km/h",1.0,60.0,20.0)

# ===============================
# 計算
# ===============================
if lons is not None:
    safety=distance_transform_edt(~land_mask)
    start=nearest_ocean_cell(s_lon,s_lat,lons,lats,land_mask)
    goal=nearest_ocean_cell(e_lon,e_lat,lons,lats,land_mask)

    wave, wind_u, wind_v = get_marine((s_lat+e_lat)/2,(s_lon+e_lon)/2)
    # 將波高與風速影響融入 A* 計算
    path=astar(start,goal,u,v,land_mask,safety,ship_speed,wave_factor=wave,wind_factor=np.sqrt(wind_u**2+wind_v**2))

    # ===============================
    # 儀表板
    # ===============================
    c1,c2,c3=st.columns(3)
    if path:
        dist_km=sum(np.sqrt(
            (lats[path[i][0]]-lats[path[i+1][0]])**2+
            (lons[path[i][1]]-lons[path[i+1][1]])**2
        ) for i in range(len(path)-1))*111
        c1.metric("航行時間",f"{dist_km/ship_speed:.1f} hr")
        c2.metric("航行距離",f"{dist_km:.1f} km")
    c3.metric("衛星數",f"{get_visible_sats()} SATS")
    st.caption(f"資料時間 {obs_time}")

    # ===============================
    # 2D 地圖 (全部因素總和)
    # ===============================
    total_field = np.sqrt(u**2+v**2) + np.sqrt(wind_u**2 + wind_v**2) + wave
    colors=["#E5F0FF","#CCE0FF","#99C2FF","#66A3FF","#3385FF","#0066FF","#0052CC","#003D99","#002966","#001433","#000E24"]
    cmap=mcolors.LinearSegmentedColormap.from_list("flow",colors)

    fig=plt.figure(figsize=(10,8))
    ax=plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118,124,21,26])
    ax.add_feature(cfeature.LAND,facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    im=ax.pcolormesh(lons,lats,total_field,cmap=cmap,shading='auto',alpha=0.8)

    # 繪製路徑
    if path:
        path_lons=[lons[p[1]] for p in path]
        path_lats=[lats[p[0]] for p in path]
        ax.plot(path_lons,path_lats,color='red',linewidth=2)
    ax.scatter(s_lon,s_lat,color='green',s=120,edgecolors='black')
    ax.scatter(e_lon,e_lat,color='yellow',marker='*',s=200,edgecolors='black')
    plt.title("HELIOS V6 Navigation (2D 海象總和)")
    st.pyplot(fig)

    # ===============================
    # 3D 海象模型
    # ===============================
    st.subheader("🌊 3D 海象模型")
    lon_grid,lat_grid=np.meshgrid(lons,lats)
    flow_speed=np.sqrt(u**2+v**2)
    wave_height=wave*1.5

    fig3d=go.Figure()
    fig3d.add_trace(go.Surface(
        x=lon_grid,
        y=lat_grid,
        z=wave_height,
        colorscale='Viridis',
        opacity=0.9,
        name="波高"
    ))
    fig3d.add_trace(go.Surface(
        x=lon_grid,
        y=lat_grid,
        z=flow_speed+wave_height,
        colorscale='Blues',
        opacity=0.7,
        name="流速"
    ))

    skip=3
    # 流速箭頭
    fig3d.add_trace(go.Cone(
        x=lon_grid[::skip,::skip].flatten(),
        y=lat_grid[::skip,::skip].flatten(),
        z=(flow_speed+wave_height)[::skip,::skip].flatten()+0.2,
        u=u[::skip,::skip].flatten(),
        v=v[::skip,::skip].flatten(),
        w=np.zeros_like(u[::skip,::skip].flatten()),
        sizemode="scaled",
        sizeref=0.5,
        showscale=False,
        anchor="tail"
    ))

    # 風速箭頭
    fig3d.add_trace(go.Cone(
        x=lon_grid[::skip,::skip].flatten(),
        y=lat_grid[::skip,::skip].flatten(),
        z=(flow_speed+wave_height)[::skip,::skip].flatten()+0.4,
        u=wind_u[::skip,::skip].flatten(),
        v=wind_v[::skip,::skip].flatten(),
        w=np.zeros_like(u[::skip,::skip].flatten()),
        sizemode="scaled",
        sizeref=0.5,
        showscale=False,
        anchor="tail",
        colorscale="Reds"
    ))

    # 路徑在最高層
    if path:
        path_lons=[lons[p[1]] for p in path]
        path_lats=[lats[p[0]] for p in path]
        fig3d.add_trace(go.Scatter3d(
            x=path_lons,
            y=path_lats,
            z=np.full(len(path_lons),flow_speed.max()+wave_height.max()+0.5),
            mode='lines',
            line=dict(color='red',width=6)
        ))

    fig3d.update_layout(
        scene=dict(
            xaxis_title="經度",
            yaxis_title="緯度",
            zaxis_title="強度"
        ),
        height=700
    )
    st.plotly_chart(fig3d,use_container_width=True)
