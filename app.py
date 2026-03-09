import streamlit as st
import numpy as np
import requests
import xarray as xr
from scipy.ndimage import distance_transform_edt

st.set_page_config(layout="wide")

# ===============================
# 海象資料 (波高 + 風速)
# ===============================
def get_realtime_marine_data(lat, lon):

    # 波高
    marine_url = "https://marine-api.open-meteo.com/v1/marine"
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height",
        "timezone": "Asia/Taipei"
    }

    marine = requests.get(marine_url, params=marine_params).json()

    # 風
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Taipei"
    }

    weather = requests.get(weather_url, params=weather_params).json()

    try:
        wave = float(marine["hourly"]["wave_height"][0])
    except:
        wave = None

    try:
        wind_speed = float(weather["hourly"]["wind_speed_10m"][0])
        wind_dir = float(weather["hourly"]["wind_direction_10m"][0])
    except:
        wind_speed = None
        wind_dir = None

    return wave, wind_speed, wind_dir


# ===============================
# HYCOM 海流
# ===============================
@st.cache_data
def load_hycom():

    try:

        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest"

        ds = xr.open_dataset(url)

        lats = ds["lat"].values
        lons = ds["lon"].values

        u = ds["water_u"][0,0].values
        v = ds["water_v"][0,0].values

        time = str(ds["time"].values[0])

        return lons,lats,u,v,time

    except:

        # fallback
        lons = np.linspace(118,124,120)
        lats = np.linspace(21,26,120)

        u = np.zeros((120,120))
        v = np.zeros((120,120))

        return lons,lats,u,v,"HYCOM離線"


# ===============================
# 最近海格點
# ===============================
def nearest_ocean_cell(lon,lat,lons,lats):

    j = np.argmin(np.abs(lons-lon))
    i = np.argmin(np.abs(lats-lat))

    return (i,j)


# ===============================
# 簡化A*航線
# ===============================
def astar(start,goal):

    path=[start]

    i0,j0=start
    i1,j1=goal

    steps=100

    for k in range(steps):

        i=int(i0+(i1-i0)*k/steps)
        j=int(j0+(j1-j0)*k/steps)

        path.append((i,j))

    path.append(goal)

    return path


# ===============================
# 假衛星數
# ===============================
def get_visible_sats():
    return np.random.randint(18,32)


# ===============================
# UI
# ===============================

st.title("HELIOS 海洋導航系統")

col1,col2=st.columns(2)

with col1:

    s_lat=st.number_input("起點緯度",value=24.0)
    s_lon=st.number_input("起點經度",value=120.0)

with col2:

    e_lat=st.number_input("終點緯度",value=23.0)
    e_lon=st.number_input("終點經度",value=121.0)

ship_speed=st.slider("船速 (km/h)",10,60,30)

run=st.button("計算航線")


# ===============================
# 計算
# ===============================

if run:

    lons,lats,u,v,obs_time = load_hycom()

    start=nearest_ocean_cell(s_lon,s_lat,lons,lats)
    goal=nearest_ocean_cell(e_lon,e_lat,lons,lats)

    path=astar(start,goal)

    # 海象
    wave,wind_speed,wind_dir = get_realtime_marine_data(
        (s_lat+e_lat)/2,
        (s_lon+e_lon)/2
    )

    # 風場轉向量
    if wind_speed is not None and wind_dir is not None:

        theta=np.deg2rad(270-wind_dir)

        wind_u=wind_speed*np.cos(theta)
        wind_v=wind_speed*np.sin(theta)

    else:

        wind_u=0
        wind_v=0

    # 距離
    dist_km=sum(np.sqrt(
        (lats[path[i][0]]-lats[path[i+1][0]])**2+
        (lons[path[i][1]]-lons[path[i+1][1]])**2
    ) for i in range(len(path)-1))*111

    time_hr=dist_km/ship_speed


    # ===============================
    # Dashboard
    # ===============================

    c1,c2,c3=st.columns(3)

    c1.metric("航行時間",f"{time_hr:.1f} hr")
    c2.metric("航行距離",f"{dist_km:.1f} km")
    c3.metric("衛星數",f"{get_visible_sats()} SATS")

    wave_status="OK" if wave is not None else "未接到"
    wind_status="OK" if wind_speed is not None else "未接到"

    st.caption(
        f"HYCOM資料時間 {obs_time} | 波高資料: {wave_status}, 風速資料: {wind_status}"
    )

    if wave is not None:
        st.write(f"波高: {wave:.2f} m")

    if wind_speed is not None:
        st.write(f"風速: {wind_speed:.2f} m/s")
