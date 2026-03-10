import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests

st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("🛰️ HELIOS V7 智慧海象導航系統")

# ===============================
# 讀取 HYCOM 海流
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)
    try:
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
    except KeyError:
        latest_time = pd.Timestamp.now()
    lat_slice = slice(21,26)
    lon_slice = slice(118,124)
    u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
    v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
    lons = u_data['lon'].values
    lats = u_data['lat'].values
    u_val = np.nan_to_num(u_data.values)
    v_val = np.nan_to_num(v_data.values)
    land_mask = np.isnan(u_data.values)
    return lons, lats, u_val, v_val, land_mask, latest_time

lons, lats, u, v, land_mask, obs_time = load_hycom_data()

# ===============================
# 側邊欄設定
# ===============================
with st.sidebar:
    st.header("航點設定")
    s_lon = st.number_input("起點經度", 118.0, 124.0, 120.3)
    s_lat = st.number_input("起點緯度", 21.0, 26.0, 22.6)
    e_lon = st.number_input("終點經度", 118.0, 124.0, 122.0)
    e_lat = st.number_input("終點緯度", 21.0, 26.0, 24.5)
    ship_speed = st.number_input("船速 km/h", 1.0, 60.0, 20.0)

# ===============================
# 側邊欄: 禁航區 & 離岸風場 (手動)
# ===============================
NO_GO_ZONES = [
    [[22.953536, 120.171678], [22.934628, 120.175472], [22.933136, 120.170942], [22.957810, 120.160780]],
]

OFFSHORE_WIND = [
    [[24.18, 120.12], [24.22, 120.28], [24.05, 120.35], [24.00, 120.15]],
]

# ===============================
# 讀取即時風速資料 (CWB)
# ===============================
@st.cache_data(ttl=600)
def load_cwb_wind():
    try:
        url = "https://opendata.cwb.gov.tw/fileapi/v1/opendataapi/O-A0002-001?Authorization=你的CWB授權碼&format=JSON"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        lons_w, lats_w, u_w, v_w = [], [], [], []
        for station in data['cwbopendata']['location']:
            lon = float(station['lon'])
            lat = float(station['lat'])
            ws = float(station['WDIR'])  # 風向
            wv = float(station['WDSD'])  # 風速
            # 轉成 U/V
            theta = np.radians(ws)
            u_w.append(wv * np.sin(theta))
            v_w.append(wv * np.cos(theta))
            lons_w.append(lon)
            lats_w.append(lat)
        return np.array(lons_w), np.array(lats_w), np.array(u_w), np.array(v_w)
    except:
        return None, None, None, None

lons_w, lats_w, u_w, v_w = load_cwb_wind()

# ===============================
# 繪圖
# ===============================
if lons is not None:
    Lon, Lat = np.meshgrid(lons, lats)
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118,124,21,26])
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)

    # 海流底圖
    colors = ["#E5F0FF","#CCE0FF","#99C2FF","#66A3FF","#3385FF","#0066FF","#0052CC","#003D99","#002966","#001433","#000E24"]
    cmap = mcolors.LinearSegmentedColormap.from_list("flow", colors)
    speed = np.sqrt(u**2 + v**2)
    ax.pcolormesh(Lon, Lat, speed, cmap=cmap, shading='auto', alpha=0.6)

    # 禁航區
    for zone in NO_GO_ZONES:
        poly = np.array(zone)
        ax.fill(poly[:,1], poly[:,0], color='red', alpha=0.5, zorder=2)

    # 離岸風場
    for zone in OFFSHORE_WIND:
        poly = np.array(zone)
        ax.fill(poly[:,1], poly[:,0], color='yellow', alpha=0.5, zorder=1)

    # 起點/終點
    ax.scatter(s_lon, s_lat, color='green', s=120, edgecolors='black', zorder=3)
    ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, edgecolors='black', zorder=3)

    # 畫即時風速箭頭
    if lons_w is not None:
        ax.quiver(lons_w, lats_w, u_w, v_w, color='magenta', scale=30, width=0.005, zorder=4)

    plt.title("HELIOS V7 - 海流 + 禁航區 + 離岸風場 + 即時風速")
    st.pyplot(fig)
