import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests

st.set_page_config(layout="wide")
st.title("⚓ 台灣海域智慧航行整合示意")

# ---------------------------
# 側邊欄選項
# ---------------------------
map_type = st.sidebar.selectbox(
    "選擇底圖",
    ["海流", "風向", "波高"]
)

# ---------------------------
# 1. 禁航區 GeoJSON
# ---------------------------
try:
    no_go_url = "https://ocean.moi.gov.tw/Map/Achievement/LayerInfo/905?utm_source=chatgpt.com"
    no_go = gpd.read_file(no_go_url)
    st.sidebar.success("禁航區資料載入成功")
except:
    no_go = None
    st.sidebar.warning("禁航區資料缺失，暫時不顯示")

# ---------------------------
# 2. 離岸風場 GeoJSON
# ---------------------------
try:
    offshore_wind_url = "https://example.com/offshore_wind.geojson"  # 公開來源替換
    wind_farm = gpd.read_file(offshore_wind_url)
    st.sidebar.success("離岸風場資料載入成功")
except:
    wind_farm = None
    st.sidebar.warning("離岸風場資料缺失，暫時不顯示")

# ---------------------------
# 3. CWB 即時風場/波高資料
# ---------------------------
try:
    API_KEY = "你的CWB_API_KEY"
    cwb_url = f"https://opendata.cwb.gov.tw/fileapi/v1/opendataapi/O-A0013-001?Authorization={API_KEY}&format=JSON"
    r = requests.get(cwb_url, timeout=10)
    r.raise_for_status()
    data_json = r.json()
    records = []
    for station in data_json['cwbopendata']['dataset']['location']:
        try:
            lat = float(station['lat'])
            lon = float(station['lon'])
            ws = float(station['weatherElement']['WDSD']['value'])
            wd = float(station['weatherElement']['WDIR']['value'])
            swh = float(station['weatherElement']['WVHT']['value'])
            records.append([station['stationName'], lat, lon, ws, wd, swh])
        except:
            continue
    df_cwb = pd.DataFrame(records, columns=['Station','Lat','Lon','WindSpeed','WindDir','WaveHeight'])
    df_cwb['Wind_u'] = df_cwb['WindSpeed'] * np.sin(np.deg2rad(df_cwb['WindDir']))
    df_cwb['Wind_v'] = df_cwb['WindSpeed'] * np.cos(np.deg2rad(df_cwb['WindDir']))
except:
    df_cwb = None
    st.sidebar.warning("CWB 即時風場資料抓取失敗")

# ---------------------------
# 4. 海流 (使用你原本的 HYCOM 方法)
# ---------------------------
try:
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)
    time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
    time_values = time_origin + pd.to_timedelta(ds['time'].values, unit='h')

    lat_slice = slice(20,26)
    lon_slice = slice(119,123)
    ssu = ds['ssu'].sel(lat=lat_slice, lon=lon_slice)
    ssv = ds['ssv'].sel(lat=lat_slice, lon=lon_slice)

    latest_index = -1
    ssu_latest = ssu.isel(time=latest_index)
    ssv_latest = ssv.isel(time=latest_index)
    latest_time = time_values[latest_index]

    lons = ds['lon'].sel(lon=lon_slice)
    lats = ds['lat'].sel(lat=lat_slice)
except:
    ssu_latest = ssv_latest = lons = lats = None
    st.sidebar.warning("HYCOM 海流資料抓取失敗")

# ---------------------------
# 5. 繪圖
# ---------------------------
fig = plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([119, 123, 21, 26], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='lightgray')

# 底圖
if map_type == "海流" and ssu_latest is not None:
    speed = np.sqrt(ssu_latest**2 + ssv_latest**2)
    ax.quiver(lons, lats, ssu_latest, ssv_latest, speed, scale=3, cmap='viridis', alpha=0.7)
elif map_type == "風向" and df_cwb is not None:
    ax.quiver(df_cwb['Lon'], df_cwb['Lat'], df_cwb['Wind_u'], df_cwb['Wind_v'],
              df_cwb['WindSpeed'], scale=5, cmap='YlOrRd', alpha=0.7)
elif map_type == "波高" and df_cwb is not None:
    sc = ax.scatter(df_cwb['Lon'], df_cwb['Lat'], c=df_cwb['WaveHeight'], cmap='cool', s=80, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='WaveHeight (m)')

# 禁航區
if no_go is not None:
    no_go.plot(ax=ax, facecolor='red', alpha=0.3, edgecolor='darkred')

# 離岸風場
if wind_farm is not None:
    wind_farm.plot(ax=ax, facecolor='yellow', alpha=0.3, edgecolor='orange')

st.pyplot(fig)
