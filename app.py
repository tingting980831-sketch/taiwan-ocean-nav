import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from datetime import datetime, timedelta
from motuclient import MotuClient
import os

st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("HELIOS V7 智慧海洋導航系統")

# =========================
# HYCOM 海流
# =========================

@st.cache_data(ttl=3600)
def load_hycom():

    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"

    try:

        ds = xr.open_dataset(url, decode_times=False)

        lat_slice = slice(21,26)
        lon_slice = slice(118,124)

        u = ds["ssu"].sel(lat=lat_slice,lon=lon_slice).isel(time=-1).values
        v = ds["ssv"].sel(lat=lat_slice,lon=lon_slice).isel(time=-1).values

        lons = ds["lon"].sel(lon=lon_slice).values
        lats = ds["lat"].sel(lat=lat_slice).values

        speed = np.sqrt(u**2 + v**2)

        return lons,lats,u,v,speed

    except Exception as e:

        st.error(f"HYCOM讀取失敗 {e}")
        return None,None,None,None,None


# =========================
# CMEMS 下載
# =========================

def download_wave_wind():

    if os.path.exists("taiwan_wave_wind.nc"):
        return

    username="你的email"
    password="你的密碼"

    now=datetime.utcnow()
    start=now-timedelta(hours=3)

    date_min=start.strftime("%Y-%m-%d %H:%M:%S")
    date_max=now.strftime("%Y-%m-%d %H:%M:%S")

    client=MotuClient()

    try:

        client.execute(
            motu='https://nrt.cmems-du.eu/motu-web/Motu',
            service_id='GLOBAL_ANALYSIS_FORECAST_WAV_001_027',
            product_id='global-analysis-forecast-waves-001-027',

            longitude_min=118,
            longitude_max=123,
            latitude_min=21,
            latitude_max=26,

            date_min=date_min,
            date_max=date_max,

            variable=['swh','uwind','vwind'],

            out_dir='.',
            out_name='taiwan_wave_wind.nc',

            user=username,
            pwd=password
        )

        st.success("CMEMS下載完成")

    except Exception as e:

        st.warning(f"CMEMS下載失敗 {e}")


# =========================
# 讀取風與波
# =========================

@st.cache_data
def load_wave_wind():

    try:

        ds=xr.open_dataset("taiwan_wave_wind.nc")

        u=ds["uwind"].isel(time=0).values
        v=ds["vwind"].isel(time=0).values

        wind_speed=np.sqrt(u**2+v**2)
        wind_dir=(270-np.degrees(np.arctan2(v,u)))%360

        wave=ds["swh"].isel(time=0).values

        lons=ds["longitude"].values
        lats=ds["latitude"].values

        return lons,lats,wind_speed,wave,u,v

    except Exception as e:

        st.warning(f"波浪資料讀取失敗 {e}")
        return None,None,None,None,None,None


# =========================
# 讀取限制區
# =========================

def load_zones():

    try:
        no_go=gpd.read_file("no_go_area.geojson")
    except:
        no_go=gpd.GeoDataFrame()

    try:
        windfarm=gpd.read_file("offshore_wind.geojson")
    except:
        windfarm=gpd.GeoDataFrame()

    return no_go,windfarm


# =========================
# 下載資料
# =========================

download_wave_wind()

hlons,hlats,u,v,current_speed=load_hycom()

wlons,wlats,wind_speed,wave_height,wu,wv=load_wave_wind()

no_go,windfarm=load_zones()

# =========================
# UI
# =========================

layer=st.sidebar.radio(
"選擇底圖",
["海流","風場","波高"]
)

# =========================
# 繪圖
# =========================

fig=plt.figure(figsize=(10,8))

ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])

ax.add_feature(cfeature.LAND,facecolor="lightgray")

ax.add_feature(cfeature.COASTLINE)

gl=ax.gridlines(draw_labels=True,alpha=0.3)

gl.top_labels=False
gl.right_labels=False

# =========================
# 海流
# =========================

if layer=="海流":

    if current_speed is not None:

        im=ax.pcolormesh(
            hlons,
            hlats,
            current_speed,
            cmap="Blues",
            shading="auto",
            alpha=0.8
        )

        ax.quiver(
            hlons[::2],
            hlats[::2],
            u[::2,::2],
            v[::2,::2],
            color="white",
            alpha=0.5,
            scale=10
        )

        plt.colorbar(im,ax=ax,label="流速 m/s")


# =========================
# 風場
# =========================

elif layer=="風場":

    if wind_speed is not None:

        im=ax.pcolormesh(
            wlons,
            wlats,
            wind_speed,
            cmap="Oranges",
            shading="auto",
            alpha=0.8
        )

        ax.quiver(
            wlons[::2],
            wlats[::2],
            wu[::2,::2],
            wv[::2,::2],
            color="black",
            alpha=0.4
        )

        plt.colorbar(im,ax=ax,label="風速 m/s")


# =========================
# 波高
# =========================

elif layer=="波高":

    if wave_height is not None:

        im=ax.pcolormesh(
            wlons,
            wlats,
            wave_height,
            cmap="coolwarm",
            shading="auto",
            alpha=0.8
        )

        plt.colorbar(im,ax=ax,label="波高 m")


# =========================
# 禁航區
# =========================

if not no_go.empty:

    no_go.plot(
        ax=ax,
        facecolor="red",
        alpha=0.3,
        edgecolor="darkred"
    )

# =========================
# 離岸風場
# =========================

if not windfarm.empty:

    windfarm.plot(
        ax=ax,
        facecolor="yellow",
        alpha=0.3,
        edgecolor="gold"
    )


plt.title("HELIOS V7 海洋導航環境監測")

st.pyplot(fig)
