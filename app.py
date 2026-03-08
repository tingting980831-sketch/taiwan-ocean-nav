import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("HELIOS V7 海洋導航環境監測")

# =========================
# HYCOM 海流
# =========================

@st.cache_data(ttl=3600)
def load_current():

    try:

        url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"

        ds = xr.open_dataset(url, decode_times=False)

        lat_slice = slice(21,26)
        lon_slice = slice(118,124)

        u = ds["ssu"].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
        v = ds["ssv"].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)

        speed = np.sqrt(u**2 + v**2)

        lons = u.lon.values
        lats = u.lat.values

        return lons, lats, u.values, v.values, speed.values

    except Exception as e:

        st.error(f"HYCOM資料讀取失敗: {e}")
        return None,None,None,None,None


# =========================
# CMEMS 風 + 波
# =========================

@st.cache_data(ttl=3600)
def load_wave_wind():

    try:

        url = "https://nrt.cmems-du.eu/thredds/dodsC/global-analysis-forecast-waves-001-027"

        ds = xr.open_dataset(url)

        lat_slice = slice(21,26)
        lon_slice = slice(118,124)

        wave = ds["swh"].sel(latitude=lat_slice, longitude=lon_slice).isel(time=0)

        u = ds["uwnd"].sel(latitude=lat_slice, longitude=lon_slice).isel(time=0)
        v = ds["vwnd"].sel(latitude=lat_slice, longitude=lon_slice).isel(time=0)

        wind_speed = np.sqrt(u**2 + v**2)

        lons = wave.longitude.values
        lats = wave.latitude.values

        return lons, lats, wind_speed.values, wave.values, u.values, v.values

    except Exception as e:

        st.warning(f"CMEMS資料讀取失敗: {e}")
        return None,None,None,None,None,None


# =========================
# 限制區
# =========================

def load_zones():

    try:
        no_go = gpd.read_file("no_go_area.geojson")
    except:
        no_go = gpd.GeoDataFrame()

    try:
        windfarm = gpd.read_file("offshore_wind.geojson")
    except:
        windfarm = gpd.GeoDataFrame()

    return no_go, windfarm


# =========================
# 載入資料
# =========================

hlons, hlats, cu, cv, current_speed = load_current()

wlons, wlats, wind_speed, wave_height, wu, wv = load_wave_wind()

no_go, windfarm = load_zones()


# =========================
# UI
# =========================

layer = st.sidebar.radio(
"選擇底圖",
["海流","風場","波高"]
)

# =========================
# 地圖
# =========================

fig = plt.figure(figsize=(10,8))

ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,124,21,26])

ax.add_feature(cfeature.LAND, facecolor="lightgray")

ax.add_feature(cfeature.COASTLINE)

gl = ax.gridlines(draw_labels=True, alpha=0.3)

gl.top_labels = False
gl.right_labels = False


# =========================
# 海流
# =========================

if layer == "海流":

    if current_speed is not None:

        im = ax.pcolormesh(
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
            cu[::2,::2],
            cv[::2,::2],
            color="white",
            alpha=0.5,
            scale=10
        )

        plt.colorbar(im, ax=ax, label="流速 m/s")


# =========================
# 風場
# =========================

elif layer == "風場":

    if wind_speed is not None:

        im = ax.pcolormesh(
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

        plt.colorbar(im, ax=ax, label="風速 m/s")


# =========================
# 波高
# =========================

elif layer == "波高":

    if wave_height is not None:

        im = ax.pcolormesh(
            wlons,
            wlats,
            wave_height,
            cmap="coolwarm",
            shading="auto",
            alpha=0.8
        )

        plt.colorbar(im, ax=ax, label="波高 m")


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


plt.title("HELIOS V7 海洋導航系統")

st.pyplot(fig)
