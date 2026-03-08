import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from scipy.ndimage import distance_transform_edt

# ==================== 1️⃣ 網頁設定 ====================
st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("🛰️ HELIOS V7 智慧導航控制台")

# ==================== 2️⃣ 讀取 HYCOM 海流資料 ====================
@st.cache_data(ttl=3600)
def load_hycom():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
        lat_slice, lon_slice = slice(21, 26), slice(118, 124)
        u = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1).values
        v = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1).values
        lons = ds['lon'].sel(lon=lon_slice).values
        lats = ds['lat'].sel(lat=lat_slice).values
        land_mask = np.isnan(u)
        return lons, lats, u, v, land_mask
    except Exception as e:
        st.error(f"HYCOM資料讀取失敗: {e}")
        return None, None, None, None, None

lons, lats, u, v, land_mask = load_hycom()

# ==================== 3️⃣ 讀取禁航區與離岸風場 ====================
try:
    no_go_area = gpd.read_file("no_go_area.geojson")
except:
    st.warning("禁航區資料缺失，暫時不顯示")
    no_go_area = gpd.GeoDataFrame()

try:
    offshore_wind = gpd.read_file("offshore_wind.geojson")
except:
    st.warning("離岸風場資料缺失，暫時不顯示")
    offshore_wind = gpd.GeoDataFrame()

# ==================== 4️⃣ 側邊欄選擇 ====================
plot_type = st.sidebar.radio("選擇底圖", ["海流", "風場", "波高"])

# ==================== 5️⃣ 繪圖 ====================
if lons is not None:
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118,124,21,26], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, zorder=3)

    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    if plot_type == "海流":
        speed = np.sqrt(u**2 + v**2)
        colors_list = ["#E5F0FF","#CCE0FF","#99C2FF","#66A3FF","#3385FF",
                       "#0066FF","#0052CC","#003D99","#002966","#001433","#000E24"]
        levels = np.linspace(0, 1.2, len(colors_list))
        cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_flow", list(zip(levels/1.2, colors_list)))
        im = ax.pcolormesh(lons, lats, speed, cmap=cmap_custom, shading='auto', alpha=0.8, zorder=1)
        ax.quiver(lons[::2], lats[::2], u[::2, ::2], v[::2, ::2], color='white', alpha=0.4, scale=10, zorder=4)
        cbar = ax.figure.colorbar(im, ax=ax, label='流速 (m/s)', shrink=0.6, pad=0.05)

    elif plot_type == "風場":
        st.info("風場資料目前未接入")

    elif plot_type == "波高":
        st.info("波高資料目前未接入")

    # 繪製禁航區（微透明紅色）
    if not no_go_area.empty:
        no_go_area.plot(ax=ax, facecolor='red', alpha=0.3, edgecolor='darkred', zorder=5)

    # 繪製離岸風場（微透明黃色）
    if not offshore_wind.empty:
        offshore_wind.plot(ax=ax, facecolor='yellow', alpha=0.3, edgecolor='goldenrod', zorder=6)

    plt.title(f"HELIOS V7 智慧導航監控 - {plot_type}")
    st.pyplot(fig)
