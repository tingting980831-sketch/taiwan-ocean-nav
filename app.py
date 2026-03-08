import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

# ---------------------
# Streamlit 側邊欄
# ---------------------
st.sidebar.header("📍 選項")
base_map = st.sidebar.selectbox("選擇底圖", ["海流流速", "風向", "波高"])
s_lon = st.sidebar.number_input("起點經度", 118, 124, 120.3)
s_lat = st.sidebar.number_input("起點緯度", 21, 26, 22.6)
e_lon = st.sidebar.number_input("終點經度", 118, 124, 122.0)
e_lat = st.sidebar.number_input("終點緯度", 21, 26, 24.5)

# ---------------------
# 讀取資料（示範用隨機）
# ---------------------
lons = np.linspace(118, 124, 60)
lats = np.linspace(21, 26, 50)
lon2d, lat2d = np.meshgrid(lons, lats)

# 海流
u_flow = np.sin(lat2d) * 0.5
v_flow = np.cos(lon2d) * 0.5
speed_flow = np.sqrt(u_flow**2 + v_flow**2)

# 風場（TRITON）
u_wind = np.sin(lat2d) * 1.0
v_wind = np.cos(lon2d) * 1.0
speed_wind = np.sqrt(u_wind**2 + v_wind**2)

# 波高
swh = 1 + 0.5*np.sin(lat2d) * np.cos(lon2d)

# 禁航區與離岸風電（示範矩形）
no_go_zone = gpd.GeoDataFrame(geometry=[gpd.GeoSeries.from_bbox((120,22,121,23)).unary_union])
wind_zone = gpd.GeoDataFrame(geometry=[gpd.GeoSeries.from_bbox((121,23,122,24)).unary_union])

# ---------------------
# 繪圖
# ---------------------
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118,124,21,26])

# 海岸線與土地
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)

# 選擇底圖
if base_map=="海流流速":
    im = ax.pcolormesh(lons, lats, speed_flow, cmap="Blues", shading='auto')
    ax.streamplot(lons, lats, u_flow, v_flow, color='white', density=1.5, arrowsize=1, alpha=0.7)
elif base_map=="風向":
    im = ax.pcolormesh(lons, lats, speed_wind, cmap="YlOrBr", shading='auto')
    ax.streamplot(lons, lats, u_wind, v_wind, color='yellow', density=1.5, arrowsize=1, alpha=0.4)
else:  # 波高
    im = ax.pcolormesh(lons, lats, swh, cmap="Purples", shading='auto')

# ---------------------
# 畫禁航區與離岸風電
# ---------------------
no_go_zone.plot(ax=ax, facecolor='red', alpha=0.3, edgecolor='red', zorder=5)
wind_zone.plot(ax=ax, facecolor='yellow', alpha=0.3, edgecolor='yellow', zorder=4)

# ---------------------
# 調整網格與色條
# ---------------------
gl = ax.gridlines(draw_labels=True, left_labels=True, bottom_labels=True, right_labels=False, top_labels=False)
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.7)
cbar.set_label(base_map)

st.pyplot(fig)
