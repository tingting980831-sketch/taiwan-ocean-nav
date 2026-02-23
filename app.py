# ===============================
# 正確比例、無白邊（A* 航線版）
# ===============================

# --- 1. 以起訖點中心作為顯示中心 ---
center_lon = (s_lon + e_lon) / 2
center_lat = (s_lat + e_lat) / 2

lon_min, lon_max = center_lon - 0.5, center_lon + 0.5
lat_min, lat_max = center_lat - 0.5, center_lat + 0.5

lon_range = lon_max - lon_min
lat_range = lat_max - lat_min

# --- 2. 計算地理比例（避免胖矮） ---
mean_lat = (lat_min + lat_max) / 2
aspect_geo = (lon_range * np.cos(np.deg2rad(mean_lat))) / lat_range

fig_height = 8
fig_width = fig_height * aspect_geo

# --- 3. 建立圖表 ---
fig, ax = plt.subplots(
    figsize=(fig_width, fig_height),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# --- 4. 流場 ---
speed = np.sqrt(subset.water_u**2 + subset.water_v**2)
land_mask = np.isnan(subset.water_u.values)
speed_masked = np.ma.masked_where(land_mask, speed)

cf = ax.pcolormesh(
    subset.lon,
    subset.lat,
    speed_masked,
    cmap='viridis',
    shading='auto',
    alpha=0.8
)

plt.colorbar(cf, ax=ax, label='Current Speed (m/s)', shrink=0.55)

# --- 5. 地形 ---
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#222222', zorder=5)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', linewidth=1.2, zorder=6)

# --- 6. 航線 ---
ax.plot(path_lon, path_lat, color='magenta', linewidth=3, label='AI Path', zorder=10)
ax.scatter(s_lon, s_lat, color='yellow', s=100, label='Start', zorder=11)
ax.scatter(e_lon, e_lat, color='red', marker='*', s=200, label='Goal', zorder=11)

ax.set_title("AI Marine Navigation & Obstacle Avoidance", fontsize=14)
ax.legend(loc='lower right')

st.pyplot(fig)fontsize=14)

st.pyplot(fig)
