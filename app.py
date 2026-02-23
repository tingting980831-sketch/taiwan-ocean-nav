# ===============================
# 正確比例、無白邊的流場繪圖
# ===============================

# --- 1. 計算經緯度範圍 ---
lon_min, lon_max = c_lon - 0.5, c_lon + 0.5
lat_min, lat_max = c_lat - 0.5, c_lat + 0.5

lon_range = lon_max - lon_min
lat_range = lat_max - lat_min

# --- 2. 依照地理比例計算 figsize ---
# 在台灣緯度，經度實際距離要乘 cos(lat)
mean_lat = (lat_min + lat_max) / 2
aspect_geo = (lon_range * np.cos(np.deg2rad(mean_lat))) / lat_range

# 固定高度，寬度依比例算（不會白邊）
fig_height = 7
fig_width = fig_height * aspect_geo

# --- 3. 建立圖表 ---
fig, ax = plt.subplots(
    figsize=(fig_width, fig_height),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# --- 4. 流速大小 ---
mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
land_mask = np.isnan(subset.water_u.values)
mag_masked = np.ma.masked_where(land_mask, mag)

cf = ax.pcolormesh(
    subset.lon,
    subset.lat,
    mag_masked,
    cmap='YlGn',
    shading='auto',
    alpha=0.9
)

plt.colorbar(cf, ax=ax, label='Current Speed (m/s)', shrink=0.55)

# --- 5. 地形 ---
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1e1e1e', zorder=5)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', linewidth=1.2, zorder=6)

# --- 6. 流向與船舶 ---
ax.quiver(
    c_lon, c_lat,
    u_val, v_val,
    color='red',
    scale=5,
    zorder=10
)

ax.scatter(
    c_lon, c_lat,
    color='#FF00FF',
    s=120,
    edgecolors='white',
    zorder=11
)

ax.set_title("HELIOS Real-time Ocean Current Field", fontsize=14)

st.pyplot(fig)
