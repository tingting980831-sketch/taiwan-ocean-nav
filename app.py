# --- 綠色系地圖繪製（正方形底圖） ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# 計算經緯度範圍，取最大邊長來設置正方形
lon_min, lon_max = c_lon-0.5, c_lon+0.5
lat_min, lat_max = c_lat-0.5, c_lat+0.5
lon_range = lon_max - lon_min
lat_range = lat_max - lat_min
max_range = max(lon_range, lat_range)

# 正方形範圍中心對齊
lon_center = (lon_max + lon_min) / 2
lat_center = (lat_max + lat_min) / 2
ax.set_extent([lon_center - max_range/2, lon_center + max_range/2,
               lat_center - max_range/2, lat_center + max_range/2])

mag = np.sqrt(subset.water_u**2 + subset.water_v**2)
land_mask = np.isnan(subset.water_u.values)
mag_masked = np.ma.masked_where(land_mask, mag)

# 使用 YlGn 綠色色階
cf = ax.pcolormesh(subset.lon, subset.lat, mag_masked, cmap='YlGn', shading='auto', alpha=0.9)
plt.colorbar(cf, label='Current Speed (m/s)', shrink=0.5)

ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#1e1e1e', zorder=5)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='white', linewidth=1.2, zorder=6)

ax.quiver(c_lon, c_lat, u_val, v_val, color='red', scale=5, zorder=10)
ax.scatter(c_lon, c_lat, color='#FF00FF', s=120, edgecolors='white', zorder=11)

st.pyplot(fig)
