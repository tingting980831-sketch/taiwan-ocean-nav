# ===============================
# Map (完全替換此區塊)
# ===============================
st.subheader("Interactive Navigation Map")

# 1. 準備地圖畫布
fig = plt.figure(figsize=(12, 10)) # 稍微放大一點，看得更清楚
ax = plt.axes(projection=ccrs.PlateCarree())

# 設定顯示範圍 (與 sidebar 的數值範圍一致)
ax.set_extent([118, 124, 21, 26], crs=ccrs.PlateCarree())

# 2. 加入地理特徵
# 使用 Cartopy 自帶的高解析度特徵
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor="#d0d0d0", edgecolor='black', linewidth=0.5, zorder=2)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1, zorder=3)
ax.add_feature(cfeature.OCEAN, facecolor="#f0f8ff", zorder=0) # 加入海洋底色

# 加入經緯度網格線
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# 3. 處理海流數據 (⭐核心修正)
time_idx = -1 # 取最新的時間點

# 取得 U/V 分量
u_da = ds['ssu'].sel(lat=slice(21, 26), lon=slice(118, 124)).isel(time=time_idx)
v_da = ds['ssv'].sel(lat=slice(21, 26), lon=slice(118, 124)).isel(time=time_idx)

# 計算流速
speed_da = np.sqrt(u_da**2 + v_da**2)

# ⭐ 關鍵修正 1：明確將維度轉置為 (lat, lon)，以匹配 pcolormesh 的預期
speed_for_plot = speed_da.transpose('lat', 'lon')

# ⭐ 關鍵修正 2：確保繪圖使用 PlateCarree 轉換
# 使用其 values 繪製，shading="nearest" 或 "auto" 均可，nearest 通常更穩定
mesh = ax.pcolormesh(speed_for_plot.lon, speed_for_plot.lat, speed_for_plot.values,
                    cmap="Blues", shading="nearest", vmin=0, vmax=1.6,
                    transform=ccrs.PlateCarree(), zorder=1, alpha=0.9)

# 4. 加入 Colorbar (⭐修正調用方式)
# 使用 plt.colorbar 並明確指定 ax
cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.03, shrink=0.7)
cbar.set_label("Current Speed (m/s)", fontsize=12)

# 5. 加入 No-Go Zones 與 風場 (加入 zorder 確保在海流之上)
for zone in NO_GO_ZONES:
    poly = np.array(zone)
    # Cartopy 的 fill 需要確保 transform 正確
    ax.fill(poly[:, 1], poly[:, 0], color="red", alpha=0.5, transform=ccrs.PlateCarree(), zorder=4, label='No-Go Zone')

for zone in OFFSHORE_WIND:
    poly = np.array(zone)
    ax.fill(poly[:, 1], poly[:, 0], color="yellow", alpha=0.5, transform=ccrs.PlateCarree(), zorder=4, label='Offshore Wind')

# 6. 繪製航段與船隻 (⭐加入 transform)
if len(path) > 0:
    full_lons = [lons[p[1]] for p in path]
    full_lats = [lats[p[0]] for p in path]
    
    # 完整規劃路徑 (粉色)
    ax.plot(full_lons, full_lats, color="#FF69B4", linewidth=3, 
            transform=ccrs.PlateCarree(), zorder=5, label='Planned Route')

    # 已走路徑 (紅色)
    done_lons = full_lons[:st.session_state.ship_step_idx + 1]
    done_lats = full_lats[:st.session_state.ship_step_idx + 1]
    ax.plot(done_lons, done_lats, color="red", linewidth=3, 
            transform=ccrs.PlateCarree(), zorder=6, label='Traveled Route')

    # 當前船隻位置 (灰色三角形)
    ax.scatter(lons[current_pos[1]], lats[current_pos[0]],
               color="#505050", marker="^", s=250, edgecolor='white',
               transform=ccrs.PlateCarree(), zorder=10)

# 7. 繪製起點與終點 (⭐加入 transform)
# 起點 (紫色圓點)
ax.scatter(s_lon, s_lat, color="#B15BFF", s=150, edgecolor="black", linewidth=1.5,
           transform=ccrs.PlateCarree(), zorder=12, label='Start')

# 終點 (黃色星星)
ax.scatter(e_lon, e_lat, color="yellow", marker="*", s=350, edgecolor="black", linewidth=1.5,
           transform=ccrs.PlateCarree(), zorder=12, label='End')

# 8. 介面優化
# 移除非必要的 plt.title，Streamlit 已經有大標題
# plt.title("HELIOS Dynamic Navigation", fontsize=16)

# 防止 labels 重疊
plt.tight_layout()

# 9. 將 Figure 傳給 Streamlit
st.pyplot(fig)
