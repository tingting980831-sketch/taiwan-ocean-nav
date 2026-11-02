"""helios_map.py -- 底圖繪製 (與舊版畫面相同: 海流底圖、波高等值線、風向箭頭、禁航區、路徑)"""
import matplotlib.pyplot as plt
import numpy as np


def draw_map(snap, lon_lat_paths, zones, ship_pos, done_idx, start, end, title="HELIOS Navigation Map"):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118, 124, 21, 26])
    ax.add_feature(cfeature.LAND, facecolor="#b0b0b0")
    ax.add_feature(cfeature.COASTLINE)
    tr = ccrs.PlateCarree()
    lo, la, u, v = snap["cur"]
    mesh = ax.pcolormesh(lo, la, np.hypot(u, v), cmap="Blues", shading="auto", vmin=0, vmax=1.6, transform=tr)
    fig.colorbar(mesh, ax=ax, label="Current Speed (m/s)")
    if "wave" in snap:
        wlo, wla, swh = snap["wave"]
        X, Y = np.meshgrid(wlo, wla)
        cs = ax.contour(X, Y, swh, levels=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], cmap="cool", linewidths=1.2, transform=tr)
        ax.clabel(cs, fmt="%.1fm", fontsize=7, inline=True)
    if "wind" in snap:
        wlo, wla, wu, wv = snap["wind"]
        X, Y = np.meshgrid(wlo, wla)
        ax.quiver(X[::2, ::2], Y[::2, ::2], wu[::2, ::2], wv[::2, ::2], scale=200, color="white", alpha=0.5, transform=tr)
    for color, polys in zones:
        for z in polys:
            p = np.array(z)
            ax.fill(p[:, 1], p[:, 0], color=color, alpha=0.4, transform=tr)
    base, plan = lon_lat_paths
    if base is not None:
        ax.plot(base[0], base[1], color="black", linestyle="--", linewidth=1.5, transform=tr, label="Baseline (shortest, 5 km offshore)")
    ax.plot(plan[0], plan[1], color="pink", linewidth=2.5, transform=tr, label="HELIOS route")
    ax.plot(plan[0][:done_idx + 1], plan[1][:done_idx + 1], color="red", linewidth=2, transform=tr)
    ax.scatter(*ship_pos, color="gray", marker="^", s=150, zorder=5, transform=tr)
    ax.scatter(*start, color="#B15BFF", s=80, edgecolors="black", transform=tr)
    ax.scatter(*end, color="yellow", marker="*", s=200, edgecolors="black", transform=tr)
    ax.legend(loc="lower left", fontsize=8)
    plt.title(title)
    return fig
