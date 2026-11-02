"""
bathymetry.py -- 把海底地形 GeoTIFF 接進航線規劃: 水深 -> 各吃水的「可通行」遮罩

做的事:
  1. 讀 GeoTIFF (EPSG:3857 網路墨卡托 / 經緯度 / TWD97 TM2)
  2. 重新取樣到我們的 0.01 度網格 (雙線性; 可選「取鄰近最淺值」較保守)
  3. 依船的吃水算「所需水深」: 吃水 + 船行下沉量 + 浪致升沉 + 安全餘裕
  4. 水深 < 所需水深 的格點 = 不可通行

重要限制 (請在報告中說明):
  - 這是「地形」資料, 不是航行用海圖; 粗解析度 (約 1.65 公里) 只適合篩選大範圍淺區,
    港口與航道附近的水深不可靠
  - 垂直基準面不明 (平均海水面或海圖基準面?) 且未含潮高; 台灣西岸潮差大
  - 所需水深的係數 (下沉量公式、安全餘裕比例) 是常用經驗值, 不是法規值

用法:
    python bathymetry.py --tif C:\\路徑\\20261028_200MDTM.tif --out C:\\NODASS\\out_2023
    -> 印出各級船的所需水深與被擋掉的海域比例, 並畫出 bathymetry_check.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

R_MERC = 6378137.0


def _read_raw(path):
    from inspect_tif import read_rasterio, read_tifffile
    try:
        return read_rasterio(path)
    except ImportError:
        return read_tifffile(path)


def _source_lonlat(x0, y0, dx, dy, nx, ny, epsg):
    """每一欄的經度、每一列的緯度 (來源格點中心)。"""
    xs = x0 + dx * (np.arange(nx) + 0.5)
    ys = y0 - dy * (np.arange(ny) + 0.5)
    if epsg == 3857:
        lons = np.degrees(xs / R_MERC)
        lats = np.degrees(np.arctan(np.sinh(ys / R_MERC)))
        return lons, lats
    if epsg in (4326, 4490, 4979, None) and abs(x0) <= 360 and abs(y0) <= 90:
        return xs, ys
    raise ValueError(f"尚未支援的座標系統 EPSG:{epsg} (目前支援 3857 與經緯度)")


def load_depth_grid(tif_path, lons, lats, conservative_cells=0, nodata=None):
    """回傳 (R, C) 水深 (公尺, 正值向下); 陸地為負值, 超出檔案範圍為 NaN。
    lons, lats: 目標網格的一維座標 (遞增)。conservative_cells: 先取來源格點 (2k+1)^2 鄰域內最淺值。"""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import minimum_filter

    arr, x0, y0, dx, dy, epsg, nod = _read_raw(tif_path)
    arr = np.asarray(arr, dtype="float64")
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < 10 else arr[..., 0]
    nodata = nod if nodata is None else nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    ny, nx = arr.shape
    slons, slats = _source_lonlat(x0, y0, dx, dy, nx, ny, epsg)

    depth = -arr                                                  # 約定: 原檔負值=水深 -> 轉成正值向下
    if np.nanmedian(arr) > 0:                                     # 若原檔是「正值=水深」就不反轉
        depth = arr.copy()
    if conservative_cells > 0:
        k = 2 * conservative_cells + 1
        tmp = np.where(np.isfinite(depth), depth, 1e9)
        depth = np.where(np.isfinite(depth), minimum_filter(tmp, size=k), np.nan)

    if slats[0] > slats[-1]:                                      # 內插需要遞增
        slats, depth = slats[::-1], depth[::-1, :]
    if slons[0] > slons[-1]:
        slons, depth = slons[::-1], depth[:, ::-1]
    f = RegularGridInterpolator((slats, slons), depth, method="linear",
                                bounds_error=False, fill_value=np.nan)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    out = f(np.column_stack([LAT.ravel(), LON.ravel()])).reshape(LAT.shape)
    return out.astype("float32")


def squat_m(cb, v_kn):
    """Barrass 近似 (開闊水域, 單位 m, V 為節)。受限水道約為 2 倍, 此處不加。"""
    return cb * v_kn ** 2 / 100.0


def required_depth_m(ship, hs=1.5, ukc_frac=0.10, ukc_min=1.0, v_kn=None):
    """所需水深 = 吃水 + 下沉量 + 0.5 * 設計浪高 + 安全餘裕 (max(吃水 * ukc_frac, ukc_min))。"""
    v = ship.vs_kn if v_kn is None else v_kn
    return float(ship.draft + squat_m(ship.cb, v) + 0.5 * hs + max(ukc_frac * ship.draft, ukc_min))


def passable(depth, req_m):
    return np.isfinite(depth) & (depth >= req_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profiles", default=None)
    ap.add_argument("--draft-state", default="p50", choices=["p10", "p50", "p90"])
    ap.add_argument("--hs", type=float, default=1.5, help="設計浪高 (m)")
    ap.add_argument("--ukc-frac", type=float, default=0.10)
    ap.add_argument("--conservative-cells", type=int, default=0)
    a = ap.parse_args()

    import fuel_model as FM
    from analyze_speed import BBOX, GRID_RES
    prof = a.profiles or os.path.join(a.out, "ship_profiles_v2.csv")
    if not os.path.exists(prof):
        prof = os.path.join(a.out, "ship_profiles.csv")
    ships = FM.ships_from_profiles(prof, a.draft_state)

    lons = np.arange(BBOX["lon0"], BBOX["lon1"] + GRID_RES / 2, GRID_RES)
    lats = np.arange(BBOX["lat0"], BBOX["lat1"] + GRID_RES / 2, GRID_RES)
    depth = load_depth_grid(a.tif, lons, lats, a.conservative_cells)
    water = np.isfinite(depth) & (depth > 0)
    print(f"網格 {depth.shape}; 有水深的格點 {water.mean():.1%}; 超出檔案範圍 {np.isnan(depth).mean():.1%}")
    print(f"水深分位數 (只看水域, 公尺): " +
          ", ".join(f"{q}%={np.nanpercentile(depth[water], q):.0f}" for q in (1, 5, 25, 50, 75)))

    print(f"\n各級船所需水深 (吃水狀態 {a.draft_state}, 設計浪高 {a.hs} m, 餘裕 {a.ukc_frac:.0%}):")
    rows = []
    for k, sh in ships.items():
        req = required_depth_m(sh, a.hs, a.ukc_frac)
        blocked = float(((depth > 0) & (depth < req)).sum() / max(water.sum(), 1))
        rows.append((k, sh.draft, req, blocked))
        print(f"  {k:9s} 吃水 {sh.draft:5.1f} m -> 所需水深 {req:5.1f} m; 擋掉水域的 {blocked:.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from monthly_corr import setup_font
    setup_font("zh")
    pick = [k for k in ("cargo_m", "cargo_l", "tank_xl") if k in ships]
    if pick:
        fig, axs = plt.subplots(1, len(pick), figsize=(5.2 * len(pick), 6), squeeze=False)
        for ax, k in zip(axs[0], pick):
            req = required_depth_m(ships[k], a.hs, a.ukc_frac)
            img = np.full(depth.shape, np.nan)
            img[(depth <= 0)] = 0.0                                  # 陸地
            img[(depth > 0) & (depth < req)] = 1.0                   # 水深不足
            img[(depth >= req)] = 2.0                                # 可通行
            ax.imshow(img, origin="lower", extent=[lons[0], lons[-1], lats[0], lats[-1]],
                      cmap=matplotlib.colors.ListedColormap(["#9e9e9e", "#d62728", "#e8f1fa"]),
                      vmin=0, vmax=2, aspect=1 / np.cos(np.radians(24)))
            ax.set_title(f"{k}: 吃水 {ships[k].draft:.1f} m, 所需水深 {req:.1f} m\n(灰=陸地, 紅=水深不足)")
        fig.tight_layout()
        outp = os.path.join(a.out, "bathymetry_check.png")
        fig.savefig(outp, dpi=150)
        print("\n已輸出:", os.path.abspath(outp))


if __name__ == "__main__":
    main()
