"""
analyze_speed.py -- 各級船舶「開闊水域服務航速」

為什麼要重做: 之前的巡航速度把進出港、壅塞、錨泊附近的低速紀錄也算進去, 數值偏低。
這裡只取「離岸 >= 指定距離」且「SOG >= 門檻」的紀錄, 並與不限制離岸的結果並排比較。

需與 ais_profile.py、monthly_corr.py 放同一資料夾; 需要 monthly_corr.py 已跑完
(要用 <out>/daily/ 的每日快取來決定每艘船屬於哪一級)。

用法:
    python analyze_speed.py --ais-dir C:\\NODASS\\AIS --out C:\\NODASS\\out_2023 --year 2023
    python analyze_speed.py ... --all-days          # 整年 (很慢)
    python analyze_speed.py ... --min-dist-km 27.8 # 改成 15 海里

選項:
    --days 3,10,17,24   每月取哪幾天 (預設, 與 monthly_corr 相同)
    --min-dist-km 18.5  離岸門檻 (預設 10 海里)
    --min-sog 5         SOG 下限(節), 排除停泊/漂流/拖網作業
    --min-reports 30    一艘船至少要有幾筆合格紀錄才採用
    --land-shp PATH     自備陸地 shapefile (預設用 cartopy Natural Earth 10m)

輸出 (<out>):
    speed/speed_YYYYMMDD_d<離岸km>_s<SOG>.csv   每日每船統計 (快取, 參數不同會分開存, 可中斷續跑)
    service_speed_by_class.csv 各級服務航速: 開闊水域 vs 不限制
    service_speed.png          比較圖
    ship_profiles_v2.csv       若 ship_profiles.csv 存在, 併入服務航速欄位
"""
import argparse
import calendar
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ais_profile as ap                                          # noqa: E402
from monthly_corr import CLASSES, assign_class, setup_font        # noqa: E402

KN2KMH = 1.852
BBOX = dict(lon0=118.0, lon1=124.0, lat0=21.0, lat1=26.0)
GRID_RES = 0.01          # 度, 約 1.1 km
RATIO_LO, RATIO_HI = 0.015, 0.20
DAILY_NEEDED = {"MMSI", "loa", "beam", "draft_p50", "cruise_sog", "cruise_sog_p90", "type_code"}


# ------------------------------------------------------------------
# 離岸距離網格
# ------------------------------------------------------------------
def load_land(shp_path=None):
    """回傳涵蓋研究範圍(含緩衝)的陸地 shapely 幾何。"""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box
    from shapely.ops import unary_union

    if shp_path is None:
        shp_path = shpreader.natural_earth(resolution="10m", category="physical", name="land")
    clip = box(BBOX["lon0"] - 2, BBOX["lat0"] - 2, BBOX["lon1"] + 2, BBOX["lat1"] + 2)
    geoms = []
    for g in shpreader.Reader(shp_path).geometries():
        if g.intersects(clip):
            geoms.append(g.intersection(clip))
    if not geoms:
        sys.exit("陸地資料裡沒有涵蓋研究範圍的幾何, 請檢查 --land-shp")
    return unary_union(geoms)


def build_dist_grid(land_geom):
    """每個 0.01 度網格到最近陸地的距離(km)。"""
    try:
        import shapely
        contains_xy = shapely.contains_xy
    except (ImportError, AttributeError):
        from shapely.vectorized import contains as contains_xy

    lons = np.arange(BBOX["lon0"], BBOX["lon1"] + GRID_RES / 2, GRID_RES)
    lats = np.arange(BBOX["lat0"], BBOX["lat1"] + GRID_RES / 2, GRID_RES)
    LON, LAT = np.meshgrid(lons, lats)
    land = np.asarray(contains_xy(land_geom, LON.ravel(), LAT.ravel())).reshape(LON.shape)
    dy_km = GRID_RES * 111.2
    dx_km = GRID_RES * 111.2 * np.cos(np.radians(23.5))
    dist = distance_transform_edt(~land, sampling=(dy_km, dx_km))
    return dist


def lookup(dist, lon, lat):
    ix = np.clip(np.rint((lon - BBOX["lon0"]) / GRID_RES).astype(int), 0, dist.shape[1] - 1)
    iy = np.clip(np.rint((lat - BBOX["lat0"]) / GRID_RES).astype(int), 0, dist.shape[0] - 1)
    return dist[iy, ix]


# ------------------------------------------------------------------
# 每艘船屬於哪一級 (沿用每日快取, 與 analyze_year 相同邏輯)
# ------------------------------------------------------------------
def build_vessel_classes(daily_dir, year):
    files = sorted(glob.glob(os.path.join(daily_dir, f"vessels_{year}*.csv")))
    if not files:
        sys.exit(f"找不到 {daily_dir}\\vessels_{year}*.csv, 請先跑完 monthly_corr.py")
    frames = [pd.read_csv(f, dtype={"MMSI": str}, usecols=lambda c: c in DAILY_NEEDED)
              for f in files]
    daily = pd.concat(frames, ignore_index=True)
    g = daily.groupby("MMSI")
    yv = pd.DataFrame({"loa": g["loa"].median(), "beam": g["beam"].median(),
                       "cruise_sog_p90": g["cruise_sog_p90"].median()})
    yv["type_code"] = ap.mode_nonzero(daily, "MMSI", "type_code")
    yv["group"] = yv["type_code"].apply(ap.type_group)
    yv["cls"] = yv.apply(assign_class, axis=1)
    return yv[yv["cls"].notna()]


# ------------------------------------------------------------------
# 每日處理 (含快取)
# ------------------------------------------------------------------
def process_day(path, cache, dist, valid_mmsi, args):
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"MMSI": str})
    if not os.path.exists(path):
        print(f"  [缺檔] {path}")
        return None
    cols = ["MMSI", "SOG", "Longitude", "Latitude", "Navigational_Status"]
    df = pd.read_csv(path, usecols=cols, dtype={"MMSI": str}, low_memory=False)
    df = df[df["MMSI"].isin(valid_mmsi)]
    inside = df["Longitude"].between(BBOX["lon0"], BBOX["lon1"]) & \
             df["Latitude"].between(BBOX["lat0"], BBOX["lat1"])
    df = df[inside]
    ok = df["SOG"].between(args.min_sog, 40) & \
         (df["Navigational_Status"].isin([0, 8, 15]) | df["Navigational_Status"].isna())
    df = df[ok].copy()
    df["d_km"] = lookup(dist, df["Longitude"].values, df["Latitude"].values)
    df = df[df["d_km"] > 0]                                  # 排除落在陸地上的錯誤定位

    g_all = df.groupby("MMSI")["SOG"].agg(n_all="size", med_all="median")
    op = df[df["d_km"] >= args.min_dist_km]
    g_op = op.groupby("MMSI")["SOG"].agg(n_open="size", med_open="median")
    p90 = op.groupby("MMSI")["SOG"].quantile(.9).rename("p90_open")
    out = g_all.join(g_op).join(p90).reset_index()
    out.to_csv(cache, index=False)
    print(f"  {os.path.basename(path)}: {len(out):,} 艘, 開闊水域紀錄 {len(op):,}", flush=True)
    return out


def q(s, p):
    s = s.dropna()
    return float(s.quantile(p)) if len(s) else np.nan


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ais-dir", required=True)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--year", type=int, default=2023)
    ap_.add_argument("--days", default="3,10,17,24")
    ap_.add_argument("--all-days", action="store_true")
    ap_.add_argument("--min-dist-km", type=float, default=18.5)
    ap_.add_argument("--min-sog", type=float, default=5.0)
    ap_.add_argument("--min-reports", type=int, default=30)
    ap_.add_argument("--land-shp", default=None)
    ap_.add_argument("--lang", default="zh", choices=["zh", "en"])
    a = ap_.parse_args()
    setup_font(a.lang)

    speed_dir = os.path.join(a.out, "speed")
    os.makedirs(speed_dir, exist_ok=True)

    print("建立船舶分級...")
    yv = build_vessel_classes(os.path.join(a.out, "daily"), a.year)
    valid = set(yv.index)
    print(f"可分級船舶: {len(valid):,}")

    print("建立離岸距離網格...")
    dist = build_dist_grid(load_land(a.land_shp))
    print(f"  最大離岸距離 {dist.max():.0f} km;  研究範圍內 <{a.min_dist_km:.1f} km 的海域比例 "
          f"{((dist > 0) & (dist < a.min_dist_km)).sum() / (dist > 0).sum():.1%}")

    pick = [int(x) for x in a.days.split(",")]
    frames = []
    for month in range(1, 13):
        nd = calendar.monthrange(a.year, month)[1]
        days = range(1, nd + 1) if a.all_days else [d for d in pick if d <= nd]
        for d in days:
            ymd = f"{a.year}{month:02d}{d:02d}"
            r = process_day(os.path.join(a.ais_dir, f"{ymd}.csv"),
                            os.path.join(speed_dir, f"speed_{ymd}_d{a.min_dist_km:g}_s{a.min_sog:g}.csv"), dist, valid, a)
            if r is not None and len(r):
                frames.append(r)
    if not frames:
        sys.exit("沒有任何資料")

    allr = pd.concat(frames, ignore_index=True)
    g = allr.groupby("MMSI")
    vs = pd.DataFrame({
        "n_all": g["n_all"].sum(), "n_open": g["n_open"].sum(),
        "med_all": g["med_all"].median(), "med_open": g["med_open"].median(),
        "p90_open": g["p90_open"].median(), "n_days": g.size(),
    }).join(yv[["cls"]])
    vs = vs[vs["cls"].notna()]

    rows = []
    for key, zh, en in CLASSES:
        d = vs[vs["cls"] == key]
        d_open = d[d["n_open"] >= a.min_reports]
        d_all = d[d["n_all"] >= a.min_reports]
        if len(d_all) == 0:
            continue
        rows.append(dict(
            class_key=key, class_zh=zh, class_en=en,
            n_vessels_seen=len(d), n_vessels_all=len(d_all), n_vessels_open=len(d_open),
            frac_reports_open=d["n_open"].sum() / max(d["n_all"].sum(), 1),
            service_kn_p10=q(d_open["med_open"], .1),
            service_kn_p25=q(d_open["med_open"], .25),
            service_kn_p50=q(d_open["med_open"], .5),
            service_kn_p75=q(d_open["med_open"], .75),
            service_kn_p90=q(d_open["med_open"], .9),
            service_kn_max_p50=q(d_open["p90_open"], .5),
            allwater_kn_p50=q(d_all["med_all"], .5),
            service_kmh_p50=q(d_open["med_open"], .5) * KN2KMH,
        ))
    res = pd.DataFrame(rows)
    res["open_minus_all_kn"] = res["service_kn_p50"] - res["allwater_kn_p50"]
    res = res.round(3)
    res.to_csv(os.path.join(a.out, "service_speed_by_class.csv"),
               index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 各級服務航速 (節) ===")
    print(res[["class_zh", "n_vessels_all", "n_vessels_open", "frac_reports_open",
               "service_kn_p25", "service_kn_p50", "service_kn_p75", "service_kn_p90",
               "allwater_kn_p50", "open_minus_all_kn"]].to_string(index=False))

    # 圖
    r2 = res[res["n_vessels_open"] >= 10].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(r2) + 2))
    y = np.arange(len(r2))
    ax.barh(y, r2["service_kn_p75"] - r2["service_kn_p25"], left=r2["service_kn_p25"],
            color="#4c78a8", alpha=.6, label="開闊水域 P25–P75" if a.lang == "zh" else "open water P25-P75")
    ax.scatter(r2["service_kn_p50"], y, color="k", zorder=3,
               label="開闊水域 中位數" if a.lang == "zh" else "open water median")
    ax.scatter(r2["allwater_kn_p50"], y, marker="x", color="#d62728", zorder=3,
               label="不限制離岸 中位數" if a.lang == "zh" else "all-water median")
    ax.set_yticks(y)
    ax.set_yticklabels(r2["class_zh" if a.lang == "zh" else "class_en"])
    ax.set_xlabel("SOG (節)" if a.lang == "zh" else "SOG (kn)")
    ax.set_title(f"服務航速 (離岸 ≥ {a.min_dist_km:.1f} km, SOG ≥ {a.min_sog:g} 節)"
                 if a.lang == "zh" else f"Service speed (offshore ≥ {a.min_dist_km:.1f} km)")
    ax.grid(axis="x", alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "service_speed.png"), dpi=180)

    sp = os.path.join(a.out, "ship_profiles.csv")
    if os.path.exists(sp):
        prof = pd.read_csv(sp)
        keep = ["class_key", "service_kn_p10", "service_kn_p25", "service_kn_p50",
                "service_kn_p75", "service_kn_p90", "service_kmh_p50", "allwater_kn_p50"]
        prof = prof.merge(res[keep], on="class_key", how="left")
        prof.to_csv(os.path.join(a.out, "ship_profiles_v2.csv"),
                    index=False, encoding="utf-8-sig")
        print("\n已寫入 ship_profiles_v2.csv")
    print("輸出位置:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
