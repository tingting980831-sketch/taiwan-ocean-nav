"""
voyages.py -- 從 AIS 軌跡切出「港口到港口」的真實航次

流程:
  第1階段 (每日快取): 讀原始 AIS -> 只留可分級船舶/研究範圍/每 N 分鐘一筆 -> <out>/tracks/track_YYYYMMDD_b10.pkl
  第2階段: 讀日期範圍內的快取 -> 偵測靠港 -> 切航次 -> 統計

靠港判定: 在港口半徑內 且 SOG <= --stop-sog, 持續 >= --min-dwell-h 小時
航次: 同一艘船連續兩次「不同港口」靠泊之間的軌跡
品質旗標 complete: 最大報告間隔 <= --max-gap-h、報告數足夠、繞行比 1~--max-detour(預設4)

需與 ais_profile.py、monthly_corr.py、analyze_speed.py 放同一資料夾,
且 monthly_corr.py 已跑完 (需要 <out>/daily/ 快取來決定船舶分級)。

用法 (試點: 2023 年 3 月):
    python voyages.py --ais-dir C:\\NODASS\\AIS --out C:\\NODASS\\out_2023 --start 20230301 --end 20230331
    python voyages.py ... --pair "Taichung,Suao;Taichung,Kaohsiung"   # 另外列表並畫軌跡 (多對用分號)
    (--pair 只影響「額外印出/畫圖」哪幾對, 不影響分析範圍: 所有港口之間的航次都會處理)

整年 (逐月分段, 前後各多讀 3 天以保留跨月航次):
    python voyages.py --ais-dir C:\\NODASS\\AIS --out C:\\NODASS\\out_2023 --start 20230101 --end 20231231

輸出 (<out>/voyages_<start>_<end>/):
    stops_map.png        低速停泊點密度圖 + 港口位置 (先檢查港口座標對不對!)
    ports_zoom.png       每個港口放大圖 (圈是否套住亮點, 圈內停泊點數)
    voyages.csv          所有航次 (含品質旗標)
    voyage_tracks.csv    complete 航次的軌跡點
    od_summary.csv       起訖港 x 船級 的航次數與統計 (嚴格間隔)
    od_summary_relaxed.csv  同上, 寬鬆間隔 (<=6h)
    voyages_by_month.csv 逐月航次數 (檢查哪些月份資料偏少)
    pair_tracks_<港A>_<港B>.png   (有給 --pair 才有)
"""
import argparse
import calendar
import os
import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_speed import build_vessel_classes, BBOX        # noqa: E402
from monthly_corr import CLASSES, setup_font                # noqa: E402

KM_PER_DEG = 111.2
KN = 1.852

# 港口: 名稱 -> (經度, 緯度, 判定半徑 km)   *** 近似值, 請用 stops_map.png 檢查並自行修正 ***
PORTS = {
    "Keelung":   (121.745, 25.155, 4.5),
    "Taipei":    (121.395, 25.160, 4.0),
    "Taichung":  (120.510, 24.290, 6.0),
    "Mailiao":   (120.180, 23.790, 4.0),
    "Anping":    (120.150, 23.000, 4.0),
    "Kaohsiung": (120.280, 22.580, 8.0),
    "Hualien":   (121.620, 23.980, 3.0),
    "Suao":      (121.870, 24.600, 3.0),
}

RAW_COLS = ["MMSI", "Longitude", "Latitude", "SOG", "COG",
            "Navigational_Status", "MAX_Draught", "Record_Time"]


# ------------------------------------------------------------------
# 第 1 階段: 每日軌跡快取
# ------------------------------------------------------------------
def build_day_track(path, cache, valid_mmsi, bin_min):
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    if not os.path.exists(path):
        print(f"  [缺檔] {path}")
        return None
    df = pd.read_csv(path, usecols=lambda c: c in RAW_COLS,
                     dtype={"MMSI": str}, low_memory=False)
    df = df[df["MMSI"].isin(valid_mmsi)]
    inside = df["Longitude"].between(BBOX["lon0"], BBOX["lon1"]) & \
             df["Latitude"].between(BBOX["lat0"], BBOX["lat1"])
    df = df[inside].copy()

    # 只解析不重複的時間字串 (比整欄解析快很多)
    codes, uniques = pd.factorize(df["Record_Time"])
    parsed = pd.to_datetime(pd.Index(uniques), errors="coerce")
    tt = np.where(codes >= 0, parsed.values[np.clip(codes, 0, None)],
                  np.datetime64("NaT"))
    df["t"] = pd.to_datetime(tt)
    df = df[df["t"].notna()]

    n_unique_t = df["t"].nunique()
    df = df.sort_values(["MMSI", "t"])
    dt = df.groupby("MMSI")["t"].diff().dt.total_seconds() / 60.0
    print(f"  {os.path.basename(path)}: {len(df):,} 列, 不同時間戳 {n_unique_t:,}, "
          f"同船相鄰紀錄間隔 中位數 {dt.median():.1f} 分", flush=True)

    df["tb"] = df["t"].dt.floor(f"{bin_min}min")
    df = df.drop_duplicates(["MMSI", "tb"], keep="first")

    out = pd.DataFrame({
        "MMSI": df["MMSI"].values, "t": df["t"].values,
        "lon": df["Longitude"].values.astype("float32"),
        "lat": df["Latitude"].values.astype("float32"),
        "sog": df["SOG"].where(df["SOG"] < 102).values.astype("float32"),
        "cog": df["COG"].where(df["COG"] < 360).values.astype("float32"),
        "draft": df["MAX_Draught"].where((df["MAX_Draught"] > 0) &
                                         (df["MAX_Draught"] <= 25.5)).values.astype("float32"),
    })
    out.to_pickle(cache)
    return out


def daterange(start, end):
    d0 = datetime.strptime(start, "%Y%m%d")
    d1 = datetime.strptime(end, "%Y%m%d")
    while d0 <= d1:
        yield d0.strftime("%Y%m%d")
        d0 += timedelta(days=1)


# ------------------------------------------------------------------
# 工具
# ------------------------------------------------------------------
def dist_km(lon1, lat1, lon2, lat2):
    lat0 = np.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * KM_PER_DEG * np.cos(lat0)
    dy = (lat2 - lat1) * KM_PER_DEG
    return np.hypot(dx, dy)


def path_length_km(lon, lat):
    if len(lon) < 2:
        return 0.0
    return float(dist_km(lon[:-1], lat[:-1], lon[1:], lat[1:]).sum())


# ------------------------------------------------------------------
# 第 2 階段: 靠港偵測與切航次
# ------------------------------------------------------------------
def detect_calls(df, args):
    """回傳 calls DataFrame: MMSI, port, start, end, n, draft。"""
    names = list(PORTS)
    port_idx = np.full(len(df), -1, dtype=np.int16)
    lon, lat, sog = df["lon"].values, df["lat"].values, df["sog"].values
    for i, nme in enumerate(names):
        plon, plat, rad = PORTS[nme]
        d = dist_km(lon, lat, plon, plat)
        port_idx[(d <= rad) & (sog <= args.stop_sog)] = i
    df["port"] = port_idx

    mm = df["MMSI"].values
    t = df["t"].values.astype("datetime64[s]").astype("int64")
    new_run = np.ones(len(df), dtype=bool)
    new_run[1:] = (mm[1:] != mm[:-1]) | (port_idx[1:] != port_idx[:-1]) | \
                  ((t[1:] - t[:-1]) > args.max_gap_h * 3600)
    df["run"] = np.cumsum(new_run)

    runs = df[df["port"] >= 0].groupby("run").agg(
        MMSI=("MMSI", "first"), port=("port", "first"),
        start=("t", "min"), end=("t", "max"), n=("t", "size"),
        draft=("draft", "median"))
    runs["dwell_h"] = (runs["end"] - runs["start"]).dt.total_seconds() / 3600
    runs = runs[(runs["dwell_h"] >= args.min_dwell_h)].copy()
    runs["port"] = runs["port"].map(lambda i: names[i])
    runs = runs.sort_values(["MMSI", "start"]).reset_index(drop=True)

    # 同一艘船連續在同一港口的段落(中間短暫離開/移泊)合併
    merged = []
    for mmsi, g in runs.groupby("MMSI", sort=False):
        cur = None
        for r in g.itertuples(index=False):
            if cur is not None and r.port == cur["port"] and \
               (r.start - cur["end"]).total_seconds() / 3600 <= args.merge_h:
                cur["end"] = r.end
                cur["n"] += r.n
                if pd.isna(cur["draft"]):
                    cur["draft"] = r.draft
            else:
                if cur is not None:
                    merged.append(cur)
                cur = dict(MMSI=mmsi, port=r.port, start=r.start, end=r.end,
                           n=r.n, draft=r.draft)
        if cur is not None:
            merged.append(cur)
    return pd.DataFrame(merged)


def cut_voyages(df, calls, cls_map, args):
    mm_idx = df.groupby("MMSI").indices          # 每艘船的列位置 (df 已依 MMSI,t 排序)
    t_all = df["t"].values
    lon_all, lat_all = df["lon"].values, df["lat"].values
    sog_all, draft_all = df["sog"].values, df["draft"].values

    voy, tracks = [], []
    vid = 0
    for mmsi, g in calls.groupby("MMSI", sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("start").reset_index(drop=True)
        pos = mm_idx[mmsi]
        tv = t_all[pos]
        for i in range(len(g) - 1):
            a, b = g.loc[i], g.loc[i + 1]
            if a["port"] == b["port"]:
                continue
            lo = np.searchsorted(tv, np.datetime64(a["end"]), side="left")
            hi = np.searchsorted(tv, np.datetime64(b["start"]), side="right")
            seg = pos[lo:hi]
            if len(seg) < 2:
                continue
            ts = t_all[seg]
            lon, lat = lon_all[seg], lat_all[seg]
            sg, dr = sog_all[seg], draft_all[seg]
            dur_h = (b["start"] - a["end"]).total_seconds() / 3600
            gaps = np.diff(ts).astype("timedelta64[s]").astype(float) / 3600
            gap_max = float(gaps.max()) if len(gaps) else np.nan
            path = path_length_km(lon, lat)
            p0, p1 = PORTS[a["port"]], PORTS[b["port"]]
            gc = float(dist_km(p0[0], p0[1], p1[0], p1[1]))
            detour = path / gc if gc > 0 else np.nan
            sea = sg[(sg >= 3)]
            # 位置跳躍檢查: 相鄰兩筆(間隔 5 分~1 小時)換算的速度不能不合理
            tsec = ts.astype("datetime64[s]").astype("int64")
            dts = np.diff(tsec) / 3600.0
            dd_ = dist_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
            mk = (dts >= 5 / 60) & (dts <= 1.0)
            max_seg = float((dd_[mk] / dts[mk] / KN).max()) if mk.any() else np.nan
            limit = 70.0 if cls_map.get(str(mmsi)) == "pax_fast" else args.max_speed_kn
            speed_ok = bool(np.isnan(max_seg) or max_seg <= limit)
            base_ok = len(seg) >= 6 and dur_h >= 0.5 and (1.0 <= detour <= args.max_detour) and speed_ok
            complete = base_ok and (gap_max <= args.max_gap_h)
            complete_relaxed = base_ok and (gap_max <= args.max_gap_relaxed_h)
            vid += 1
            voy.append(dict(
                voyage_id=vid, MMSI=str(mmsi), cls=cls_map.get(str(mmsi)),
                origin=a["port"], dest=b["port"],
                depart=a["end"], arrive=b["start"], dur_h=round(dur_h, 2),
                n_reports=len(seg), gap_max_h=round(gap_max, 2),
                path_km=round(path, 1), gc_km=round(gc, 1), detour=round(detour, 3),
                mean_sog_kn=round(path / dur_h / KN, 2) if dur_h > 0 else np.nan,
                sea_sog_median_kn=round(float(np.median(sea)), 2) if len(sea) else np.nan,
                draft_at_origin=a["draft"], draft_at_dest=b["draft"],
                draft_voyage_median=float(np.nanmedian(dr)) if np.isfinite(dr).any() else np.nan,
                max_seg_kn=round(max_seg, 1) if not np.isnan(max_seg) else np.nan,
                speed_ok=speed_ok,
                complete=bool(complete), complete_relaxed=bool(complete_relaxed)))
            if complete_relaxed:
                tracks.append(pd.DataFrame({
                    "voyage_id": vid, "t": ts, "lon": lon, "lat": lat, "sog": sg}))
    voy = pd.DataFrame(voy)
    trk = pd.concat(tracks, ignore_index=True) if tracks else pd.DataFrame()
    return voy, trk


# ------------------------------------------------------------------
# 圖
# ------------------------------------------------------------------
def base_axes(ax):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND, facecolor="#cccccc")
        ax.add_feature(cfeature.COASTLINE, linewidth=.6)
        return dict(transform=ccrs.PlateCarree())
    except Exception:
        return {}


def plot_stops(df, out_png, zh):
    try:
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(8, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([BBOX["lon0"], BBOX["lon1"], BBOX["lat0"], BBOX["lat1"]])
        kw = base_axes(ax)
    except Exception:
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.set_xlim(BBOX["lon0"], BBOX["lon1"]); ax.set_ylim(BBOX["lat0"], BBOX["lat1"])
        kw = {}
    st = df[df["sog"] <= 0.5]
    ax.hist2d(st["lon"], st["lat"], bins=[300, 250],
              range=[[BBOX["lon0"], BBOX["lon1"]], [BBOX["lat0"], BBOX["lat1"]]],
              cmin=1, norm=matplotlib.colors.LogNorm(), cmap="inferno", **kw)
    for nme, (lo, la, r) in PORTS.items():
        th = np.linspace(0, 2 * np.pi, 60)
        ax.plot(lo + r / (KM_PER_DEG * np.cos(np.radians(la))) * np.cos(th),
                la + r / KM_PER_DEG * np.sin(th), color="cyan", lw=1, **kw)
        ax.text(lo + .05, la + .05, nme, color="cyan", fontsize=8, **kw)
    ax.set_title("停泊點密度 (SOG 0.5 節以下) 與港口判定圈 — 請確認圈有對準亮點" if zh
                 else "Stationary-report density and port circles")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_port_zoom(df, out_png, zh):
    """每個港口放大圖: 停泊點密度 + 判定圈, 用來檢查座標與半徑。"""
    names = list(PORTS)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), squeeze=False)
    st = df[df["sog"] <= 0.5]
    for ax in axs.ravel():
        ax.axis("off")
    for ax, nme in zip(axs.ravel(), names):
        lo, la, r = PORTS[nme]
        half = max(0.25, 2.5 * r / KM_PER_DEG)
        ax.axis("on")
        w = st[(st["lon"].between(lo - half, lo + half)) & (st["lat"].between(la - half, la + half))]
        n_in = int((dist_km(w["lon"].values, w["lat"].values, lo, la) <= r).sum()) if len(w) else 0
        if len(w):
            ax.hist2d(w["lon"], w["lat"], bins=70,
                      range=[[lo - half, lo + half], [la - half, la + half]],
                      cmin=1, cmap="inferno", norm=matplotlib.colors.LogNorm())
        th = np.linspace(0, 2 * np.pi, 80)
        ax.plot(lo + r / (KM_PER_DEG * np.cos(np.radians(la))) * np.cos(th),
                la + r / KM_PER_DEG * np.sin(th), color="cyan", lw=1.5)
        ax.scatter([lo], [la], color="cyan", s=15, zorder=5)
        ax.set_xlim(lo - half, lo + half); ax.set_ylim(la - half, la + half)
        ax.set_title(f"{nme}  r={r:g} km  (圈內停泊點 {n_in:,}/{len(w):,})" if zh
                     else f"{nme}  r={r:g} km  (in-circle {n_in:,}/{len(w):,})", fontsize=9)
        ax.set_facecolor("#dddddd")
        ax.set_aspect(1 / np.cos(np.radians(la)))
        ax.tick_params(labelsize=7)
    fig.suptitle("港口放大圖: 青色圓點=我設定的港口中心; 亮點應落在圈內" if zh
                 else "Port zoom: cyan = configured centre; bright spots should fall inside the circle")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_pair(voy, trk, pair, out_png, zh):
    a, b = pair
    sel = voy[voy["complete"] & (((voy["origin"] == a) & (voy["dest"] == b)) |
                                  ((voy["origin"] == b) & (voy["dest"] == a)))]
    if sel.empty:
        print(f"  {a}↔{b}: 沒有 complete 航次, 不畫圖")
        return
    try:
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(8, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([BBOX["lon0"], BBOX["lon1"], BBOX["lat0"], BBOX["lat1"]])
        kw = base_axes(ax)
    except Exception:
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.set_xlim(BBOX["lon0"], BBOX["lon1"]); ax.set_ylim(BBOX["lat0"], BBOX["lat1"])
        kw = {}
    for vid, d in trk[trk["voyage_id"].isin(sel["voyage_id"])].groupby("voyage_id"):
        fwd = sel.loc[sel["voyage_id"] == vid, "origin"].iloc[0] == a
        ax.plot(d["lon"], d["lat"], color="#d62728" if fwd else "#1f77b4", alpha=.5, lw=1, **kw)
    for nme in (a, b):
        lo, la, _ = PORTS[nme]
        ax.scatter(lo, la, color="k", s=40, zorder=5, **kw)
        ax.text(lo + .05, la + .05, nme, fontsize=9, **kw)
    ax.set_title(f"{a}→{b} (紅) / {b}→{a} (藍), n={len(sel)}" if zh
                 else f"{a}→{b} (red) / {b}→{a} (blue), n={len(sel)}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ais-dir", required=True)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--start", required=True, help="YYYYMMDD")
    ap_.add_argument("--end", required=True, help="YYYYMMDD")
    ap_.add_argument("--year", type=int, default=None, help="分級快取年份 (預設取 --start 的年份)")
    ap_.add_argument("--bin-min", type=int, default=10)
    ap_.add_argument("--stop-sog", type=float, default=1.0)
    ap_.add_argument("--min-dwell-h", type=float, default=1.0)
    ap_.add_argument("--merge-h", type=float, default=6.0)
    ap_.add_argument("--max-gap-h", type=float, default=3.0, help="complete: 最大報告間隔(小時, 嚴格)")
    ap_.add_argument("--max-gap-relaxed-h", type=float, default=6.0, help="complete_relaxed: 寬鬆間隔上限")
    ap_.add_argument("--max-speed-kn", type=float, default=40.0,
                     help="航次內相鄰紀錄換算速度上限(節); 超過視為位置跳躍/MMSI 重複 (高速客船放寬到 70)")
    ap_.add_argument("--chunk-days", type=int, default=31, help="每段處理幾天 (控制記憶體)")
    ap_.add_argument("--pad-days", type=int, default=3, help="每段前後多讀幾天, 避免跨段航次被切斷")
    ap_.add_argument("--max-detour", type=float, default=4.0,
                     help="航程/兩港直線距離 的上限; 跨島航線的直線穿過陸地, 比值本來就偏大")
    ap_.add_argument("--pair", default=None, help="例如 Taichung,Suao")
    ap_.add_argument("--lang", default="zh", choices=["zh", "en"])
    a = ap_.parse_args()
    zh = a.lang == "zh"
    setup_font(a.lang)
    year = a.year or int(a.start[:4])

    print("建立船舶分級...")
    yv = build_vessel_classes(os.path.join(a.out, "daily"), year)
    valid = set(yv.index)
    cls_map = yv["cls"].to_dict()
    print(f"可分級船舶: {len(valid):,}")

    track_dir = os.path.join(a.out, "tracks")
    os.makedirs(track_dir, exist_ok=True)
    odir = os.path.join(a.out, f"voyages_{a.start}_{a.end}")
    os.makedirs(odir, exist_ok=True)

    d_start = datetime.strptime(a.start, "%Y%m%d")
    d_end = datetime.strptime(a.end, "%Y%m%d")
    one = timedelta(days=1)
    pad = timedelta(days=a.pad_days)

    def load_range(d0, d1):
        out_ = []
        d = d0
        while d <= d1:
            ymd = d.strftime("%Y%m%d")
            r = build_day_track(os.path.join(a.ais_dir, f"{ymd}.csv"),
                                os.path.join(track_dir, f"track_{ymd}_b{a.bin_min}.pkl"),
                                valid, a.bin_min)
            if r is not None and len(r):
                out_.append(r)
            d += one
        return out_

    voy_all, trk_all, stops_all, call_counts = [], [], [], {}
    id_offset = 0
    c0 = d_start
    while c0 <= d_end:
        c1 = min(c0 + timedelta(days=a.chunk_days - 1), d_end)
        lo, hi = max(c0 - pad, d_start), min(c1 + pad, d_end)
        print(f"\n=== 分段 {c0:%Y%m%d} ~ {c1:%Y%m%d} (讀取 {lo:%Y%m%d} ~ {hi:%Y%m%d}) ===", flush=True)
        frames = load_range(lo, hi)
        if not frames:
            c0 = c1 + one
            continue
        df = pd.concat(frames, ignore_index=True)
        del frames
        df["MMSI"] = df["MMSI"].astype("int64")          # 整數比字串省很多記憶體
        df = df.sort_values(["MMSI", "t"]).drop_duplicates(["MMSI", "t"]).reset_index(drop=True)

        t_arr = df["t"].values
        in_core = (t_arr >= np.datetime64(c0)) & (t_arr < np.datetime64(c1 + one))
        stops_all.append(df.loc[in_core & (df["sog"].values <= 0.5), ["lon", "lat", "sog"]])

        calls = detect_calls(df, a)
        n_v = 0
        if len(calls):
            core_c = calls[(calls["start"] >= pd.Timestamp(c0)) &
                           (calls["start"] < pd.Timestamp(c1 + one))]
            for pn, n in core_c["port"].value_counts().items():
                call_counts[pn] = call_counts.get(pn, 0) + int(n)
            voy, trk = cut_voyages(df, calls, cls_map, a)
            if len(voy):
                keep = (voy["depart"] >= pd.Timestamp(c0)) & (voy["depart"] < pd.Timestamp(c1 + one))
                voy = voy[keep].copy()
                if len(trk):
                    trk = trk[trk["voyage_id"].isin(voy["voyage_id"])].copy()
                if len(voy):
                    mx = int(voy["voyage_id"].max())
                    voy["voyage_id"] += id_offset
                    if len(trk):
                        trk["voyage_id"] += id_offset
                        trk_all.append(trk)
                    voy_all.append(voy)
                    id_offset += mx
                    n_v = len(voy)
        print(f"  軌跡 {len(df):,} 筆, {df['MMSI'].nunique():,} 艘; 本段航次 {n_v:,}", flush=True)
        del df
        c0 = c1 + one

    if not voy_all:
        sys.exit("沒有切出任何航次 (港口座標或停泊判定可能不對)")
    voy = pd.concat(voy_all, ignore_index=True)
    trk = pd.concat(trk_all, ignore_index=True) if trk_all else pd.DataFrame()

    stops = pd.concat(stops_all, ignore_index=True)
    plot_stops(stops, os.path.join(odir, "stops_map.png"), zh)
    plot_port_zoom(stops, os.path.join(odir, "ports_zoom.png"), zh)
    print("\n已輸出 stops_map.png 與 ports_zoom.png")
    print("靠港次數 (全期間):")
    print(pd.Series(call_counts).sort_values(ascending=False).to_string())

    voy.to_csv(os.path.join(odir, "voyages.csv"), index=False, encoding="utf-8-sig")
    if len(trk):
        trk.to_csv(os.path.join(odir, "voyage_tracks.csv"), index=False)

    zh_of = {k: z for k, z, _ in CLASSES}
    voy["class_zh"] = voy["cls"].map(zh_of)
    voy["month"] = voy["depart"].dt.strftime("%Y-%m")
    c = voy[voy["complete"]]
    cr = voy[voy["complete_relaxed"]]
    print(f"\n航次總數 {len(voy):,}; complete(間隔<={a.max_gap_h:g}h) {len(c):,} ({len(c)/len(voy):.0%}); "
          f"complete_relaxed(<={a.max_gap_relaxed_h:g}h) {len(cr):,} ({len(cr)/len(voy):.0%})")
    print("未通過原因(嚴格): 報告間隔過大 %d, 報告數/時間不足 %d, 繞行比不合理 %d, 速度異常(位置跳躍) %d" % (
        (voy["gap_max_h"] > a.max_gap_h).sum(),
        ((voy["n_reports"] < 6) | (voy["dur_h"] < 0.5)).sum(),
        (~voy["detour"].between(1.0, a.max_detour)).sum(),
        (~voy["speed_ok"]).sum()))
    long_gap = voy[voy["gap_max_h"] > a.max_gap_relaxed_h]
    print("間隔超過 %g 小時的航次 %d 個; 其中超過 24 小時 %d 個 (可能是離開研究區再回來)" % (
        a.max_gap_relaxed_h, len(long_gap), (long_gap["gap_max_h"] > 24).sum()))

    by_m = voy.groupby("month").agg(total=("voyage_id", "size"),
                                    complete=("complete", "sum"),
                                    complete_relaxed=("complete_relaxed", "sum"),
                                    vessels=("MMSI", "nunique"))
    by_m.to_csv(os.path.join(odir, "voyages_by_month.csv"), encoding="utf-8-sig")
    print("\n=== 逐月航次數 ===")
    print(by_m.to_string())

    def od_table(d):
        return (d.groupby(["origin", "dest", "class_zh"])
                 .agg(n=("voyage_id", "size"), vessels=("MMSI", "nunique"),
                      path_km=("path_km", "median"), gc_km=("gc_km", "median"),
                      detour=("detour", "median"), dur_h=("dur_h", "median"),
                      mean_sog_kn=("mean_sog_kn", "median"),
                      sea_sog_kn=("sea_sog_median_kn", "median"),
                      draft_at_origin=("draft_at_origin", "median"))
                 .round(2).reset_index().sort_values("n", ascending=False))

    od, od_r = od_table(c), od_table(cr)
    od.to_csv(os.path.join(odir, "od_summary.csv"), index=False, encoding="utf-8-sig")
    od_r.to_csv(os.path.join(odir, "od_summary_relaxed.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 起訖港 x 船級 (complete 航次, 依數量排序, 前 25) ===")
    print(od.head(25).to_string(index=False))

    print("\n=== 各船級 complete 航次數 (嚴格 / 寬鬆) ===")
    cc = pd.DataFrame({"strict": c["class_zh"].value_counts(),
                       "relaxed": cr["class_zh"].value_counts()}).fillna(0).astype(int)
    print(cc.sort_values("relaxed", ascending=False).to_string())

    if a.pair:
        for pr in a.pair.split(";"):
            pa = tuple(x.strip() for x in pr.split(","))
            if len(pa) != 2:
                print(f"\n[略過] --pair 格式錯誤: {pr!r} (應為 港A,港B; 多對用分號隔開)")
                continue
            sub = od[((od["origin"] == pa[0]) & (od["dest"] == pa[1])) |
                     ((od["origin"] == pa[1]) & (od["dest"] == pa[0]))]
            print(f"\n=== {pa[0]} ↔ {pa[1]} (complete, 嚴格) ===")
            print(sub.to_string(index=False) if len(sub) else "沒有 complete 航次")
            subr = od_r[((od_r["origin"] == pa[0]) & (od_r["dest"] == pa[1])) |
                        ((od_r["origin"] == pa[1]) & (od_r["dest"] == pa[0]))]
            print(f"--- 寬鬆版 航次合計 {int(subr['n'].sum()) if len(subr) else 0}")
            plot_pair(cr.assign(complete=True), trk, pa,
                      os.path.join(odir, f"pair_tracks_{pa[0]}_{pa[1]}.png"), zh)
    print("\n輸出位置:", os.path.abspath(odir))


if __name__ == "__main__":
    main()
