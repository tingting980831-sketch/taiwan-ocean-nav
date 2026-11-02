"""
baseline.py -- 基準路徑: 「不考慮海象的最短海上路徑」 vs 真實航次

為什麼要做:
  論文的省油率是「和誰比」沒有說清楚。公平的起點是: 在同一片海域、不看風浪流,
  船最短能走多遠? 真實航次比這條路徑多繞了多少, 就是「純路徑優化」理論上最多能省的距離。

做法:
  1. 把陸地變成 0.01 度格點, 離岸距離 >= buffer 公里的海面才可通行
  2. 每個港口選一個「進出口」格點 (離港口中心最近的可通行格點)
  3. 用 16 方向的最短路徑 (Dijkstra) 算港口之間的海上最短距離
  4. 每個真實航次: 基準 = 起點到進出口 + 進出口之間最短路徑 + 進出口到終點
  5. 比較 實際航程 / 基準; 同時用多個 buffer 做敏感度檢查

重要: 起訖點用「每個航次軌跡自己的第一點與最後一點」, 不是港口中心。

需與 voyages.py、analyze_speed.py、ais_profile.py、monthly_corr.py 放同一資料夾。

用法 (Windows):
    python baseline.py --out C:\\NODASS\\out_2023
    python baseline.py --out C:\\NODASS\\out_2023 --pairs "Taichung,Kaohsiung;Keelung,Taichung"

輸出 (<voyages資料夾>\\baseline\\):
    voyage_baseline.csv     每個航次: 實際航程、各 buffer 的基準、比值、品質旗標 (use=False 不進摘要)
    baseline_summary.csv    每條航廊(起點,終點,船級): 比值、以船為單位的中位數與 95% 信賴區間
    baseline_<A>_<B>.png    基準路徑 (紅) 與真實航跡 (灰)
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_speed import load_land, build_dist_grid, BBOX, GRID_RES   # noqa: E402
from voyages import PORTS, dist_km                                    # noqa: E402
from monthly_corr import CLASSES, setup_font                          # noqa: E402

KM_PER_DEG = 111.2


# ------------------------------------------------------------------
# 圖與最短路徑
# ------------------------------------------------------------------
def build_graph(nav, lats):
    """16 方向格點圖; 邊權重=公里。nav: (R,C) bool, True=可通行。"""
    R, C = nav.shape
    idx = np.arange(R * C).reshape(R, C)
    dy_km = GRID_RES * KM_PER_DEG
    # (dr, dc): 對角線與「騎士步」; 需要兩個中間格都可通行 (避免穿過陸地角落)
    inter = {
        (0, 1): [], (1, 0): [],
        (1, 1): [(0, 1), (1, 0)], (1, -1): [(0, -1), (1, 0)],
        (1, 2): [(0, 1), (1, 1)], (2, 1): [(1, 0), (1, 1)],
        (1, -2): [(0, -1), (1, -1)], (2, -1): [(1, 0), (1, -1)],
    }
    S, D, W = [], [], []
    for (dr, dc), mids in inter.items():
        cs = 0 if dc >= 0 else -dc
        nc = C - abs(dc)
        nr = R - dr
        ok = nav[0:nr, cs:cs + nc] & nav[dr:R, cs + dc:cs + dc + nc]
        for rr, cc in mids:
            ok &= nav[rr:rr + nr, cs + cc:cs + cc + nc]
        src = idx[0:nr, cs:cs + nc]
        dst = idx[dr:R, cs + dc:cs + dc + nc]
        latm = lats[0:nr] + dr * GRID_RES / 2
        dx_km = dc * GRID_RES * KM_PER_DEG * np.cos(np.radians(latm))[:, None]
        w = np.hypot(dr * dy_km, dx_km)
        w = np.broadcast_to(w, src.shape)
        S.append(src[ok]); D.append(dst[ok]); W.append(w[ok])
    S, D, W = np.concatenate(S), np.concatenate(D), np.concatenate(W)
    N = R * C
    # 無向: 補反向邊
    G = csr_matrix((np.concatenate([W, W]), (np.concatenate([S, D]), np.concatenate([D, S]))),
                   shape=(N, N))
    return G


def largest_component(nav):
    lab, n = label(nav, structure=np.ones((3, 3)))
    if n == 0:
        return nav
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def pick_gate(comp, lons, lats, plon, plat):
    rr, cc = np.nonzero(comp)
    d = dist_km(lons[cc], lats[rr], plon, plat)
    k = int(d.argmin())
    return int(rr[k]), int(cc[k]), float(d[k])


def trace_path(pred_row, src_idx, dst_idx):
    path = [dst_idx]
    cur = dst_idx
    guard = 0
    while cur != src_idx and guard < 2_000_000:
        cur = pred_row[cur]
        if cur < 0:
            return None
        path.append(cur)
        guard += 1
    return path[::-1]


def track_qc(trk, dist, lons, lats, trim_km=12.0, step_km=2.0):
    """每個航次: 最西經度, 以及軌跡「穿過陸地」的取樣點數 (忽略頭尾各 trim_km 的港區)。
    相鄰兩點之間會內插, 這樣位置「跳躍」穿過陸地也抓得到。"""
    t = trk.sort_values(["voyage_id", "t"])
    vid = t["voyage_id"].values
    lon, lat = t["lon"].values.astype(float), t["lat"].values.astype(float)
    same = vid[1:] == vid[:-1]
    seg = np.where(same, dist_km(lon[:-1], lat[:-1], lon[1:], lat[1:]), 0.0)
    seg_full = np.concatenate([[0.0], seg])
    cum = pd.Series(seg_full).groupby(vid).cumsum().values            # 航次內累積距離
    total = pd.Series(cum).groupby(vid).transform("max").values        # 該航次總長
    n_i = np.where(same, np.maximum(1, np.ceil(seg / step_km)).astype(int), 0)
    tot = int(n_i.sum())
    codes, uniq = pd.factorize(vid)
    out = pd.DataFrame({"min_lon": pd.Series(lon).groupby(vid).min()})
    out["land_hits"] = 0
    if tot > 0:
        seg_id = np.repeat(np.arange(len(n_i)), n_i)
        k = np.arange(tot) - np.repeat(np.cumsum(n_i) - n_i, n_i)
        frac = (k + 1) / n_i[seg_id]
        slon = lon[seg_id] + frac * (lon[seg_id + 1] - lon[seg_id])
        slat = lat[seg_id] + frac * (lat[seg_id + 1] - lat[seg_id])
        scum = cum[seg_id] + frac * seg[seg_id]
        keep = (scum >= trim_km) & (scum <= total[seg_id] - trim_km)
        ix = np.clip(np.rint((slon - lons[0]) / GRID_RES).astype(int), 0, dist.shape[1] - 1)
        iy = np.clip(np.rint((slat - lats[0]) / GRID_RES).astype(int), 0, dist.shape[0] - 1)
        hit = keep & (dist[iy, ix] == 0)
        cnt = np.bincount(codes[seg_id][hit], minlength=len(uniq))
        out["land_hits"] = pd.Series(cnt, index=uniq).reindex(out.index).fillna(0).astype(int).values
    return out


# ------------------------------------------------------------------
def find_voy_dir(out, given):
    if given:
        return given
    cands = sorted(glob.glob(os.path.join(out, "voyages_*", "voyages.csv")),
                   key=os.path.getmtime, reverse=True)
    if not cands:
        sys.exit("找不到 voyages_*\\voyages.csv, 請先跑 voyages.py, 或用 --voy-dir 指定")
    return os.path.dirname(cands[0])


def boot_median_ci(values, n_boot=1000, seed=0):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--voy-dir", default=None)
    ap.add_argument("--buffers", default="2,5,10", help="離岸最小距離(公里), 逗號分隔")
    ap.add_argument("--main-buffer", type=float, default=5.0, help="主要結果與地圖使用的 buffer")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-vessels", type=int, default=10)
    ap.add_argument("--max-speed-kn", type=float, default=40.0)
    ap.add_argument("--min-lon", type=float, default=119.3,
                    help="軌跡最西經度低於此值, 視為繞去大陸沿岸 (非直達航次)")
    ap.add_argument("--max-ratio", type=float, default=1.6,
                    help="實際/基準 超過此值視為非直達 (中途繞去別處)")
    ap.add_argument("--land-hits-max", type=int, default=3,
                    help="軌跡穿過陸地的取樣點數達此值, 視為位置跳躍")
    ap.add_argument("--pairs", default="Taichung,Kaohsiung;Keelung,Taichung")
    ap.add_argument("--land-shp", default=None)
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    a = ap.parse_args()
    zh = a.lang == "zh"
    setup_font(a.lang)

    vdir = find_voy_dir(a.out, a.voy_dir)
    odir = os.path.join(vdir, "baseline")
    os.makedirs(odir, exist_ok=True)
    print("使用航次資料夾:", vdir)

    voy = pd.read_csv(os.path.join(vdir, "voyages.csv"), dtype={"MMSI": str})
    trk = pd.read_csv(os.path.join(vdir, "voyage_tracks.csv"),
                      usecols=["voyage_id", "t", "lon", "lat"], parse_dates=["t"])
    sel = voy[voy["complete_relaxed"] & voy["speed_ok"] &
              (voy["mean_sog_kn"].fillna(0) <= a.max_speed_kn)].copy()
    print(f"航次 {len(voy):,}; 採用 (complete_relaxed 且速度合理) {len(sel):,}")

    trk = trk[trk["voyage_id"].isin(sel["voyage_id"])].sort_values(["voyage_id", "t"])
    g = trk.groupby("voyage_id")
    ends = pd.DataFrame({"lon0": g["lon"].first(), "lat0": g["lat"].first(),
                         "lon1": g["lon"].last(), "lat1": g["lat"].last()})
    sel = sel.join(ends, on="voyage_id").dropna(subset=["lon0"])

    print("建立陸地與離岸距離網格...")
    dist = build_dist_grid(load_land(a.land_shp))
    lons = np.arange(BBOX["lon0"], BBOX["lon1"] + GRID_RES / 2, GRID_RES)
    lats = np.arange(BBOX["lat0"], BBOX["lat1"] + GRID_RES / 2, GRID_RES)
    R, C = dist.shape
    land = dist == 0

    qc = track_qc(trk, dist, lons, lats)
    sel = sel.join(qc, on="voyage_id")

    buffers = [float(x) for x in a.buffers.split(",")]
    if a.main_buffer not in buffers:
        buffers.append(a.main_buffer)
    ports = list(PORTS)
    results, preds, gates_info = {}, {}, {}

    for B in buffers:
        nav = (dist >= B)
        comp = largest_component(nav)
        G = build_graph(comp, lats)
        gates = {}
        for p in ports:
            plon, plat, _ = PORTS[p]
            r, c, d = pick_gate(comp, lons, lats, plon, plat)
            gates[p] = (r, c, d)
        gi = [gates[p][0] * C + gates[p][1] for p in ports]
        D, P = dijkstra(G, directed=True, indices=gi, return_predecessors=True)
        results[B] = (D, gates)
        preds[B] = P
        gates_info[B] = gates
        print(f"  buffer {B:g} km: 可通行格點 {comp.sum():,}; 進出口離港口中心 " +
              ", ".join(f"{p}:{gates[p][2]:.1f}km" for p in ports), flush=True)

    # ---------- 每個航次的基準 ----------
    pidx = {p: i for i, p in enumerate(ports)}
    rows = []
    for rv in sel.itertuples(index=False):
        rec = dict(voyage_id=rv.voyage_id, MMSI=rv.MMSI, cls=rv.cls, origin=rv.origin,
                   dest=rv.dest, actual_km=rv.path_km, dur_h=rv.dur_h,
                   min_lon=rv.min_lon, land_hits=rv.land_hits)
        for B in buffers:
            D, gates = results[B]
            go, gd = gates[rv.origin], gates[rv.dest]
            core = D[pidx[rv.origin], gd[0] * C + gd[1]]
            c0 = float(dist_km(rv.lon0, rv.lat0, lons[go[1]], lats[go[0]]))
            c1 = float(dist_km(rv.lon1, rv.lat1, lons[gd[1]], lats[gd[0]]))
            base = core + c0 + c1 if np.isfinite(core) else np.nan
            rec[f"base_km_B{B:g}"] = round(base, 1)
            rec[f"ratio_B{B:g}"] = round(rv.path_km / base, 4) if np.isfinite(base) and base > 0 else np.nan
        rows.append(rec)
    vb = pd.DataFrame(rows)
    mb = f"B{a.main_buffer:g}"
    vb["non_direct"] = (vb["min_lon"] < a.min_lon) | (vb[f"ratio_{mb}"] > a.max_ratio)
    vb["crosses_land"] = vb["land_hits"] >= a.land_hits_max
    vb["use"] = ~(vb["non_direct"] | vb["crosses_land"])
    vb.to_csv(os.path.join(odir, "voyage_baseline.csv"), index=False, encoding="utf-8-sig")
    print(f"\n品質篩選 (共 {len(vb):,} 個航次): 非直達 {int(vb['non_direct'].sum()):,} "
          f"(最西經度<{a.min_lon:g} 或 比值>{a.max_ratio:g}); 穿過陸地 {int(vb['crosses_land'].sum()):,}; "
          f"保留 {int(vb['use'].sum()):,}")
    vall = vb
    vb = vb[vb["use"]].copy()

    # ---------- 航廊摘要 ----------
    zh_of = {k: z for k, z, _ in CLASSES}
    out = []
    for (o, d_, cl), grp in vb.groupby(["origin", "dest", "cls"]):
        if len(grp) < a.min_n or grp["MMSI"].nunique() < a.min_vessels:
            continue
        r_main = grp[f"ratio_{mb}"].dropna()
        per_v = grp.groupby("MMSI")[f"ratio_{mb}"].median()
        med_v, lo, hi = boot_median_ci(per_v.values)
        rec = dict(origin=o, dest=d_, class_zh=zh_of.get(cl, cl), n=len(grp),
                   vessels=grp["MMSI"].nunique(),
                   actual_km_p10=grp["actual_km"].quantile(.1), actual_km_p50=grp["actual_km"].median(),
                   actual_km_p90=grp["actual_km"].quantile(.9),
                   base_km_main=grp[f"base_km_{mb}"].median(),
                   ratio_median=r_main.median(),
                   ratio_by_vessel_median=med_v, ratio_ci_lo=lo, ratio_ci_hi=hi,
                   dist_saving_pct_upper=(1 - 1 / med_v) * 100 if med_v and med_v > 1 else 0.0,
                   frac_shorter_than_base=float((r_main < 0.98).mean()))
        for B in buffers:
            if f"B{B:g}" != mb:
                rr_ = grp[f"ratio_B{B:g}"].dropna()
                rec[f"ratio_median_B{B:g}"] = rr_.median()
                rec[f"frac_shorter_B{B:g}"] = float((rr_ < 0.98).mean())
        out.append(rec)
    summ = pd.DataFrame(out)
    if len(summ):
        summ = summ.round(3).sort_values("n", ascending=False)
    summ.to_csv(os.path.join(odir, "baseline_summary.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"\n=== 航廊摘要 (主 buffer {a.main_buffer:g} km; 實際航程 / 基準) ===")
    if len(summ):
        cols = ["origin", "dest", "class_zh", "n", "vessels", "actual_km_p50", "base_km_main",
                "ratio_median", "ratio_by_vessel_median", "ratio_ci_lo", "ratio_ci_hi",
                "dist_saving_pct_upper", "frac_shorter_than_base"]
        print(summ[cols].head(25).to_string(index=False))
    else:
        print("沒有航廊達到門檻 (--min-n / --min-vessels)")

    allr = vb[f"ratio_{mb}"].dropna()
    print(f"\n保留航次: 比值中位數 {allr.median():.3f}; 比基準還短 (<0.98) 的比例 {(allr < 0.98).mean():.1%}")
    for B in buffers:
        rr_ = vb[f"ratio_B{B:g}"].dropna()
        print(f"  buffer {B:g} km: 中位數 {rr_.median():.3f}; 比基準短的比例 {(rr_ < 0.98).mean():.1%}")
    print("  (比基準短的比例太高 => buffer 太寬或資料有跳躍; 要看這個再決定採用哪個 buffer)")

    # ---------- 地圖 ----------
    D, gates = results[a.main_buffer]
    P = preds[a.main_buffer]
    for pr in [x for x in a.pairs.split(";") if x.strip()]:
        pa = [x.strip() for x in pr.split(",")]
        if len(pa) != 2 or pa[0] not in pidx or pa[1] not in pidx:
            print(f"[略過] --pairs 格式或港口名稱錯誤: {pr!r}")
            continue
        go, gd = gates[pa[0]], gates[pa[1]]
        s_idx, d_idx = go[0] * C + go[1], gd[0] * C + gd[1]
        pth = trace_path(P[pidx[pa[0]]], s_idx, d_idx)
        ids = vb[((vb["origin"] == pa[0]) & (vb["dest"] == pa[1])) |
                 ((vb["origin"] == pa[1]) & (vb["dest"] == pa[0]))]["voyage_id"]
        if pth is None or len(ids) == 0:
            print(f"  {pa[0]}↔{pa[1]}: 沒有航次或路徑, 不畫圖")
            continue
        rr_ = np.array(pth) // C
        cc_ = np.array(pth) % C
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.imshow(land, origin="lower", extent=[lons[0], lons[-1], lats[0], lats[-1]],
                  cmap="Greys", alpha=.35, aspect=1 / np.cos(np.radians(24)))
        samp = ids.sample(min(200, len(ids)), random_state=0)
        for vid, dd in trk[trk["voyage_id"].isin(samp)].groupby("voyage_id"):
            ax.plot(dd["lon"], dd["lat"], color="#555", alpha=.15, lw=.8)
        ax.plot(lons[cc_], lats[rr_], color="red", lw=2,
                label=f"基準 (離岸>={a.main_buffer:g} km)" if zh else f"baseline (>= {a.main_buffer:g} km offshore)")
        for p in pa:
            lo, la, _ = PORTS[p]
            ax.scatter(lo, la, c="k", s=30, zorder=5)
            ax.text(lo + .03, la + .03, p, fontsize=9)
        lo_all = np.concatenate([lons[cc_], trk[trk["voyage_id"].isin(ids)]["lon"].values])
        la_all = np.concatenate([lats[rr_], trk[trk["voyage_id"].isin(ids)]["lat"].values])
        ax.set_xlim(lo_all.min() - .15, lo_all.max() + .15)
        ax.set_ylim(la_all.min() - .15, la_all.max() + .15)
        ax.set_title(f"{pa[0]}↔{pa[1]}: 基準路徑(紅) vs 真實航跡(灰, 抽樣 {len(samp)})" if zh
                     else f"{pa[0]}<->{pa[1]}: baseline (red) vs actual tracks (grey)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(odir, f"baseline_{pa[0]}_{pa[1]}.png"), dpi=170)
        plt.close(fig)
    print("\n輸出位置:", os.path.abspath(odir))


if __name__ == "__main__":
    main()
