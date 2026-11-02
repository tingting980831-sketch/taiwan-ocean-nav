"""
monthly_corr.py -- 吃水 vs 船長 相關係數的「逐月穩定性」分析

列 = 分級類別, 欄 = 1~12 月, 格子 = 該月該類的 log-log Pearson r

需要與 ais_profile.py 放在同一個資料夾 (會 import 它的清理與每船統計函式)。

用法 (Windows):
    python monthly_corr.py --ais-dir C:\\NODASS\\AIS --year 2023 --out C:\\NODASS\\out_2023

選項:
    --days 3,10,17,24     每月取哪幾天 (預設)
    --all-days            每月用全部天數 (很慢)
    --lang en             圖用英文標籤 (預設 zh)
    --min-n 15            格子樣本數低於此值不報告 r

第一次會逐日讀原始檔並把「每日每船一列」存到 <out>/daily/, 之後重跑會直接用快取。

輸出 (<out>):
    monthly_r.csv          類別 x 月份 的 r        (論文用主表)
    monthly_n.csv          類別 x 月份 的樣本數
    monthly_ratio.csv      類別 x 月份 的 吃水/船長 中位數
    monthly_summary.csv    各類別跨月平均/標準差/最小/最大
    monthly_long.csv       長表 (含 Spearman、迴歸斜率、95% 信賴區間)
    monthly_corr.png       熱圖
"""
import argparse
import calendar
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ais_profile as ap   # noqa: E402

# ------------------------------------------------------------------
# 分級定義 (key, 中文, English)
# ------------------------------------------------------------------
CLASSES = [
    ("fish_s",   "小型漁船 <24m",        "Fishing <24 m"),
    ("fish_m",   "中型漁船 24-50m",      "Fishing 24-50 m"),
    ("fish_l",   "大型漁船 >50m",        "Fishing >50 m"),
    ("cargo_s",  "近海貨輪 <100m",       "Cargo <100 m"),
    ("cargo_m",  "中型貨輪 100-200m",    "Cargo 100-200 m"),
    ("cargo_l",  "大型貨輪 200-300m",    "Cargo 200-300 m"),
    ("cargo_xl", "超大貨櫃船 >300m",     "Cargo >300 m"),
    ("tank_s",   "小型油輪 <100m",       "Tanker <100 m"),
    ("tank_m",   "中大型油輪 100-300m",  "Tanker 100-300 m"),
    ("tank_xl",  "超大型油輪 >300m",     "Tanker >300 m"),
    ("pax_conv", "客船(常規)",           "Passenger (conv.)"),
    ("pax_fast", "客船(高速)",           "Passenger (fast)"),
    ("tug",      "拖船/工作船",          "Tug / tow"),
]
POOLED = [
    ("pool_ct",  "【合併】貨輪+油輪",    "[Pooled] Cargo+Tanker"),
    ("pool_all", "【合併】全部有資料船", "[Pooled] All vessels"),
]


def assign_class(row):
    g, L = row["group"], row["loa"]
    if pd.isna(L):
        return None
    if g == "fishing":
        return "fish_s" if L < 24 else ("fish_m" if L < 50 else "fish_l")
    if g == "cargo":
        if L < 100: return "cargo_s"
        if L < 200: return "cargo_m"
        if L < 300: return "cargo_l"
        return "cargo_xl"
    if g == "tanker":
        if L < 100: return "tank_s"
        if L < 300: return "tank_m"
        return "tank_xl"
    if g == "high_speed":
        return "pax_fast"
    if g == "passenger":
        p90 = row.get("cruise_sog_p90", np.nan)
        return "pax_fast" if (pd.notna(p90) and p90 >= 20) else "pax_conv"
    if g == "tug_tow":
        return "tug"
    return None


# ------------------------------------------------------------------
# 統計
# ------------------------------------------------------------------
def corr_stats(d, min_n):
    """log-log Pearson r、Spearman、斜率、Fisher 95% CI、吃水/船長中位數。"""
    n = len(d)
    out = dict(n=n, r=np.nan, spearman=np.nan, slope=np.nan,
               ci_lo=np.nan, ci_hi=np.nan, ratio_med=np.nan)
    if n < 3:
        return out
    out["ratio_med"] = float((d["draft_p50"] / d["loa"]).median())
    if n < min_n:
        return out
    lx, ly = np.log(d["loa"].values), np.log(d["draft_p50"].values)
    if lx.std() == 0 or ly.std() == 0:
        return out
    r = float(np.corrcoef(lx, ly)[0, 1])
    out["r"] = r
    out["spearman"] = float(d["loa"].corr(d["draft_p50"], method="spearman"))
    out["slope"] = float(np.polyfit(lx, ly, 1)[0])
    if n > 3 and abs(r) < 1:
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        out["ci_lo"], out["ci_hi"] = float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))
    return out


def monthly_table(vm, month, min_n):
    """vm: 該月每船一列。回傳 list of dict。"""
    w = vm[vm["loa"].notna() & vm["draft_p50"].notna()].copy()
    ratio = w["draft_p50"] / w["loa"]
    w = w[(ratio > 0.015) & (ratio < 0.20)]          # 剔除明顯輸入錯誤
    w["cls"] = w.apply(assign_class, axis=1)

    rows = []
    for key, zh, en in CLASSES:
        rows.append(dict(month=month, cls=key, **corr_stats(w[w["cls"] == key], min_n)))
    rows.append(dict(month=month, cls="pool_ct",
                     **corr_stats(w[w["cls"].isin(
                         ["cargo_s", "cargo_m", "cargo_l", "cargo_xl",
                          "tank_s", "tank_m", "tank_xl"])], min_n)))
    rows.append(dict(month=month, cls="pool_all",
                     **corr_stats(w[w["cls"].notna()], min_n)))
    return rows


# ------------------------------------------------------------------
# 逐日資料 (含快取)
# ------------------------------------------------------------------
def daily_vessels(ais_dir, daily_dir, ymd, log):
    cache = os.path.join(daily_dir, f"vessels_{ymd}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"MMSI": str}, index_col="MMSI")
    raw = os.path.join(ais_dir, f"{ymd}.csv")
    if not os.path.exists(raw):
        log.append(f"  [缺檔] {raw}")
        return None
    rep = []
    df = ap.load_and_clean(raw, rep)
    v = ap.per_vessel(df)
    v.to_csv(cache, encoding="utf-8-sig")
    del df
    log.append(f"  {ymd}: {len(v):,} 艘")
    print(f"  {ymd}: {len(v):,} 艘", flush=True)
    return v


# ------------------------------------------------------------------
# 作圖
# ------------------------------------------------------------------
def setup_font(lang):
    if lang != "zh":
        return
    prefer = ["Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
              "Noto Sans CJK TC", "Noto Sans CJK JP", "SimHei", "Arial Unicode MS"]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in avail:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def heatmap(ax, mat, rows, cols, title, cmap, vmin, vmax, fmt):
    m = np.ma.masked_invalid(mat)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#e6e6e6")
    im = ax.imshow(m, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color="gray", fontsize=8)
            else:
                frac = (v - vmin) / (vmax - vmin)
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                        color="white" if frac > .75 or frac < .2 else "black")
    ax.set_title(title, fontweight="bold")
    return im


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ais-dir", required=True, help="放 YYYYMMDD.csv 的資料夾")
    ap_.add_argument("--year", type=int, default=2023)
    ap_.add_argument("--out", default="out_monthly")
    ap_.add_argument("--days", default="3,10,17,24")
    ap_.add_argument("--all-days", action="store_true")
    ap_.add_argument("--min-n", type=int, default=15)
    ap_.add_argument("--lang", default="zh", choices=["zh", "en"])
    a = ap_.parse_args()

    daily_dir = os.path.join(a.out, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    pick = [int(x) for x in a.days.split(",")]
    log, all_rows = [], []

    for month in range(1, 13):
        ndays = calendar.monthrange(a.year, month)[1]
        days = range(1, ndays + 1) if a.all_days else [d for d in pick if d <= ndays]
        print(f"== {a.year}-{month:02d}: {len(list(days))} 天", flush=True)
        frames = []
        for d in days:
            v = daily_vessels(a.ais_dir, daily_dir, f"{a.year}{month:02d}{d:02d}", log)
            if v is not None and len(v):
                frames.append(v)
        if not frames:
            print("   (本月無資料)")
            continue
        vm = ap.aggregate_vessels(frames) if len(frames) > 1 else frames[0].copy()
        all_rows += monthly_table(vm, month, a.min_n)
        print(f"   本月合併 {len(vm):,} 艘", flush=True)

    if not all_rows:
        sys.exit("沒有任何資料, 請檢查 --ais-dir 與檔名格式 (YYYYMMDD.csv)")

    long = pd.DataFrame(all_rows)
    name_zh = {k: z for k, z, _ in CLASSES + POOLED}
    name_en = {k: e for k, _, e in CLASSES + POOLED}
    order = [k for k, _, _ in CLASSES + POOLED]
    long["class_zh"] = long["cls"].map(name_zh)
    long.round(3).to_csv(os.path.join(a.out, "monthly_long.csv"),
                         index=False, encoding="utf-8-sig")

    def pivot(col):
        p = long.pivot(index="cls", columns="month", values=col).reindex(order)
        p = p.reindex(columns=range(1, 13))
        return p

    tr, tn, tq = pivot("r"), pivot("n"), pivot("ratio_med")
    for p, nm in ((tr, "monthly_r"), (tn, "monthly_n"), (tq, "monthly_ratio")):
        q = p.copy()
        q.index = [name_zh[k] for k in q.index]
        q.columns = [f"{m}月" for m in q.columns]
        q.round(3).to_csv(os.path.join(a.out, f"{nm}.csv"), encoding="utf-8-sig")

    summ = pd.DataFrame({
        "class": [name_zh[k] for k in order],
        "months_valid": tr.notna().sum(axis=1).values,
        "r_mean": tr.mean(axis=1).values,
        "r_std": tr.std(axis=1).values,
        "r_min": tr.min(axis=1).values,
        "r_max": tr.max(axis=1).values,
        "ratio_mean": tq.mean(axis=1).values,
        "ratio_cv": (tq.std(axis=1) / tq.mean(axis=1)).values,
        "n_mean": tn.mean(axis=1).values,
    }).round(3)
    summ.to_csv(os.path.join(a.out, "monthly_summary.csv"),
                index=False, encoding="utf-8-sig")
    print("\n" + summ.to_string(index=False))

    # ---- 圖 ----
    setup_font(a.lang)
    names = name_zh if a.lang == "zh" else name_en
    rlab = [names[k] for k in order]
    clab = [f"{m}" for m in range(1, 13)]
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.5))
    im1 = heatmap(axs[0], tr.values, rlab, clab,
                  "log-log Pearson r (draft vs LOA)" if a.lang == "en"
                  else "吃水–船長 相關係數 r (逐月)", "RdYlGn", 0, 1, ".2f")
    fig.colorbar(im1, ax=axs[0], fraction=.04)
    im2 = heatmap(axs[1], tq.values, [""] * len(order), clab,
                  "Median draft / LOA" if a.lang == "en" else "吃水/船長 中位數 (逐月)",
                  "YlOrBr", 0.02, 0.14, ".3f")
    axs[1].set_yticks([])
    fig.colorbar(im2, ax=axs[1], fraction=.04)
    for ax in axs:
        ax.set_xlabel("Month" if a.lang == "en" else "月份")
        ax.axhline(len(CLASSES) - .5, color="k", lw=1.5)   # 分隔合併列
    fig.suptitle(f"{a.year}  (grey '–' = n < {a.min_n})" if a.lang == "en"
                 else f"{a.year} 年  (灰色 '–' = 樣本數 < {a.min_n})", y=.99)
    fig.tight_layout()
    out = os.path.join(a.out, "monthly_corr.png")
    fig.savefig(out, dpi=200)
    print("\n已輸出到:", os.path.abspath(a.out))

    with open(os.path.join(a.out, "monthly_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))


if __name__ == "__main__":
    main()
