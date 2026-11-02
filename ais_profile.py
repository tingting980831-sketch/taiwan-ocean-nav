"""
ais_profile.py  --  AIS 船舶分級統計工具

用法 (Windows 命令列):
    python ais_profile.py C:\\NODASS\\AIS\\20221201.csv
    python ais_profile.py C:\\NODASS\\AIS\\20221201.csv --out C:\\NODASS\\out

多個檔案 (例如每季挑幾天), 會各自壓成「每艘船一列」後再合併:
    python ais_profile.py 20221201.csv 20230301.csv 20230601.csv --out out

輸出 (在 --out 資料夾):
    vessels_<檔名>.csv     每檔的每艘船一列
    vessels_all.csv        合併後每艘船一列
    class_table.csv        船型 x 船長分級統計
    quality_report.txt     缺失率、單位檢查等
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------
USECOLS = [
    "IMO_Number", "MMSI", "Navigational_Status", "SOG",
    "Longitude", "Latitude", "COG", "Ship_and_Cargo_Type",
    "Reference_Position_A", "Reference_Position_B",
    "Reference_Position_C", "Reference_Position_D",
    "MAX_Draught", "Gross_Tonnage", "Record_Time",
]

LOA_BINS = [0, 24, 50, 100, 150, 200, 300, 500]

# 研究範圍 (與 HELIOS 相同), 設 None 則不裁切
BBOX = dict(lon_min=118.0, lon_max=124.0, lat_min=21.0, lat_max=26.0)

NON_VESSEL_PREFIX = ("99", "00", "970", "972", "974")


# ------------------------------------------------------------------
# 工具函式
# ------------------------------------------------------------------
def imo_valid(x):
    """IMO 檢查碼: 前六位依序乘 7,6,5,4,3,2, 加總個位數 = 第七位。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).split(".")[0].strip()
    if not (s.isdigit() and len(s) == 7):
        return False
    total = sum(int(d) * w for d, w in zip(s[:6], range(7, 1, -1)))
    return total % 10 == int(s[6])


def type_group(code):
    """AIS Ship_and_Cargo_Type -> 大類。"""
    if pd.isna(code):
        return "unknown"
    c = int(code)
    if c == 30:
        return "fishing"
    if c in (31, 32, 52):
        return "tug_tow"
    if 36 <= c <= 37:
        return "pleasure"
    if 40 <= c <= 49:
        return "high_speed"
    if 60 <= c <= 69:
        return "passenger"
    if 70 <= c <= 79:
        return "cargo"
    if 80 <= c <= 89:
        return "tanker"
    return "other"


def mode_nonzero(df, key, col):
    """每個 key 中, col 大於 0 的值的眾數 (向量化, 比 groupby.apply 快很多)。"""
    sub = df.loc[df[col] > 0, [key, col]]
    if sub.empty:
        return pd.Series(dtype=float, name=col)
    cnt = sub.groupby([key, col]).size().reset_index(name="n")
    idx = cnt.groupby(key)["n"].idxmax()
    return cnt.loc[idx].set_index(key)[col].rename(col)


# ------------------------------------------------------------------
# 讀取與清理
# ------------------------------------------------------------------
def load_and_clean(path, report):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in USECOLS if c in header]
    missing = [c for c in USECOLS if c not in header]
    if missing:
        report.append(f"[警告] 檔案缺少欄位: {missing}")

    df = pd.read_csv(path, usecols=use,
                     dtype={"MMSI": str, "IMO_Number": str},
                     low_memory=False)
    for c in missing:
        df[c] = np.nan

    report.append(f"原始列數: {len(df):,}   不同 MMSI: {df['MMSI'].nunique():,}")

    df["Record_Time"] = pd.to_datetime(df["Record_Time"], errors="coerce")

    # 只保留 9 位數 MMSI, 排除助航設備/岸台/求救設備
    m = df["MMSI"].astype(str).str.strip()
    ok = m.str.fullmatch(r"\d{9}") & ~m.str.startswith(NON_VESSEL_PREFIX)
    df = df[ok].copy()
    df["MMSI"] = m[ok]
    report.append(f"排除非船舶/格式錯誤 MMSI 後: {len(df):,} 列, "
                  f"{df['MMSI'].nunique():,} 艘")

    # 無效位置
    bad_pos = (~df["Latitude"].between(-90, 90)) | (~df["Longitude"].between(-180, 180)) \
              | (df["Latitude"].abs() > 89.9) | (df["Longitude"].abs() > 179.9)
    df = df[~bad_pos]
    if BBOX is not None:
        inside = df["Longitude"].between(BBOX["lon_min"], BBOX["lon_max"]) & \
                 df["Latitude"].between(BBOX["lat_min"], BBOX["lat_max"])
        df = df[inside].copy()
        report.append(f"裁切研究範圍後: {len(df):,} 列, {df['MMSI'].nunique():,} 艘")

    # 船體尺寸: 0 視為未提供
    for k in ("A", "B", "C", "D"):
        col = f"Reference_Position_{k}"
        df.loc[df[col] <= 0, col] = np.nan
    df["loa"] = df["Reference_Position_A"] + df["Reference_Position_B"]
    df["beam"] = df["Reference_Position_C"] + df["Reference_Position_D"]
    ratio = df["loa"] / df["beam"]
    bad = (~df["loa"].between(5, 400)) | (~df["beam"].between(2, 70)) \
          | (~ratio.between(2, 12))
    n_bad = int((bad & df["loa"].notna()).sum())
    df.loc[bad, ["loa", "beam"]] = np.nan
    report.append(f"尺寸不合理而剔除: {n_bad:,} 列")

    # 吃水: 0 = 未填; 偵測單位
    d = df["MAX_Draught"].where(df["MAX_Draught"] > 0)
    dmax = d.max()
    report.append(f"MAX_Draught 最大值(原始): {dmax}")
    if pd.notna(dmax):
        if dmax > 25.5:
            if dmax <= 255:
                report.append("[注意] 最大值 > 25.5, 判斷單位為 0.1 公尺, 已除以 10")
                d = d / 10.0
            else:
                report.append("[注意] 最大值 > 255, 單位無法判斷, 超出範圍者設為缺失")
    d = d.where(d <= 25.5)
    df["draft"] = d

    # 噸位
    df["gt"] = df["Gross_Tonnage"].where(df["Gross_Tonnage"] > 0)

    # IMO
    df["imo_ok"] = df["IMO_Number"].apply(imo_valid)

    # 非「未提供」的船型
    df["type_code"] = df["Ship_and_Cargo_Type"].where(df["Ship_and_Cargo_Type"] > 0)

    # 欄位非空比例
    cov = pd.DataFrame({
        "MAX_Draught": [df["draft"].notna().mean()],
        "Gross_Tonnage": [df["gt"].notna().mean()],
        "Ship_and_Cargo_Type": [df["type_code"].notna().mean()],
        "LOA/beam": [df["loa"].notna().mean()],
        "IMO(有效)": [df["imo_ok"].mean()],
    }).T.rename(columns={0: "列非空比例"})
    report.append("\n欄位非空比例(以列計):\n" + cov.round(3).to_string())
    return df


# ------------------------------------------------------------------
# 每艘船一列
# ------------------------------------------------------------------
def per_vessel(df):
    key = "MMSI"
    g = df.groupby(key)

    # 巡航速度: 只用航速合理且在航/未定義狀態的紀錄
    under = df[df["SOG"].between(3, 40) &
               df["Navigational_Status"].isin([0, 8, 15])]
    cruise = under.groupby(key)["SOG"].median().rename("cruise_sog")
    cruise_p90 = under.groupby(key)["SOG"].quantile(0.9).rename("cruise_sog_p90")

    out = pd.DataFrame({
        "loa": g["loa"].median(),
        "beam": g["beam"].median(),
        "n_reports": g.size(),
        "draft_p50": g["draft"].median(),
        "draft_p90": g["draft"].quantile(0.9),
        "draft_max": g["draft"].max(),
        "gt": g["gt"].median(),
        "has_imo": g["imo_ok"].max(),
    })
    out["type_code"] = mode_nonzero(df, key, "type_code")
    out = out.join(cruise).join(cruise_p90)
    out["group"] = out["type_code"].apply(type_group)
    return out


def aggregate_vessels(frames):
    """合併多個每檔的每船表: 數值取中位數, 船型取眾數, 回報數加總。"""
    allv = pd.concat(frames)
    allv = allv.reset_index().rename(columns={"index": "MMSI"})
    key = "MMSI"
    g = allv.groupby(key)
    out = pd.DataFrame({
        "loa": g["loa"].median(),
        "beam": g["beam"].median(),
        "n_reports": g["n_reports"].sum(),
        "draft_p50": g["draft_p50"].median(),
        "draft_p90": g["draft_p90"].median(),
        "draft_max": g["draft_max"].max(),
        "gt": g["gt"].median(),
        "has_imo": g["has_imo"].max(),
        "cruise_sog": g["cruise_sog"].median(),
        "cruise_sog_p90": g["cruise_sog_p90"].median(),
        "n_days": g.size(),
    })
    out["type_code"] = mode_nonzero(allv, key, "type_code")
    out["group"] = out["type_code"].apply(type_group)
    return out


# ------------------------------------------------------------------
# 分級統計表
# ------------------------------------------------------------------
def class_table(v):
    v = v.copy()
    v["loa_bin"] = pd.cut(v["loa"], LOA_BINS)
    v = v[v["loa_bin"].notna()]

    def q(p):
        return lambda s: s.quantile(p) if s.notna().any() else np.nan

    tab = (v.groupby(["group", "loa_bin"], observed=True)
             .agg(n=("loa", "size"),
                  beam_med=("beam", "median"),
                  draft_cov=("draft_p50", lambda s: s.notna().mean()),
                  draft_p10=("draft_p50", q(0.1)),
                  draft_p50=("draft_p50", "median"),
                  draft_p90=("draft_p50", q(0.9)),
                  gt_cov=("gt", lambda s: s.notna().mean()),
                  cruise_sog_p50=("cruise_sog", "median"),
                  cruise_sog_p90=("cruise_sog", q(0.9)))
             .round(2))
    return tab


# ------------------------------------------------------------------
# 主程式
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="AIS 船舶分級統計")
    ap.add_argument("files", nargs="+", help="AIS csv 檔 (可多個)")
    ap.add_argument("--out", default="ais_out", help="輸出資料夾")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = []
    frames = []

    for path in args.files:
        name = os.path.splitext(os.path.basename(path))[0]
        report.append(f"\n===== {path} =====")
        if not os.path.exists(path):
            report.append("[錯誤] 找不到檔案, 略過")
            continue
        df = load_and_clean(path, report)
        v = per_vessel(df)
        v.to_csv(os.path.join(args.out, f"vessels_{name}.csv"),
                 encoding="utf-8-sig")
        report.append(f"本檔船舶數: {len(v):,}")
        frames.append(v)
        del df

    if not frames:
        print("\n".join(report))
        sys.exit("沒有可用資料")

    if len(frames) == 1:
        allv = frames[0].copy()
        allv["n_days"] = 1
        allv = allv.rename(columns={})
    else:
        allv = aggregate_vessels(frames)

    allv.to_csv(os.path.join(args.out, "vessels_all.csv"), encoding="utf-8-sig")

    tab = class_table(allv)
    tab.to_csv(os.path.join(args.out, "class_table.csv"), encoding="utf-8-sig")

    report.append("\n===== 各船型數量 =====")
    report.append(allv["group"].value_counts().to_string())
    report.append("\n===== class_table (船型 x 船長分級) =====")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        report.append(tab.to_string())

    text = "\n".join(report)
    with open(os.path.join(args.out, "quality_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n輸出已存到: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
