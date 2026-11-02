"""
fuel_model.py -- 沿著一條路徑計算油耗 (簡化的物理模型, 用於比較路徑, 不是設計用的精密模型)

核心想法 (白話):
  1. 船以固定的「對水速度」前進; 海流改變「對地速度」(順流變快、逆流變慢)。
  2. 平靜水域的阻力 ~ 速度平方; 功率 ~ 速度立方 (主機功率用 Admiralty 係數從排水量與服務航速推估)。
  3. 風與浪只增加「額外阻力」, 為了維持同樣對水速度, 功率增加。
  4. 油耗 = 單位油耗 x 功率 x 時間。
  5. 公平比較: 不同路徑要在「同一個出發時間、同一個到達時間」下比油耗 (fuel_fixed_eta),
     否則省下的油可能只是因為開得比較慢。

所有物理公式都是常用的簡化近似, 係數都放在 Ship 裡, 可調整, 結論需做敏感度分析。
假設的係數 (請在報告中列為假設, 不是由資料推得):
  c_adm=500 (Admiralty 係數), eta=0.65 (推進效率), sfoc=190 g/kWh,
  k_aw=6 (波浪附加阻力無因次係數), cx=0.9 (風阻係數), p_max_ratio=1.25 (主機最大功率/服務功率)

座標與方向慣例:
  航向/航跡: 由北順時針的度數; 風 (u10, v10): 風「吹向」的方向分量 (m/s);
  浪向 mwd: 假設為「浪從哪個方向來」 (ERA5 mean wave direction; 實作前需對照官方文件確認,
  若相反, 把 Ship.wave_dir_is_from 設為 False)。

用法:
    python fuel_model.py --selftest
"""
import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

KM_PER_DEG = 111.2
KN = 0.514444          # 1 節 = 0.514444 m/s
RHO_W, RHO_A, G = 1025.0, 1.225, 9.81

# 各級船的方塊係數 (假設值, 依船型的典型範圍)
CB_BY_CLASS = {
    "fish_s": 0.50, "fish_m": 0.55, "fish_l": 0.58,
    "cargo_s": 0.72, "cargo_m": 0.72, "cargo_l": 0.75, "cargo_xl": 0.62,
    "tank_s": 0.78, "tank_m": 0.80, "tank_xl": 0.83,
    "pax_conv": 0.60, "pax_fast": 0.45, "tug": 0.55,
}


@dataclass
class Ship:
    name: str
    loa: float               # 船長 m
    beam: float              # 船寬 m
    draft: float             # 吃水 m
    cb: float                # 方塊係數
    vs_kn: float             # 服務航速 (對水, 節)
    c_adm: float = 500.0
    eta: float = 0.65
    sfoc: float = 190.0      # g/kWh
    cx: float = 0.9
    k_aw: float = 6.0
    p_max_ratio: float = 1.25
    wave_dir_is_from: bool = True
    # 衍生量
    disp_t: float = field(init=False)
    ps_kw: float = field(init=False)
    rs_n: float = field(init=False)
    af_m2: float = field(init=False)

    def __post_init__(self):
        self.disp_t = self.loa * self.beam * self.draft * self.cb * 1.025
        self.ps_kw = self.disp_t ** (2 / 3) * self.vs_kn ** 3 / self.c_adm
        self.rs_n = self.eta * self.ps_kw * 1000.0 / (self.vs_kn * KN)     # 服務航速下的總阻力 (N)
        self.af_m2 = self.beam * (0.08 * self.loa + 4.0)                     # 正面受風面積 (粗估)


def ships_from_profiles(csv_path, draft_state="p50"):
    """從 ship_profiles(_v2).csv 建立各級 Ship。draft_state: p10(輕載) / p50 / p90(滿載)。"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    vcol = "service_kn_p50" if "service_kn_p50" in df.columns else "cruise_kn_p50"
    out = {}
    for _, r in df.iterrows():
        try:
            out[r["class_key"]] = Ship(r["class_key"], float(r["loa_p50"]), float(r["beam_p50"]),
                                       float(r[f"draft_{draft_state}"]), CB_BY_CLASS.get(r["class_key"], 0.7),
                                       float(r[vcol]))
        except (KeyError, ValueError, TypeError):
            continue
    return out


# ------------------------------------------------------------------
# 幾何
# ------------------------------------------------------------------
def segments(lon, lat):
    """相鄰點之間的長度(km)與航跡方向(度, 由北順時針)。"""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    latm = (lat[:-1] + lat[1:]) / 2
    dx = (lon[1:] - lon[:-1]) * KM_PER_DEG * np.cos(np.radians(latm))
    dy = (lat[1:] - lat[:-1]) * KM_PER_DEG
    return np.hypot(dx, dy), (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def resample_path(lon, lat, step_km=5.0):
    """沿路徑每 step_km 取一點 (保留首尾)。"""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    d, _ = segments(lon, lat)
    cum = np.concatenate([[0], np.cumsum(d)])
    n = max(int(np.ceil(cum[-1] / step_km)), 1)
    s = np.linspace(0, cum[-1], n + 1)
    return np.interp(s, cum, lon), np.interp(s, cum, lat)


# ------------------------------------------------------------------
# 一段的物理
# ------------------------------------------------------------------
def segment_physics(ship, V_ms, track_deg, env):
    """固定對水速度 V_ms (m/s) 走一段。env: dict, 可含 u_cur v_cur u10 v10 swh mwd (缺值當 0)。
    回傳 (對地速度 m/s, 需要功率 kW, 實際對水速度 m/s)。"""
    e = {k: (0.0 if (v is None or not np.isfinite(v)) else float(v)) for k, v in env.items()}
    th = np.radians(track_deg)
    et = np.array([np.sin(th), np.cos(th)])                    # 航跡單位向量 (東, 北)
    cur = np.array([e.get("u_cur", 0.0), e.get("v_cur", 0.0)])
    c_par = float(cur @ et)
    c_perp = float(cur @ np.array([et[1], -et[0]]))
    wind = np.array([e.get("u10", 0.0), e.get("v10", 0.0)])
    hs = e.get("swh", 0.0)
    wave_from = e.get("mwd", 0.0)
    if not ship.wave_dir_is_from:
        wave_from = (wave_from + 180.0) % 360.0
    cos_rel = np.cos(np.radians(wave_from - track_deg))
    f_wave = ((1 + cos_rel) / 2) ** 2                          # 迎浪 1, 橫浪 0.25, 順浪 0

    def power_for(V):
        if V <= abs(c_perp):
            return np.nan, np.nan
        vg = c_par + np.sqrt(V * V - c_perp ** 2)               # 側流需要修正航向, 損失一點前進速度
        r_cw = ship.rs_n * (V / (ship.vs_kn * KN)) ** 2
        rel_long = float((wind - vg * et) @ et)                  # 相對風的縱向分量
        r_wind = -0.5 * RHO_A * ship.cx * ship.af_m2 * rel_long * abs(rel_long)
        r_wind_calm = 0.5 * RHO_A * ship.cx * ship.af_m2 * vg * vg    # 無風時船自己造成的空氣阻力
        r_aw = ship.k_aw * RHO_W * G * (hs * hs / 8.0) * (ship.beam ** 2 / ship.loa) * f_wave
        r_tot = max(r_cw + (r_wind - r_wind_calm) + r_aw, 0.05 * r_cw)
        return vg, r_tot * V / ship.eta / 1000.0

    vg, p = power_for(V_ms)
    p_max = ship.p_max_ratio * ship.ps_kw
    if np.isfinite(p) and p > p_max:                              # 海況太差, 主機頂不住 -> 被迫減速
        lo, hi = 0.2 * V_ms, V_ms
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            _, pm = power_for(mid)
            if np.isfinite(pm) and pm > p_max:
                hi = mid
            else:
                lo = mid
        V_ms = lo
        vg, p = power_for(V_ms)
    return vg, p, V_ms


# ------------------------------------------------------------------
# 沿路徑
# ------------------------------------------------------------------
def run_path(ship, env, lon, lat, t0, V_kn, step_km=5.0):
    """以固定對水速度 V_kn (節) 沿路徑前進。env.at(t, lat_array, lon_array) -> dict 陣列。
    回傳 dict: time_h, fuel_t(噸), dist_km, missing_frac, speed_loss(是否曾被迫減速)。"""
    lo, la = resample_path(lon, lat, step_km)
    d_km, trk = segments(lo, la)
    t = np.datetime64(t0, "s")
    total_s, fuel_kg, miss, loss = 0.0, 0.0, 0, False
    V = V_kn * KN
    for i in range(len(d_km)):
        mlat, mlon = (la[i] + la[i + 1]) / 2, (lo[i] + lo[i + 1]) / 2
        ev = env.at(t, [mlat], [mlon]) if env is not None else {}
        e1 = {k: (v[0] if hasattr(v, "__len__") else v) for k, v in ev.items()}
        if any(not np.isfinite(e1.get(k, 0.0)) for k in ("u_cur", "swh", "u10") if k in e1):
            miss += 1
        vg, p, v_eff = segment_physics(ship, V, trk[i], e1)
        if v_eff < V - 1e-9:
            loss = True
        if not np.isfinite(vg) or vg <= 0.05:
            return dict(time_h=np.inf, fuel_t=np.inf, dist_km=float(d_km.sum()), missing_frac=1.0, speed_loss=True)
        dt = d_km[i] * 1000.0 / vg
        total_s += dt
        fuel_kg += ship.sfoc / 1000.0 * p * dt / 3600.0
        t = t + np.timedelta64(int(round(dt)), "s")
    return dict(time_h=total_s / 3600.0, fuel_t=fuel_kg / 1000.0, dist_km=float(d_km.sum()),
                missing_frac=miss / max(len(d_km), 1), speed_loss=loss)


def fuel_fixed_eta(ship, env, lon, lat, t0, T_h, step_km=5.0, v_lo_kn=2.0, v_hi_ratio=1.3, tol_h=1e-3):
    """找出剛好在 T_h 小時走完這條路徑所需的對水速度, 回傳該速度下的結果 (含 V_kn)。
    無法在 T_h 內走完 (需要超過 v_hi_ratio 倍服務航速) -> 回傳 feasible=False。"""
    v_hi = v_hi_ratio * ship.vs_kn
    r_hi = run_path(ship, env, lon, lat, t0, v_hi, step_km)
    if not np.isfinite(r_hi["time_h"]) or r_hi["time_h"] > T_h + tol_h:
        r_hi.update(V_kn=v_hi, feasible=False)
        return r_hi
    r_lo = run_path(ship, env, lon, lat, t0, v_lo_kn, step_km)
    if r_lo["time_h"] <= T_h:                                   # 慢速也夠快 -> 不需要這麼慢; 以慢速為界
        r_lo.update(V_kn=v_lo_kn, feasible=True)
        return r_lo
    lo, hi = v_lo_kn, v_hi
    best = r_hi
    bestv = v_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = run_path(ship, env, lon, lat, t0, mid, step_km)
        if r["time_h"] > T_h:
            lo = mid
        else:
            hi, best, bestv = mid, r, mid
        if abs(r["time_h"] - T_h) < tol_h:
            best, bestv = r, mid
            break
    best = dict(best)
    best.update(V_kn=bestv, feasible=True)
    return best


# ------------------------------------------------------------------
# 加速版: 沿路徑只查一次海況 (用「名目時間軸」), 之後解船速時重複使用
# ------------------------------------------------------------------
def prepare_path(env, lon, lat, t0, T_h, step_km=5.0):
    """沿路徑切段, 並依「名目時間軸」(等速走完 T_h 小時) 一次查好每段的海況。
    船速改變造成的實際到達時間差異, 相對於海況資料的時間解析度 (3~6 小時) 可忽略。"""
    lo, la = resample_path(lon, lat, step_km)
    d_km, trk = segments(lo, la)
    L = float(d_km.sum())
    cum = np.cumsum(d_km) - d_km / 2
    mlat, mlon = (la[:-1] + la[1:]) / 2, (lo[:-1] + lo[1:]) / 2
    if env is None or L <= 0:
        evs, miss = [{} for _ in d_km], 0.0
    else:
        tt = np.datetime64(t0, "s") + np.round(cum / L * T_h * 3600).astype("int64").astype("timedelta64[s]")
        ev = env.at(tt, mlat, mlon)
        evs = [{k: float(v[i]) for k, v in ev.items()} for i in range(len(d_km))]
        keys = [k for k in ("u_cur", "u10", "swh") if k in ev]
        bad = np.zeros(len(d_km), dtype=bool)
        for k in keys:
            bad |= ~np.isfinite(ev[k])
        miss = float(bad.mean())
    return dict(d_km=d_km, trk=trk, env=evs, missing_frac=miss, dist_km=L)


def run_prepared(ship, prep, V_kn):
    V = V_kn * KN
    total_s, fuel_kg, loss = 0.0, 0.0, False
    for i in range(len(prep["d_km"])):
        vg, p, v_eff = segment_physics(ship, V, prep["trk"][i], prep["env"][i])
        if v_eff < V - 1e-9:
            loss = True
        if not np.isfinite(vg) or vg <= 0.05:
            return dict(time_h=np.inf, fuel_t=np.inf, speed_loss=True)
        dt = prep["d_km"][i] * 1000.0 / vg
        total_s += dt
        fuel_kg += ship.sfoc / 1000.0 * p * dt / 3600.0
    return dict(time_h=total_s / 3600.0, fuel_t=fuel_kg / 1000.0, speed_loss=loss)


def fixed_eta_prepared(ship, prep, T_h, v_lo_kn=2.0, v_hi_ratio=1.3):
    """在 T_h 小時內走完所需的對水速度與油耗。feasible=False: 要超過 v_hi_ratio 倍服務航速才趕得上。"""
    from scipy.optimize import brentq
    v_hi = v_hi_ratio * ship.vs_kn
    r_hi = run_prepared(ship, prep, v_hi)
    if not np.isfinite(r_hi["time_h"]) or r_hi["time_h"] > T_h:
        return dict(r_hi, V_kn=v_hi, feasible=False, slack=False)
    r_lo = run_prepared(ship, prep, v_lo_kn)
    if r_lo["time_h"] <= T_h:                                   # 時間非常充裕, 以最低速度為界
        return dict(r_lo, V_kn=v_lo_kn, feasible=True, slack=True)
    f = lambda v: run_prepared(ship, prep, v)["time_h"] - T_h   # noqa: E731
    v = brentq(f, v_lo_kn, v_hi, xtol=1e-3)
    return dict(run_prepared(ship, prep, v), V_kn=v, feasible=True, slack=False)


# ------------------------------------------------------------------
# 自我測試 (極限情況: 結果必須符合物理直覺或可手算的答案)
# ------------------------------------------------------------------
class ConstEnv:
    def __init__(self, **kw):
        self.kw = kw

    def at(self, t, lat, lon):
        n = len(np.atleast_1d(lat))
        return {k: np.full(n, float(v)) for k, v in self.kw.items()}


def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'通過' if cond else '失敗'}] {name} {extra}")

    ship = Ship("測試貨輪", 150, 25, 8, 0.72, 12.0)
    print(f"測試船: 排水量 {ship.disp_t:,.0f} t, 服務功率 {ship.ps_kw:,.0f} kW, "
          f"服務航速總阻力 {ship.rs_n / 1000:,.0f} kN, 正面受風面積 {ship.af_m2:,.0f} m^2")
    lon0, lat0 = 120.0, 23.0
    d_km = 100.0
    lat1 = lat0 + d_km / KM_PER_DEG                              # 正北 100 km
    N = (np.array([lon0, lon0]), np.array([lat0, lat1]))
    S = (np.array([lon0, lon0]), np.array([lat1, lat0]))
    E = (np.array([lon0, lon0 + d_km / (KM_PER_DEG * np.cos(np.radians(lat0)))]), np.array([lat0, lat0]))
    t0 = "2023-03-01T00:00:00"
    vs = ship.vs_kn

    print("1. 無海況: 油耗 = 單位油耗 x 服務功率 x 時間")
    r = run_path(ship, ConstEnv(), *N, t0, vs)
    t_exp = d_km * 1000 / (vs * KN) / 3600
    f_exp = ship.sfoc / 1e6 * ship.ps_kw * t_exp
    check("航行時間 = 距離/速度", abs(r["time_h"] - t_exp) < 1e-3, f"({r['time_h']:.3f} vs {t_exp:.3f} h)")
    check("油耗 = SFOC x Ps x t", abs(r["fuel_t"] - f_exp) / f_exp < 1e-3, f"({r['fuel_t']:.3f} vs {f_exp:.3f} t)")

    print("2. 距離加倍 -> 時間與油耗加倍")
    lat2 = lat0 + 2 * d_km / KM_PER_DEG
    r2 = run_path(ship, ConstEnv(), np.array([lon0, lon0]), np.array([lat0, lat2]), t0, vs)
    check("距離加倍", abs(r2["fuel_t"] / r["fuel_t"] - 2) < 1e-3)

    print("3. 無海況時, A->B 與 B->A 完全相同")
    rb = run_path(ship, ConstEnv(), *S, t0, vs)
    check("往返對稱", abs(rb["fuel_t"] - r["fuel_t"]) < 1e-6)

    print("4. 順流 1 m/s (固定對水速度): 時間縮短, 油耗等比例減少")
    rc = run_path(ship, ConstEnv(u_cur=0.0, v_cur=1.0), *N, t0, vs)
    t_c = d_km * 1000 / (vs * KN + 1.0) / 3600
    check("順流時間 = 距離/(V+c)", abs(rc["time_h"] - t_c) < 1e-3, f"({rc['time_h']:.3f} vs {t_c:.3f} h)")
    check("固定速度下, 順流省的是時間 (每小時耗油不變)",
          abs(rc["fuel_t"] / rc["time_h"] - r["fuel_t"] / r["time_h"]) < 1e-3)
    ru = run_path(ship, ConstEnv(u_cur=0.0, v_cur=-1.0), *N, t0, vs)
    check("逆流比無流慢、耗油多", ru["time_h"] > r["time_h"] and ru["fuel_t"] > r["fuel_t"])

    print("5. 側流: 對地速度 = sqrt(V^2 - c_perp^2)")
    re_ = run_path(ship, ConstEnv(u_cur=1.0, v_cur=0.0), *N, t0, vs)
    vg_exp = np.sqrt((vs * KN) ** 2 - 1.0)
    check("側流損失前進速度", abs(re_["time_h"] - d_km * 1000 / vg_exp / 3600) < 1e-3)

    print("6. 固定到達時間 (公平比較): 順流 -> 可以降速 -> 省油 (立方效應)")
    T = t_exp                                                   # 無流時剛好用服務航速
    f0 = fuel_fixed_eta(ship, ConstEnv(), *N, t0, T)
    f1 = fuel_fixed_eta(ship, ConstEnv(v_cur=1.0), *N, t0, T)
    v_need = d_km * 1000 / (T * 3600) - 1.0                     # m/s
    f_exp1 = ship.sfoc / 1e6 * ship.ps_kw * (v_need / (vs * KN)) ** 3 * T
    check("無流: 解出的速度 = 服務航速", abs(f0["V_kn"] - vs) < 0.01, f"({f0['V_kn']:.3f} 節)")
    check("順流: 解出的速度 = 距離/T - 流速", abs(f1["V_kn"] * KN - v_need) < 0.01)
    check("順流: 油耗 = Ps x (V/Vs)^3 x T x SFOC", abs(f1["fuel_t"] - f_exp1) / f_exp1 < 0.01,
          f"({f1['fuel_t']:.3f} vs {f_exp1:.3f} t; 省 {100 * (1 - f1['fuel_t'] / f0['fuel_t']):.1f}%)")
    f2 = fuel_fixed_eta(ship, ConstEnv(v_cur=-1.0), *N, t0, T)
    check("逆流: 要加速、油耗增加", f2["V_kn"] > vs and f2["fuel_t"] > f0["fuel_t"])

    print("7. 浪: 迎浪 > 橫浪 > 順浪 >= 無浪 (方向為「浪從哪來」)")
    base = r["fuel_t"]
    head = run_path(ship, ConstEnv(swh=2.0, mwd=0.0), *N, t0, vs)["fuel_t"]       # 朝北走, 浪從北來
    beam = run_path(ship, ConstEnv(swh=2.0, mwd=90.0), *N, t0, vs)["fuel_t"]
    foll = run_path(ship, ConstEnv(swh=2.0, mwd=180.0), *N, t0, vs)["fuel_t"]
    check("迎浪比無浪費油", head > base, f"(+{100 * (head / base - 1):.1f}%)")
    check("順序: 迎浪 > 橫浪 > 順浪", head > beam > foll, f"(迎 {head:.2f} 橫 {beam:.2f} 順 {foll:.2f})")
    check("順浪不會比無浪省油", foll >= base - 1e-9)
    big = run_path(ship, ConstEnv(swh=4.0, mwd=0.0), *N, t0, vs)["fuel_t"]
    check("浪高加倍, 額外油耗約 4 倍 (平方關係)",
          3.0 < (big / base - 1) / (head / base - 1) < 4.6, f"(比值 {(big / base - 1) / (head / base - 1):.2f})")

    print("8. 風: 迎風增加油耗, 順風減少")
    wind_head = run_path(ship, ConstEnv(u10=0.0, v10=-12.0), *N, t0, vs)["fuel_t"]    # 往北走, 風往南吹 = 迎風
    wind_tail = run_path(ship, ConstEnv(u10=0.0, v10=12.0), *N, t0, vs)["fuel_t"]
    check("迎風 > 無風 > 順風", wind_head > base > wind_tail,
          f"(迎 +{100 * (wind_head / base - 1):.1f}%, 順 {100 * (wind_tail / base - 1):.1f}%)")

    print("9. 海況太差: 主機頂不住, 被迫減速 (時間變長)")
    storm = run_path(ship, ConstEnv(swh=9.0, mwd=0.0), *N, t0, vs)
    check("極端迎浪觸發減速", storm["speed_loss"] and storm["time_h"] > r["time_h"],
          f"(時間 {storm['time_h']:.1f} h vs {r['time_h']:.1f} h)")

    print("10. 東西向與南北向在無海況時, 同距離的油耗相同")
    re2 = run_path(ship, ConstEnv(), *E, t0, vs)
    check("與方向無關", abs(re2["fuel_t"] - r["fuel_t"]) / r["fuel_t"] < 5e-3)

    print("11. 加速版 (沿路徑只查一次海況) 與原版結果一致")
    env_c = ConstEnv(v_cur=0.7, swh=1.5, mwd=20.0, u10=2.0, v10=-5.0)
    prep = prepare_path(env_c, *N, t0, T, step_km=5.0)
    fa = fixed_eta_prepared(ship, prep, T)
    fb = fuel_fixed_eta(ship, env_c, *N, t0, T)
    check("速度一致", abs(fa["V_kn"] - fb["V_kn"]) < 0.02, f"({fa['V_kn']:.3f} vs {fb['V_kn']:.3f} 節)")
    check("油耗一致", abs(fa["fuel_t"] - fb["fuel_t"]) / fb["fuel_t"] < 0.005,
          f"({fa['fuel_t']:.3f} vs {fb['fuel_t']:.3f} t)")
    p_none = prepare_path(None, *N, t0, T)
    check("無海況版 = 服務航速", abs(fixed_eta_prepared(ship, p_none, T)["V_kn"] - vs) < 0.01)
    p_far = prepare_path(None, np.array([lon0, lon0]), np.array([lat0, lat0 + 600 / KM_PER_DEG]), t0, T)
    check("趕不上時回報 feasible=False", not fixed_eta_prepared(ship, p_far, T)["feasible"])

    print("\n全部通過" if ok else "\n有項目失敗, 請檢查")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    print("用法: python fuel_model.py --selftest   (或在程式中 from fuel_model import Ship, run_path, fuel_fixed_eta)")
