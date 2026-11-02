"""
helios_core.py -- HELIOS 系統的路徑規劃核心 (不含 Streamlit, 可獨立測試)

與回測評估 (evaluate_routes.py) 使用同一套東西:
  基準路徑 (baseline.py)  ->  planner.plan_path (油耗 + lam * 時間)  ->  fuel_model 的物理油耗
所以系統展示的演算法 = 說明書「節能評估」評估的演算法。

流程 (route_plan):
  1. 離岸 >= buffer_km 的海上最短路徑 = 基準路徑 (起訖點取最近的可通行格點)
  2. 目標航速 -> 要求的航行時間 T_h; 在預報海況下解出基準路徑剛好準時所需的對水速度與油耗
  3. 規劃器在基準路徑兩側 corridor_km 內找油耗最小的路徑; 若沒有比基準好就沿用基準
  4. 回傳兩條路徑的距離、時間、油耗與沿途時間表 (供儀表板使用)

限制 (請在報告中說明): 全程固定對水速度、海況取名目時間、海況來自預報 (有誤差)。
"""
import dataclasses
import os
import sys

import numpy as np
from matplotlib.path import Path
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline as BL            # noqa: E402
import bathymetry as BY          # noqa: E402
import fuel_model as FM          # noqa: E402
import planner as PL             # noqa: E402
from analyze_speed import BBOX, GRID_RES   # noqa: E402

CLASS_LABELS = {
    "fish_s": "小型漁船 <24m", "fish_m": "中型漁船 24-50m", "fish_l": "大型漁船 >50m",
    "cargo_s": "近海貨輪 <100m", "cargo_m": "中型貨輪 100-200m", "cargo_l": "大型貨輪 200-300m",
    "cargo_xl": "超大貨櫃船 >300m", "tank_s": "小型油輪 <100m", "tank_m": "中大型油輪 100-300m",
    "tank_xl": "超大型油輪 >300m", "pax_conv": "客船(常規)", "pax_fast": "客船(高速)", "tug": "拖船/工作船",
}
# 吃水會隨裝載明顯變動的船級 (依船舶參數庫的吃水申報分析: 貨輪、油輪)
DRAFT_VARIES = {"cargo_m", "cargo_l", "cargo_xl", "tank_m", "tank_xl", "cargo_s", "tank_s"}


def grid_axes():
    lons = np.arange(BBOX["lon0"], BBOX["lon1"] + GRID_RES / 2, GRID_RES)
    lats = np.arange(BBOX["lat0"], BBOX["lat1"] + GRID_RES / 2, GRID_RES)
    return lons, lats


def polygons_mask(lons, lats, polys):
    """polys: [[(lat, lon), ...], ...] -> 格點是否落在任一多邊形內 (bool, shape=(len(lats), len(lons)))。"""
    LON, LAT = np.meshgrid(lons, lats)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    m = np.zeros(len(pts), dtype=bool)
    for z in polys:
        xy = [(p[1], p[0]) for p in z]
        m |= Path(xy).contains_points(pts)
    return m.reshape(LON.shape)


# ------------------------------------------------------------------
# 船
# ------------------------------------------------------------------
def load_profiles(csv_path):
    """讀 ship_profiles_v2.csv -> {class_key: dict(loa, beam, d10, d50, d90, vs, cb)}。缺欄位的船級略過。"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    vcol = "service_kn_p50" if "service_kn_p50" in df.columns else "cruise_kn_p50"
    out = {}
    for _, r in df.iterrows():
        try:
            k = r["class_key"]
            out[k] = dict(loa=float(r["loa_p50"]), beam=float(r["beam_p50"]),
                          d10=float(r["draft_p10"]), d50=float(r["draft_p50"]), d90=float(r["draft_p90"]),
                          vs=float(r[vcol]), cb=FM.CB_BY_CLASS.get(k, 0.7))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def make_ship(key, prof, draft_m=None, speed_kn=None):
    """依船級參數與使用者輸入的吃水建立 Ship。回傳 (Ship, 參考航速)。
    與評估一致: 參考航速取 max(服務航速, 目標航速), 只決定主機最大功率上限。"""
    p = prof[key]
    draft = p["d50"] if draft_m is None else float(draft_m)
    vs = p["vs"] if speed_kn is None else max(p["vs"], float(speed_kn))
    return FM.Ship(key, p["loa"], p["beam"], draft, p["cb"], vs)


# ------------------------------------------------------------------
# 航道 (離岸距離 + 禁航區 + 水深)
# ------------------------------------------------------------------
class Router:
    def __init__(self, dist, buffer_km=5.0, blocked=None, depth=None):
        self.lons, self.lats = grid_axes()
        self.C = len(self.lons)
        self.dist, self.buffer, self.blocked, self.depth = dist, buffer_km, blocked, depth
        self._cache = {}

    def _graph(self, req_depth):
        key = None if (self.depth is None or req_depth is None) else int(round(req_depth))
        if key not in self._cache:
            nav = self.dist >= self.buffer
            if self.blocked is not None:
                nav = nav & ~self.blocked
            if key is not None:
                nav = nav & BY.passable(self.depth, float(key))
            comp = BL.largest_component(nav)
            self._cache[key] = (comp, BL.build_graph(comp, self.lats))
        return self._cache[key]

    def baseline(self, start, end, req_depth=None):
        """start, end: (lon, lat)。回傳 dict(nodes, lon, lat, comp, gate_km) 或 None。"""
        comp, G = self._graph(req_depth)
        gs = BL.pick_gate(comp, self.lons, self.lats, start[0], start[1])
        gd = BL.pick_gate(comp, self.lons, self.lats, end[0], end[1])
        s_i, d_i = gs[0] * self.C + gs[1], gd[0] * self.C + gd[1]
        _, pred = dijkstra(G, directed=True, indices=[s_i], return_predecessors=True)
        nodes = BL.trace_path(pred[0], s_i, d_i)
        if nodes is None or len(nodes) < 2:
            return None
        nodes = np.array(nodes)
        return dict(nodes=nodes, lon=self.lons[nodes % self.C], lat=self.lats[nodes // self.C],
                    comp=comp, gate_km=(gs[2], gd[2]))


# ------------------------------------------------------------------
# 沿路徑時間表
# ------------------------------------------------------------------
def timeline(ship, env, lon, lat, t0, T_h, V_kn, step_km=3.0):
    """以對水速度 V_kn 沿路徑前進, 回傳每個節點的累積距離 (km)、累積時間 (h)、累積油耗 (t) 與每段航向。"""
    lo, la = FM.resample_path(lon, lat, step_km)                    # 與 prepare_path 內部切法相同
    prep = FM.prepare_path(env, lon, lat, t0, T_h, step_km=step_km)
    d, trk = prep["d_km"], prep["trk"]
    V = V_kn * FM.KN
    dt = np.zeros(len(d)); fuel = np.zeros(len(d))
    for i in range(len(d)):
        vg, p, _ = FM.segment_physics(ship, V, trk[i], prep["env"][i])
        dt[i] = d[i] * 1000.0 / vg if np.isfinite(vg) and vg > 0.05 else np.inf
        fuel[i] = ship.sfoc / 1000.0 * p * dt[i] / 3600.0 if np.isfinite(dt[i]) else np.inf
    cum_km = np.concatenate([[0.0], np.cumsum(d)])
    cum_h = np.concatenate([[0.0], np.cumsum(dt) / 3600.0])
    cum_t = np.concatenate([[0.0], np.cumsum(fuel) / 1000.0])
    return dict(lon=lo, lat=la, cum_km=cum_km, cum_h=cum_h, cum_fuel_t=cum_t,
                heading=np.concatenate([trk, trk[-1:]]), seg_env=prep["env"])


def _eval(ship, env, lon, lat, t0, T_h, step_km):
    prep = FM.prepare_path(env, lon, lat, t0, T_h, step_km)
    r = FM.fixed_eta_prepared(ship, prep, T_h)
    r["dist_km"] = prep["dist_km"]
    r["missing_frac"] = prep["missing_frac"]
    return r


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def route_plan(router, env, ship, start, end, V_target_kn, t0, corridor_km=60.0, step_km=6.0, req_depth=None):
    """回傳 dict; 失敗時回傳 dict(error=...)。t0: datetime64 出發時間 (UTC)。"""
    base = router.baseline(start, end, req_depth)
    if base is None:
        return dict(error="找不到可通行的基準路徑 (起訖點被離岸距離、禁航區或水深限制隔開)")
    blon = np.concatenate([[start[0]], base["lon"], [end[0]]])
    blat = np.concatenate([[start[1]], base["lat"], [end[1]]])
    L_km = float(FM.segments(blon, blat)[0].sum())
    T_h = L_km * 1000.0 / (V_target_kn * FM.KN) / 3600.0
    rb = _eval(ship, env, blon, blat, t0, T_h, step_km)
    stretched = 1.0
    for _ in range(8):                                   # 目標時間在預報海況下趕不上 -> 放寬時間
        if rb["feasible"]:
            break
        stretched *= 1.1
        rb = _eval(ship, env, blon, blat, t0, T_h * stretched, step_km)
    if not rb["feasible"]:
        return dict(error="在預報海況下, 基準路徑也無法在合理時間內走完, 請降低目標航速或改變出發條件")
    T_h *= stretched
    lam = 2.0 * rb["fuel_t"] * 1000.0 / T_h              # kg/小時: 與評估相同
    pw = PL.plan_path(ship, env, base["comp"], router.lons, router.lats, base["nodes"], t0, T_h,
                      rb["V_kn"], lam, corridor_km)
    plan = None
    if pw is not None:
        plon = np.concatenate([[start[0]], pw["lon"], [end[0]]])
        plat = np.concatenate([[start[1]], pw["lat"], [end[1]]])
        rp = _eval(ship, env, plon, plat, t0, T_h, step_km)
        if rp["feasible"]:
            plan = dict(lon=plon, lat=plat, res=rp)
    use_plan = plan is not None and plan["res"]["fuel_t"] < rb["fuel_t"]
    chosen = plan if use_plan else dict(lon=blon, lat=blat, res=rb)
    tl = timeline(ship, env, chosen["lon"], chosen["lat"], t0, T_h, chosen["res"]["V_kn"])
    return dict(
        error=None, T_h=T_h, stretched=stretched, used_plan=use_plan,
        base=dict(lon=blon, lat=blat, res=rb), plan=plan, chosen=chosen, timeline=tl,
        saving_pct=100.0 * (1 - chosen["res"]["fuel_t"] / rb["fuel_t"]), gate_km=base["gate_km"],
    )

