"""
planner.py -- 依海況規劃的路徑 (HELIOS 的核心想法, 用物理油耗模型重做)

做法 (白話):
  1. 在「基準路徑」(不看海況的最短海上路徑) 周圍 corridor_km 公里的範圍內切成格點。
  2. 每兩個相鄰格點之間的「路」, 都用 fuel_model 的物理公式, 依該位置、該時刻的
     海流、風、浪, 算出走這一段要燒多少油 (有方向性: 順流便宜、逆流貴)。
  3. 用最短路徑演算法 (Dijkstra) 找出「成本最小」的路線, 起訖點與基準路徑相同。
  4. 成本 = 油耗 + lam * 時間。lam 是「每多花 1 小時算多少公斤油」, 用來顧及準時到達:
     固定到達時間下油耗 ~ 1/T^2, 所以 lam ~ 2 * 基準油耗 / T (kg/小時)。

簡化與限制 (請在報告中說明):
  - 時間: 每個格點的海況取「名目時間」(出發時間 + T * 該點在基準路徑上的進度), 不做完整的時間相依搜尋。
    海況資料的時間解析度是 3~6 小時, 與這個近似的誤差同量級。
  - 速度: 全程固定對水速度 V (取基準路徑準時抵達所需的速度)。
  - 這裡用的是「事後的真實海況」(ERA5 事後再分析 + HYCOM 預報), 等於預知天氣, 是理論上限。
    真正的預報有誤差, 規劃出的路徑效益會更低。

用法:
    python planner.py --selftest
    (實際使用由 evaluate_routes.py --plan 呼叫)
"""
import argparse
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import fuel_model as FM

GRID_RES = 0.01
KM_PER_DEG = 111.2
# 前向的 8 種位移 (列, 欄) 與需要檢查的中間格 (避免斜穿陸地角落); 反向邊另外補
INTER = {
    (0, 1): [], (1, 0): [],
    (1, 1): [(0, 1), (1, 0)], (1, -1): [(0, -1), (1, 0)],
    (1, 2): [(0, 1), (1, 1)], (2, 1): [(1, 0), (1, 1)],
    (1, -2): [(0, -1), (1, -1)], (2, -1): [(1, 0), (1, -1)],
}


# ------------------------------------------------------------------
# 向量化的物理 (與 fuel_model.segment_physics 一致, 供大量邊同時計算)
# ------------------------------------------------------------------
def power_vec(ship, V, head_deg, u_cur, v_cur, u10, v10, hs, wave_from_deg):
    th = np.radians(head_deg)
    s, c = np.sin(th), np.cos(th)
    c_par = u_cur * s + v_cur * c
    c_perp = u_cur * c - v_cur * s
    inside = V * V - c_perp ** 2
    ok = inside > 0
    vg = c_par + np.sqrt(np.where(ok, inside, 1.0))
    r_cw = ship.rs_n * (V / (ship.vs_kn * FM.KN)) ** 2
    rel = (u10 * s + v10 * c) - vg
    k = 0.5 * FM.RHO_A * ship.cx * ship.af_m2
    r_added_wind = -k * rel * np.abs(rel) - k * vg * vg
    f_wave = ((1 + np.cos(np.radians(wave_from_deg - head_deg))) / 2) ** 2
    r_aw = ship.k_aw * FM.RHO_W * FM.G * (hs * hs / 8.0) * (ship.beam ** 2 / ship.loa) * f_wave
    r_tot = np.maximum(r_cw + r_added_wind + r_aw, 0.05 * r_cw)
    return vg, r_tot * V / ship.eta / 1000.0, ok


def power_capped(ship, V0, head_deg, u_cur, v_cur, u10, v10, hs, wave_from_deg):
    """與 segment_physics 相同: 主機頂不住時被迫減速。回傳 (對地速度, 功率 kW, 實際對水速度, 有效)。"""
    shape = np.shape(head_deg)
    V = np.full(shape, float(V0))
    vg, P, ok = power_vec(ship, V, head_deg, u_cur, v_cur, u10, v10, hs, wave_from_deg)
    pmax = ship.p_max_ratio * ship.ps_kw
    over = ok & (P > pmax)
    if over.any():
        lo, hi = np.full(shape, 0.2 * V0), np.full(shape, float(V0))
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            _, pm, okm = power_vec(ship, mid, head_deg, u_cur, v_cur, u10, v10, hs, wave_from_deg)
            gt = okm & (pm > pmax)
            hi = np.where(over & gt, mid, hi)
            lo = np.where(over & ~gt, mid, lo)
        V = np.where(over, lo, V)
        vg, P, ok = power_vec(ship, V, head_deg, u_cur, v_cur, u10, v10, hs, wave_from_deg)
    return vg, P, V, ok


def edge_cost(ship, V0, head_deg, len_km, ev, lam):
    """每條邊的成本 = 油耗(kg) + lam * 時間(小時)。無效邊 (側流大於船速等) 給極大值。"""
    wf = np.degrees(np.arctan2(ev["dir_sin"], ev["dir_cos"]))
    if not ship.wave_dir_is_from:
        wf = wf + 180.0
    vg, P, V, ok = power_capped(ship, V0, head_deg, ev["u_cur"], ev["v_cur"], ev["u10"], ev["v10"],
                                ev["swh"], wf)
    good = ok & (vg > 0.05)
    dt_h = len_km * 1000.0 / np.where(good, vg, 1.0) / 3600.0
    cost = ship.sfoc / 1000.0 * P * dt_h + lam * dt_h
    return np.where(good, cost, 1e9)


# ------------------------------------------------------------------
# 規劃
# ------------------------------------------------------------------
def _node_env(env, t0, T_h, frac, lat, lon):
    n = len(lat)
    zero = np.zeros(n)
    out = {"u_cur": zero.copy(), "v_cur": zero.copy(), "u10": zero.copy(), "v10": zero.copy(),
           "swh": zero.copy(), "dir_sin": zero.copy(), "dir_cos": np.ones(n)}
    if env is None:
        return out
    tt = np.datetime64(t0, "s") + np.round(frac * T_h * 3600.0).astype("int64").astype("timedelta64[s]")
    ev = env.at(tt, lat, lon)
    for k in ("u_cur", "v_cur", "u10", "v10", "swh"):
        if k in ev:
            out[k] = np.nan_to_num(np.asarray(ev[k], dtype=float), nan=0.0)
    if "mwd" in ev:
        m = np.asarray(ev["mwd"], dtype=float)
        ok = np.isfinite(m)
        out["dir_sin"] = np.where(ok, np.sin(np.radians(np.where(ok, m, 0.0))), 0.0)
        out["dir_cos"] = np.where(ok, np.cos(np.radians(np.where(ok, m, 0.0))), 1.0)
    return out


def plan_path(ship, env, nav, lons, lats, base_nodes, t0, T_h, V_kn, lam, corridor_km=60.0):
    """回傳 dict(lon, lat, cost) 或 None (起訖點在走廊內不連通)。
    nav: 全網格的可通行遮罩 (bool, 與 baseline 使用的相同); base_nodes: 基準路徑的節點 (r*C + c)。"""
    R, C = nav.shape
    base_nodes = np.asarray(base_nodes)
    br, bc = base_nodes // C, base_nodes % C
    mr = int(np.ceil(corridor_km / (GRID_RES * KM_PER_DEG))) + 1
    mc = int(np.ceil(corridor_km / (GRID_RES * KM_PER_DEG * np.cos(np.radians(np.mean(lats[br])))))) + 1
    r0, r1 = max(br.min() - mr, 0), min(br.max() + mr + 1, R)
    c0, c1 = max(bc.min() - mc, 0), min(bc.max() + mc + 1, C)
    nr, nc = r1 - r0, c1 - c0
    slats, slons = lats[r0:r1], lons[c0:c1]
    LAT, LON = np.meshgrid(slats, slons, indexing="ij")

    # 走廊遮罩: 離基準路徑 <= corridor_km; 同時算每個格點在基準路徑上的「進度」(用來決定名目時間)
    lat0 = float(np.mean(lats[br]))
    kx = KM_PER_DEG * np.cos(np.radians(lat0))
    bxy = np.column_stack([lons[bc] * kx, lats[br] * KM_PER_DEG])
    seg = np.hypot(np.diff(bxy[:, 0]), np.diff(bxy[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    tot = max(cum[-1], 1e-9)
    dist_km, idx = cKDTree(bxy).query(np.column_stack([LON.ravel() * kx, LAT.ravel() * KM_PER_DEG]))
    frac = (cum[idx] / tot).reshape(nr, nc)
    mask = nav[r0:r1, c0:c1] & (dist_km.reshape(nr, nc) <= corridor_km)

    ev = _node_env(env, t0, T_h, frac.ravel(), LAT.ravel(), LON.ravel())
    flat = {k: v.ravel() for k, v in ev.items()}
    V0 = V_kn * FM.KN
    idm = np.arange(nr * nc).reshape(nr, nc)
    S, D, W = [], [], []
    for (dr, dc), mids in INTER.items():
        cs = 0 if dc >= 0 else -dc
        ncol = nc - abs(dc)
        nrow = nr - dr
        if nrow <= 0 or ncol <= 0:
            continue
        ok = mask[0:nrow, cs:cs + ncol] & mask[dr:nr, cs + dc:cs + dc + ncol]
        for rr, cc in mids:
            ok &= mask[rr:rr + nrow, cs + cc:cs + cc + ncol]
        src = idm[0:nrow, cs:cs + ncol][ok]
        dst = idm[dr:nr, cs + dc:cs + dc + ncol][ok]
        if len(src) == 0:
            continue
        latm = slats[0:nrow][:, None] + dr * GRID_RES / 2
        dy = dr * GRID_RES * KM_PER_DEG
        dx = dc * GRID_RES * KM_PER_DEG * np.cos(np.radians(latm))
        len_km = np.broadcast_to(np.hypot(dy, dx), (nrow, ncol))[ok]
        head = np.broadcast_to((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0, (nrow, ncol))[ok]
        mid = {k: 0.5 * (v[src] + v[dst]) for k, v in flat.items()}
        S.append(src); D.append(dst)
        W.append((edge_cost(ship, V0, head, len_km, mid, lam), edge_cost(ship, V0, (head + 180.0) % 360.0, len_km, mid, lam)))
    if not S:
        return None
    S, D = np.concatenate(S), np.concatenate(D)
    wf = np.concatenate([w[0] for w in W])
    wr = np.concatenate([w[1] for w in W])
    G = csr_matrix((np.concatenate([wf, wr]), (np.concatenate([S, D]), np.concatenate([D, S]))),
                   shape=(nr * nc, nr * nc))

    s_node = (br[0] - r0) * nc + (bc[0] - c0)
    d_node = (br[-1] - r0) * nc + (bc[-1] - c0)
    if not (mask.ravel()[s_node] and mask.ravel()[d_node]):
        return None
    dist, pred = dijkstra(G, directed=True, indices=s_node, return_predecessors=True)
    if not np.isfinite(dist[d_node]) or dist[d_node] >= 1e8:
        return None
    path, cur = [d_node], d_node
    while cur != s_node:
        cur = pred[cur]
        if cur < 0:
            return None
        path.append(cur)
    path = np.array(path[::-1])
    return dict(lon=slons[path % nc], lat=slats[path // nc], cost=float(dist[d_node]))


# ------------------------------------------------------------------
# 自我測試
# ------------------------------------------------------------------
class _SpatialEnv:
    """測試用海況: 可依位置給定海流, 其餘為零。"""

    def __init__(self, cur_fn=None, wind=(0.0, 0.0), hs_fn=None, mwd=0.0):
        self.cur_fn, self.wind, self.hs_fn, self.mwd = cur_fn, wind, hs_fn, mwd

    def at(self, t, lat, lon):
        lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
        n = len(lat)
        u, v = self.cur_fn(lat, lon) if self.cur_fn else (np.zeros(n), np.zeros(n))
        hs = self.hs_fn(lat, lon) if self.hs_fn else np.zeros(n)
        return {"u_cur": u, "v_cur": v, "u10": np.full(n, self.wind[0]), "v10": np.full(n, self.wind[1]),
                "swh": hs, "mwd": np.full(n, self.mwd)}


def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'通過' if cond else '失敗'}] {name} {extra}")

    rng = np.random.default_rng(0)
    ship = FM.Ship("測試貨輪", 150, 25, 8, 0.72, 12.0)

    print("1. 向量化物理與 fuel_model.segment_physics 一致 (含主機頂不住的情況)")
    n = 400
    head = rng.uniform(0, 360, n)
    ev = dict(u_cur=rng.normal(0, .6, n), v_cur=rng.normal(0, .6, n), u10=rng.normal(0, 7, n),
              v10=rng.normal(0, 7, n), swh=rng.uniform(0, 6, n), mwd=rng.uniform(0, 360, n))
    V0 = 6.2
    vg, P, V, okv = power_capped(ship, V0, head, ev["u_cur"], ev["v_cur"], ev["u10"], ev["v10"], ev["swh"], ev["mwd"])
    maxerr_p = maxerr_v = 0.0
    for i in range(n):
        e = {k: float(v[i]) for k, v in ev.items()}
        vg_s, p_s, v_s = FM.segment_physics(ship, V0, head[i], e)
        maxerr_p = max(maxerr_p, abs(p_s - P[i]) / p_s)
        maxerr_v = max(maxerr_v, abs(vg_s - vg[i]))
    check("功率一致", maxerr_p < 1e-3, f"(最大相對差 {maxerr_p:.2e})")
    check("對地速度一致", maxerr_v < 1e-3, f"(最大差 {maxerr_v:.2e} m/s)")
    check("測試涵蓋被迫減速的情況", (V < V0 - 1e-6).any(), f"({(V < V0 - 1e-6).mean():.0%} 的邊被迫減速)")

    print("2. 無海況: 規劃出的路徑長度 ~ 最短路徑")
    R, C = 260, 120
    lats = 23.0 + GRID_RES * np.arange(R)
    lons = 120.0 + GRID_RES * np.arange(C)
    nav = np.ones((R, C), dtype=bool)
    base = np.array([r * C + 20 for r in range(10, 230)])           # 沿第 20 欄正北走
    t0 = "2023-03-01T00:00:00"
    L_base = (230 - 10 - 1) * GRID_RES * KM_PER_DEG
    T_h = L_base * 1000 / (12 * FM.KN) / 3600
    p0 = plan_path(ship, None, nav, lons, lats, base, t0, T_h, 12.0, lam=100.0)
    d0, _ = FM.segments(p0["lon"], p0["lat"])
    check("起點終點與基準相同", (p0["lat"][0] == lats[10]) and (p0["lat"][-1] == lats[229]))
    check("長度與最短路徑相差 < 1.5%", abs(d0.sum() / L_base - 1) < 0.015, f"({d0.sum():.1f} vs {L_base:.1f} km)")

    print("3. 有利的順流帶: 規劃器應該繞去順流區, 且在固定到達時間下確實更省油")
    cur_fn = lambda la, lo: (np.zeros(len(la)), np.where(lo > 120.45, 1.4, 0.0))   # 東側 (>120.45E) 有 1.4 m/s 北向流
    env = _SpatialEnv(cur_fn=cur_fn)
    v_base = FM.fixed_eta_prepared(ship, FM.prepare_path(env, lons[base % C], lats[base // C], t0, T_h), T_h)
    f_base = v_base["fuel_t"] * 1000
    lam = 2 * f_base / T_h
    pw = plan_path(ship, env, nav, lons, lats, base, t0, T_h, v_base["V_kn"], lam, corridor_km=60)
    f_plan = FM.fixed_eta_prepared(ship, FM.prepare_path(env, pw["lon"], pw["lat"], t0, T_h), T_h)
    check("路徑繞往東側順流區", pw["lon"].max() > 120.45, f"(最東 {pw['lon'].max():.3f}E)")
    check("油耗比直走基準路徑少", f_plan["fuel_t"] < v_base["fuel_t"] * 0.97,
          f"(基準 {v_base['fuel_t']:.2f} t, 規劃 {f_plan['fuel_t']:.2f} t, 省 {100 * (1 - f_plan['fuel_t'] / v_base['fuel_t']):.1f}%)")

    print("4. 逆流帶在路徑上: 規劃器應該避開")
    cur_fn2 = lambda la, lo: (np.zeros(len(la)), np.where((np.abs(lo - 120.2) < 0.06), -1.4, 0.0))   # 基準路徑 (120.2E) 上是逆流
    env2 = _SpatialEnv(cur_fn=cur_fn2)
    vb2 = FM.fixed_eta_prepared(ship, FM.prepare_path(env2, lons[base % C], lats[base // C], t0, T_h), T_h)
    pw2 = plan_path(ship, env2, nav, lons, lats, base, t0, T_h, vb2["V_kn"], 2 * vb2["fuel_t"] * 1000 / T_h)
    f2 = FM.fixed_eta_prepared(ship, FM.prepare_path(env2, pw2["lon"], pw2["lat"], t0, T_h), T_h)
    check("避開逆流帶後更省油", f2["fuel_t"] < vb2["fuel_t"] * 0.97,
          f"(基準 {vb2['fuel_t']:.2f} t, 規劃 {f2['fuel_t']:.2f} t)")

    print("5. 不可通行的格點絕不穿越")
    nav5 = nav.copy()
    nav5[100:140, 10:40] = False                                       # 基準路徑上放一塊障礙 (島)
    nav5[100:140, 20] = False
    base5 = np.array([r * C + 20 for r in range(10, 100)] + [99 * C + 21 + k for k in range(0, 1)])  # 只取到障礙前
    # 用「繞障礙」的基準: 先走到 r=99, 繞 (r=99..141, c=41) 再回到 c=20
    path_nodes = [r * C + 20 for r in range(10, 99)] + [99 * C + c for c in range(21, 42)] + \
                 [r * C + 41 for r in range(100, 141)] + [141 * C + c for c in range(40, 19, -1)] + \
                 [r * C + 20 for r in range(142, 230)]
    p5 = plan_path(ship, None, nav5, lons, lats, np.array(path_nodes), t0, T_h, 12.0, lam=100.0)
    rr = np.rint((p5["lat"] - lats[0]) / GRID_RES).astype(int)
    cc = np.rint((p5["lon"] - lons[0]) / GRID_RES).astype(int)
    check("所有節點都在可通行區", bool(nav5[rr, cc].all()))

    print("6. 起點或終點不通時回傳 None")
    nav6 = nav.copy()
    nav6[10, 20] = False
    check("起點被封鎖 -> None", plan_path(ship, None, nav6, lons, lats, base, t0, T_h, 12.0, 100.0) is None)

    print("\n全部通過" if ok else "\n有項目失敗, 請檢查")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    print("用法: python planner.py --selftest   (實際使用由 evaluate_routes.py --plan 呼叫)")
