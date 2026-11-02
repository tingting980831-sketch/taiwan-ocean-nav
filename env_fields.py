"""
env_fields.py -- 給「時間 + 位置」, 查出海流、風、浪

資料來源 (全部 UTC):
  風  : ERA5   era5_2023_MM.nc          u10, v10                    0.25 度, 6 小時
  浪  : ERA5   wave_2023_MM.nc          swh, shww, shts, mwd, mwp, pp1d   0.5 度, 6 小時
  海流: HYCOM  hycom_glby_930_<起報YYYYMMDDHH>_t<時效>_uv3z_subscene.nc   water_u, water_v, 3 小時
        (檔案內的時間單位是 "hours since analysis", 無法標準解碼 ->
         有效時間改由檔名算: 起報時間 + 時效; 每個時間點只有一個檔案)

做的事:
  - 空間: 雙線性內插 (遇到缺值格子, 只用有值的格子並重新加權)
  - 時間: 前後兩個時間點之間線性內插; 資料空窗 (> 2 倍間隔) 回傳 NaN, 不亂補
  - 陸地/沿岸缺值: 只在「最近有值格子距離 <= fill_km」時補值 (浪 80 km, 海流 25 km)
  - 浪向: 用 sin/cos 內插 (350 度與 10 度的平均是 0 度, 不是 180 度)

用法 (程式中):
    from env_fields import Environment
    env = Environment(wind_dir=r"C:\\Users\\tingt\\ERA5", wave_dir=r"C:\\Users\\tingt\\ERA5_WAVE_2023",
                      hycom_dir=r"C:\\NODASS\\HYCOM\\2023")
    out = env.at(t, lat, lon)      # t: numpy datetime64 (UTC); lat, lon: 陣列
    out["u_cur"], out["v_cur"], out["u10"], out["v10"], out["swh"], out["mwd"] ...

命令列:
    python env_fields.py --selftest          # 用模擬資料檢查內插邏輯 (不需要真實資料)
    python env_fields.py --check --wind-dir ... --wave-dir ... --hycom-dir ... \\
                         --time 2023-03-15T06:00 --lat 24.0 --lon 120.0
    python env_fields.py --coverage --wind-dir ... --wave-dir ... --hycom-dir ...
"""
import argparse
import glob
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
from scipy.ndimage import distance_transform_edt

KM_PER_DEG = 111.2


# ------------------------------------------------------------------
# 空間內插與補值 (純 numpy, 可獨立測試)
# ------------------------------------------------------------------
def bilinear(arr, lats, lons, plat, plon, min_weight=0.25):
    """arr (Y,X); lats/lons 嚴格遞增; 回傳 (n,)。
    缺值格子不參與, 其餘格子重新加權; 有效權重不足 min_weight 或超出範圍 -> NaN。"""
    plat = np.atleast_1d(np.asarray(plat, dtype=float))
    plon = np.atleast_1d(np.asarray(plon, dtype=float))
    ny, nx = arr.shape
    inside = (plat >= lats[0]) & (plat <= lats[-1]) & (plon >= lons[0]) & (plon <= lons[-1])
    iy = np.clip(np.searchsorted(lats, plat) - 1, 0, ny - 2)
    ix = np.clip(np.searchsorted(lons, plon) - 1, 0, nx - 2)
    ty = np.clip((plat - lats[iy]) / (lats[iy + 1] - lats[iy]), 0, 1)
    tx = np.clip((plon - lons[ix]) / (lons[ix + 1] - lons[ix]), 0, 1)
    num = np.zeros(plat.shape)
    den = np.zeros(plat.shape)
    for dy, dx, w in ((0, 0, (1 - ty) * (1 - tx)), (0, 1, (1 - ty) * tx),
                      (1, 0, ty * (1 - tx)), (1, 1, ty * tx)):
        v = arr[iy + dy, ix + dx].astype(float)
        ok = np.isfinite(v)
        num += np.where(ok, v * w, 0.0)
        den += np.where(ok, w, 0.0)
    good = (den >= min_weight) & inside
    return np.where(good, num / np.where(den > 0, den, 1.0), np.nan)


def fill_nan(arr, lats, lons, max_km):
    """缺值格子 -> 最近有值格子的值, 但只補距離 <= max_km 的 (陸地深處維持缺值)。"""
    bad = ~np.isfinite(arr)
    if max_km <= 0 or not bad.any() or bad.all():
        return arr
    dy = abs(float(np.median(np.diff(lats)))) * KM_PER_DEG
    dx = abs(float(np.median(np.diff(lons)))) * KM_PER_DEG * np.cos(np.radians(float(np.mean(lats))))
    dist, (iy, ix) = distance_transform_edt(bad, sampling=(dy, dx), return_indices=True)
    out = arr.copy()
    m = bad & (dist <= max_km)
    out[m] = arr[iy[m], ix[m]]
    return out


# ------------------------------------------------------------------
# 時間內插
# ------------------------------------------------------------------
class LazyField:
    """一組變數的時間序列; 切片要用時才載入, 並以 LRU 快取。

    store 需提供:
        store.times            : 已排序的 datetime64[s] 陣列
        store.load(i)          : -> (lats, lons, {變數名: 2D 陣列}), lats/lons 遞增
    """

    def __init__(self, store, max_gap_h, cache_size=24):
        self.store = store
        self.times = np.asarray(store.times).astype("datetime64[s]")
        self.max_gap_s = max_gap_h * 3600.0
        self._cache = OrderedDict()
        self._cache_size = cache_size

    def _get(self, i):
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        v = self.store.load(i)
        self._cache[i] = v
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return v

    def sample(self, names, t, plat, plon):
        plat = np.atleast_1d(np.asarray(plat, dtype=float))
        plon = np.atleast_1d(np.asarray(plon, dtype=float))
        n = len(plat)
        tt = np.broadcast_to(np.asarray(t, dtype="datetime64[s]"), plat.shape).astype("int64")
        ts = self.times.astype("int64")
        out = {nm: np.full(n, np.nan) for nm in names}
        if len(ts) == 0:
            return out
        if len(ts) == 1:
            idx = np.nonzero(tt == ts[0])[0]
            if len(idx):
                lats, lons, arrs = self._get(0)
                for nm in names:
                    out[nm][idx] = bilinear(arrs[nm], lats, lons, plat[idx], plon[idx])
            return out
        in_range = (tt >= ts[0]) & (tt <= ts[-1])
        i1 = np.clip(np.searchsorted(ts, tt, side="left"), 1, len(ts) - 1)
        i0 = i1 - 1
        span = (ts[i1] - ts[i0]).astype(float)
        w = np.where(span > 0, (tt - ts[i0]) / np.where(span > 0, span, 1.0), 0.0)
        ok = in_range & (span <= self.max_gap_s)
        for k in np.unique(i0[ok]):
            sel = np.nonzero(ok & (i0 == k))[0]
            lats, lons, a0 = self._get(int(k))
            _, _, a1 = self._get(int(k) + 1)
            ww = w[sel]
            for nm in names:
                v0 = bilinear(a0[nm], lats, lons, plat[sel], plon[sel])
                v1 = bilinear(a1[nm], lats, lons, plat[sel], plon[sel])
                out[nm][sel] = np.where(ww <= 0, v0, np.where(ww >= 1, v1, (1 - ww) * v0 + ww * v1))
        return out


# ------------------------------------------------------------------
# 資料來源 (需要 xarray)
# ------------------------------------------------------------------
def _ascending(lats, lons, arrs):
    """確保緯度、經度遞增 (ERA5 的緯度通常是由北到南)。"""
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        arrs = {k: v[::-1, :] for k, v in arrs.items()}
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        arrs = {k: v[:, ::-1] for k, v in arrs.items()}
    return lats, lons, arrs


class Era5Store:
    """ERA5 月檔 (valid_time, latitude, longitude)。整月讀進記憶體, 保留最近 3 個月。"""

    def __init__(self, files, variables, fill_km=0.0, derive_dir=None):
        import xarray as xr
        self.files = sorted(files)
        if not self.files:
            raise FileNotFoundError("找不到 ERA5 檔案")
        self.variables = variables
        self.fill_km = fill_km
        self.derive_dir = derive_dir
        tl, self.index = [], []
        for fi, f in enumerate(self.files):
            with xr.open_dataset(f) as ds:
                tname = "valid_time" if "valid_time" in ds.coords else "time"
                t = ds[tname].values.astype("datetime64[s]")
            for li, tv in enumerate(t):
                tl.append(tv)
                self.index.append((fi, li))
        order = np.argsort(np.array(tl))
        self.times = np.array(tl)[order]
        self.index = [self.index[j] for j in order]
        self._month = OrderedDict()

    def _load_month(self, fi):
        if fi in self._month:
            self._month.move_to_end(fi)
            return self._month[fi]
        import xarray as xr
        with xr.open_dataset(self.files[fi]) as ds:
            latn = "latitude" if "latitude" in ds.coords else "lat"
            lonn = "longitude" if "longitude" in ds.coords else "lon"
            lats = ds[latn].values.astype(float)
            lons = ds[lonn].values.astype(float)
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            data = {v: ds[v].transpose(tname, latn, lonn).values.astype("float32")
                    for v in self.variables}
        self._month[fi] = (lats, lons, data)
        if len(self._month) > 3:
            self._month.popitem(last=False)
        return self._month[fi]

    def load(self, i):
        fi, li = self.index[i]
        lats, lons, data = self._load_month(fi)
        arrs = {v: data[v][li] for v in self.variables}
        lats, lons, arrs = _ascending(lats, lons, arrs)
        arrs = {k: fill_nan(v, lats, lons, self.fill_km) for k, v in arrs.items()}
        if self.derive_dir and self.derive_dir in arrs:
            r = np.radians(arrs[self.derive_dir])
            arrs["dir_sin"], arrs["dir_cos"] = np.sin(r), np.cos(r)
        return lats, lons, arrs


HYCOM_PAT = re.compile(r"hycom_glby_930_(\d{10})_t(\d{3})_uv3z_subscene\.nc$", re.I)


class HycomStore:
    """HYCOM uv3z 檔。有效時間 = 檔名的起報時間 + 時效; 每個時間點挑時效最短的檔案。
    檔內時間單位 'hours since analysis' 不解碼 (decode_times=False)。"""

    def __init__(self, root, fill_km=25.0, depth_m=0.0):
        self.fill_km = fill_km
        self.depth_m = depth_m
        best = {}
        for dp, _, fns in os.walk(root):
            for fn in fns:
                m = HYCOM_PAT.match(fn)
                if not m:
                    continue
                init = datetime.strptime(m.group(1), "%Y%m%d%H")
                lead = int(m.group(2))
                valid = init + timedelta(hours=lead)
                if valid not in best or lead < best[valid][0]:
                    best[valid] = (lead, os.path.join(dp, fn))
        if not best:
            raise FileNotFoundError(f"在 {root} 找不到 HYCOM uv3z 檔案")
        keys = sorted(best)
        self.times = np.array(keys, dtype="datetime64[s]")
        self.paths = [best[k][1] for k in keys]
        self.leads = [best[k][0] for k in keys]
        self._depth_idx = None

    def load(self, i):
        import xarray as xr
        with xr.open_dataset(self.paths[i], decode_times=False) as ds:
            if self._depth_idx is None:
                self._depth_idx = int(np.abs(ds["depth"].values - self.depth_m).argmin())
            lats = ds["lat"].values.astype(float)
            lons = ds["lon"].values.astype(float)
            lons = np.where(lons > 180, lons - 360, lons)
            u = ds["water_u"].isel(time=0, depth=self._depth_idx).values.astype("float32")
            v = ds["water_v"].isel(time=0, depth=self._depth_idx).values.astype("float32")
        bad = (np.abs(u) > 10.0) | (np.abs(v) > 10.0)          # 海流不可能超過 10 m/s -> 視為缺值
        u = np.where(bad, np.nan, u)
        v = np.where(bad, np.nan, v)
        arrs = {"u_cur": u, "v_cur": v}
        lats, lons, arrs = _ascending(lats, lons, arrs)
        arrs = {k: fill_nan(a, lats, lons, self.fill_km) for k, a in arrs.items()}
        return lats, lons, arrs


# ------------------------------------------------------------------
# 組合
# ------------------------------------------------------------------
class Environment:
    def __init__(self, wind_dir=None, wave_dir=None, hycom_dir=None,
                 fill_km_wave=80.0, fill_km_cur=25.0, depth_m=0.0):
        self.parts = {}
        if wind_dir:
            st = Era5Store(glob.glob(os.path.join(wind_dir, "era5_*.nc")), ["u10", "v10"])
            self.parts["wind"] = (LazyField(st, max_gap_h=12), ["u10", "v10"])
        if wave_dir:
            vs = ["swh", "shww", "shts", "mwp", "pp1d", "mwd"]
            st = Era5Store(glob.glob(os.path.join(wave_dir, "wave_*.nc")), vs,
                           fill_km=fill_km_wave, derive_dir="mwd")
            self.parts["wave"] = (LazyField(st, max_gap_h=12), ["swh", "shww", "shts", "mwp", "pp1d",
                                                                 "dir_sin", "dir_cos"])
        if hycom_dir:
            st = HycomStore(hycom_dir, fill_km=fill_km_cur, depth_m=depth_m)
            self.parts["current"] = (LazyField(st, max_gap_h=6), ["u_cur", "v_cur"])

    def at(self, t, lat, lon):
        """t: datetime64 (UTC) 純量或陣列; lat, lon: 陣列。回傳 dict (缺資料為 NaN)。"""
        out = {}
        for _, (field, names) in self.parts.items():
            out.update(field.sample(names, t, lat, lon))
        if "dir_sin" in out:
            ang = (np.degrees(np.arctan2(out.pop("dir_sin"), out.pop("dir_cos"))) + 360.0) % 360.0
            out["mwd"] = ang
        if "u10" in out:
            out["wind_speed"] = np.hypot(out["u10"], out["v10"])
        if "u_cur" in out:
            out["cur_speed"] = np.hypot(out["u_cur"], out["v_cur"])
        return out


# ------------------------------------------------------------------
# 自我測試 (模擬資料, 不需要 xarray 與真實檔案)
# ------------------------------------------------------------------
class _FakeStore:
    def __init__(self, times, builder):
        self.times = np.array(times, dtype="datetime64[s]")
        self.builder = builder
        self.loads = 0

    def load(self, i):
        self.loads += 1
        return self.builder(i)


def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'通過' if cond else '失敗'}] {name} {extra}")

    lats = np.arange(21, 26.001, 0.5)
    lons = np.arange(118, 124.001, 0.5)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    print("1. 空間雙線性: 線性場 f = 2*lat + 3*lon 應該完全準確")
    f = 2 * LAT + 3 * LON
    pl, po = np.array([22.13, 24.77]), np.array([119.41, 122.88])
    got = bilinear(f, lats, lons, pl, po)
    check("線性場內插", np.allclose(got, 2 * pl + 3 * po, atol=1e-9))
    check("超出範圍回傳 NaN", np.isnan(bilinear(f, lats, lons, [30.0], [120.0])[0]))

    print("2. 缺值處理")
    g = f.copy()
    g[2, 2] = np.nan
    v = bilinear(g, lats, lons, [22.1], [119.1])          # 角落之一缺值, 仍可重新加權
    check("一個角落缺值時用其餘三個", np.isfinite(v[0]))
    g2 = f.copy()
    g2[0:3, 0:3] = np.nan
    check("四個角落都缺值 -> NaN", np.isnan(bilinear(g2, lats, lons, [21.7], [118.7])[0]))
    filled = fill_nan(g2, lats, lons, max_km=80)
    check("fill_nan 補緊鄰有值格子的缺值 (距離 55 km <= 80 km)", np.isfinite(filled[2, 2]))
    check("fill_nan 不補離有值格子 111 km 以上的缺值", np.isnan(filled[0, 0]))
    far = np.full((6, 6), np.nan)
    far[0, 0] = 1.0
    lt, ln = np.arange(6) * 0.5, np.arange(6) * 0.5
    ff = fill_nan(far, lt, ln, max_km=60)
    check("fill_nan 不補太遠的格子", np.isnan(ff[5, 5]) and np.isfinite(ff[0, 1]))

    print("3. 時間內插 (切片值 = 小時數, 中間應線性)")
    t0 = np.datetime64("2023-03-01T00:00:00")
    times = [t0 + np.timedelta64(6 * k, "h") for k in range(5)]
    store = _FakeStore(times, lambda i: (lats, lons, {"a": np.full(f.shape, 6.0 * i)}))
    fld = LazyField(store, max_gap_h=12)
    r = fld.sample(["a"], np.datetime64("2023-03-01T09:00:00"), [23.0], [120.0])["a"][0]
    check("09:00 介於 06:00(6) 與 12:00(12) -> 9", abs(r - 9.0) < 1e-9, f"(得到 {r})")
    r = fld.sample(["a"], np.datetime64("2023-03-01T12:00:00"), [23.0], [120.0])["a"][0]
    check("剛好整點取該切片值", abs(r - 12.0) < 1e-9)
    r = fld.sample(["a"], np.datetime64("2023-02-28T23:00:00"), [23.0], [120.0])["a"][0]
    check("時間早於資料 -> NaN", np.isnan(r))
    r = fld.sample(["a"], np.datetime64("2023-03-02T01:00:00"), [23.0], [120.0])["a"][0]
    check("時間晚於資料 -> NaN", np.isnan(r))

    print("4. 資料空窗 (缺一個切片, 間隔超過 max_gap) 不可亂補")
    times_gap = [t0, t0 + np.timedelta64(6, "h"), t0 + np.timedelta64(30, "h")]
    sg = _FakeStore(times_gap, lambda i: (lats, lons, {"a": np.full(f.shape, float(i))}))
    fg = LazyField(sg, max_gap_h=12)
    r = fg.sample(["a"], np.datetime64("2023-03-01T18:00:00"), [23.0], [120.0])["a"][0]
    check("空窗中間 -> NaN", np.isnan(r))
    r = fg.sample(["a"], np.datetime64("2023-03-01T03:00:00"), [23.0], [120.0])["a"][0]
    check("空窗之前仍正常", abs(r - 0.5) < 1e-9)

    print("5. 向量化與快取")
    n0 = store.loads
    tt = np.array([np.datetime64("2023-03-01T01:00:00") + np.timedelta64(30 * k, "m") for k in range(10)])
    fld.sample(["a"], tt, np.full(10, 23.0), np.full(10, 120.0))
    check("多點查詢只載入需要的切片", store.loads - n0 <= 3, f"(新增載入 {store.loads - n0} 次)")

    print("6. 浪向 (350 度與 10 度的平均應是 0 度, 不是 180 度)")
    class _Dir:
        times = np.array([t0, t0 + np.timedelta64(6, "h")], dtype="datetime64[s]")

        def load(self, i):
            d = np.full(f.shape, 350.0 if i == 0 else 10.0)
            return lats, lons, {"dir_sin": np.sin(np.radians(d)), "dir_cos": np.cos(np.radians(d))}
    fd = LazyField(_Dir(), max_gap_h=12)
    o = fd.sample(["dir_sin", "dir_cos"], np.datetime64("2023-03-01T03:00:00"), [23.0], [120.0])
    ang = (np.degrees(np.arctan2(o["dir_sin"][0], o["dir_cos"][0])) + 360) % 360
    check("平均浪向", min(ang, 360 - ang) < 1e-6, f"(得到 {ang:.3f} 度)")

    print("\n全部通過" if ok else "\n有項目失敗, 請檢查")
    return 0 if ok else 1


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--wind-dir")
    ap.add_argument("--wave-dir")
    ap.add_argument("--hycom-dir")
    ap.add_argument("--time", default="2023-03-15T06:00")
    ap.add_argument("--lat", type=float, default=24.0)
    ap.add_argument("--lon", type=float, default=120.0)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    env = Environment(a.wind_dir, a.wave_dir, a.hycom_dir)
    if a.check:
        t = np.datetime64(a.time)
        out = env.at(t, [a.lat], [a.lon])
        print(f"時間 {a.time} UTC, 位置 ({a.lat}, {a.lon})")
        for k, v in out.items():
            print(f"  {k:12s} {float(v[0]): .3f}")
        print("\n(參考: 台灣東側黑潮表層流速常見約 0.5~1.5 m/s 向北; 冬季台灣海峽 10 m 風約 5~12 m/s)")
    if a.coverage:
        rng = np.random.default_rng(0)
        pts = [(24.0, 120.0), (23.0, 120.0), (24.8, 121.0), (22.6, 120.2), (24.3, 120.5), (24.0, 122.0)]
        secs = int((np.datetime64("2023-12-31T00:00") - np.datetime64("2023-01-02T00:00")).astype("timedelta64[s]").astype(int))
        ts = np.datetime64("2023-01-02T00:00") + rng.integers(0, secs, 300).astype("timedelta64[s]")
        print("隨機 300 個時間點, 各測點的有效比例 (NaN 代表缺資料):")
        for la, lo in pts:
            o = env.at(ts, np.full(300, la), np.full(300, lo))
            line = ", ".join(f"{k}:{np.isfinite(v).mean():.0%}" for k, v in o.items()
                             if k in ("u10", "swh", "u_cur", "mwd"))
            print(f"  ({la:5.1f}, {lo:6.1f})  {line}")


if __name__ == "__main__":
    main()
