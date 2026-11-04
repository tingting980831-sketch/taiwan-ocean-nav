"""
data_live.py -- 即時海況資料 (HYCOM 海流 + GFS 風 + GFS-Wave 浪) 的抓取與介面

抓取邏輯沿用舊版 app (HYCOM 逐時歸檔 -> FMRC 3 小時預報備援; GFS/GFS-Wave 逐時預報),
再加一層 LiveEnv, 讓它符合 planner.py / fuel_model.py 需要的介面:
    env.at(t, lat, lon) -> dict(u_cur, v_cur, u10, v10, swh, mwd)
方向慣例 (與 fuel_model.py 相同): 風 u10,v10 為「吹向」分量 (m/s); mwd 為「浪從哪個方向來」(度)。
"""
import os
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xarray as xr

from env_fields import bilinear

# ===================== 以下抓取程式沿用舊版 (未改動) =====================
HYCOM_FORECAST_HOURS = 72  # 抓未來72小時的預報，涵蓋大多數航程長度

FMRC_UV3Z_URL = "https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"


def _load_hycom_from_archive(bbox, hours_ahead, now_utc):
    """嘗試從逐時歸檔資料集抓取。回傳 None 代表往未來抓不到足夠的時間點。"""
    lat_min, lat_max, lon_min, lon_max = bbox
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)

    if 'time_origin' in ds['time'].attrs:
        origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        if origin.tzinfo is None:
            origin = origin.tz_localize('UTC')
        time_vals = origin + pd.to_timedelta(ds['time'].values, unit='h')
    else:
        ds_decoded = xr.decode_cf(ds)
        time_vals = pd.DatetimeIndex(pd.to_datetime(ds_decoded['time'].values, utc=True))

    time_vals = pd.DatetimeIndex(time_vals)
    deltas = (time_vals - now_utc).total_seconds() / 3600.0
    start_idx = int(np.argmin(np.abs(deltas)))
    end_mask = deltas <= hours_ahead
    valid_idx = np.where(end_mask)[0]
    valid_idx = valid_idx[valid_idx >= start_idx]

    # 往未來抓不到至少 2 個時間點，代表這個歸檔端點沒有未來預報，改用 FMRC 備援
    if len(valid_idx) < 2:
        return None

    sub = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    lons = sub.lon.values
    lats = sub.lat.values

    u_ts = sub['ssu'].isel(time=valid_idx).values  # (T, lat, lon)
    v_ts = sub['ssv'].isel(time=valid_idx).values
    times_used = time_vals[valid_idx]
    times_rel = np.array([(t - now_utc).total_seconds() / 3600.0 for t in times_used])
    land_mask = np.all(np.isnan(u_ts), axis=0)

    return lons, lats, land_mask, times_rel, times_used, u_ts, v_ts


def _load_hycom_from_fmrc(bbox, hours_ahead, now_utc):
    """備援：從 FMRC 3小時解析度預報聚合端點抓取（涵蓋未來最多8天）。

    🆕 FMRC 聚合資料集裡常有像 runtime/time_run 這類輔助座標使用
    "hours since analysis" 這種非標準時間單位，若用 xr.open_dataset 預設
    decode_times=True 或事後呼叫 xr.decode_cf()，會因為這些輔助座標解碼失敗
    而整個炸掉。這裡改用 decode_times=False 打開，只手動解碼真正需要的
    'time' 座標本身，其餘座標維持原樣不去動它。
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    ds = xr.open_dataset(FMRC_UV3Z_URL, decode_times=False)

    raw_time = ds['time'].values
    time_units = ds['time'].attrs.get('units', '')
    calendar = ds['time'].attrs.get('calendar', 'standard')

    try:
        decoded = xr.coding.times.decode_cf_datetime(raw_time, time_units, calendar=calendar)
        time_vals = pd.DatetimeIndex(decoded)
    except ImportError:
        st.error("缺少 cftime 套件，請先執行 `pip install cftime` 後再重新啟動。")
        st.stop()
    except Exception:
        try:
            import cftime
        except ImportError:
            st.error("缺少 cftime 套件，請先執行 `pip install cftime` 後再重新啟動。")
            st.stop()
        decoded = cftime.num2date(raw_time, time_units, calendar=calendar)
        time_vals = pd.DatetimeIndex([pd.Timestamp(d.isoformat()) for d in decoded])

    if time_vals.tz is None:
        time_vals = time_vals.tz_localize('UTC')

    deltas = (time_vals - now_utc).total_seconds() / 3600.0
    start_idx = int(np.argmin(np.abs(deltas)))
    end_mask = deltas <= hours_ahead
    valid_idx = np.where(end_mask)[0]
    valid_idx = valid_idx[valid_idx >= start_idx]
    if len(valid_idx) == 0:
        valid_idx = np.array([start_idx])

    sub = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    lons = sub.lon.values
    lats = sub.lat.values

    # uv3z 含深度層，取最接近海表（depth=0）的那一層
    depth_dim = 'depth' if 'depth' in sub['water_u'].dims else None
    if depth_dim:
        d_idx = int(np.abs(sub[depth_dim].values - 0).argmin())
        u_ts = sub['water_u'].isel(time=valid_idx).isel({depth_dim: d_idx}).values
        v_ts = sub['water_v'].isel(time=valid_idx).isel({depth_dim: d_idx}).values
    else:
        u_ts = sub['water_u'].isel(time=valid_idx).values
        v_ts = sub['water_v'].isel(time=valid_idx).values

    times_used = time_vals[valid_idx]
    times_rel = np.array([(t - now_utc).total_seconds() / 3600.0 for t in times_used])
    land_mask = np.all(np.isnan(u_ts), axis=0)

    return lons, lats, land_mask, times_rel, times_used, u_ts, v_ts


@st.cache_data(ttl=3600)
def load_hycom_series(bbox=(21, 26, 118, 124), hours_ahead=HYCOM_FORECAST_HOURS):
    """
    回傳從「現在」開始，未來 hours_ahead 小時內的海流時間序列。
    times_rel: 各時間切片相對於「現在」的小時數
    u_ts, v_ts: shape = (T, lat, lon)

    先試逐時歸檔資料，抓不到足夠未來時間點時自動改用 FMRC 3小時預報備援。
    """
    now_utc = pd.Timestamp(datetime.now(timezone.utc))

    try:
        result = _load_hycom_from_archive(bbox, hours_ahead, now_utc)
        if result is not None:
            return result
    except Exception:
        pass

    try:
        result = _load_hycom_from_fmrc(bbox, hours_ahead, now_utc)
        if result is not None:
            return result
    except Exception as e:
        st.error(f"無法連接到 HYCOM 數據庫（歸檔與 FMRC 備援皆失敗）: {e}")
        st.stop()

    st.error("無法取得任何 HYCOM 海流資料。")
    st.stop()


# ===============================
# 🆕 風場 + 波浪：改成逐小時時間序列
#
# 問題背景：原本風、浪只抓 f000（預報起始那一刻）單一時間切片，
# 全航程都套用同一組資料。跟海流一樣，若航程規劃三小時，
# 第二、三小時經過海域的風浪早就不是「現在」的樣子。
#
# 修正方式：一次抓 GFS/GFS-Wave 未來 WEATHER_FORECAST_HOURS 小時內、
# 每小時（或每 3 小時，依資料本身解析度）一筆的預報時效（f000, f003, f006...），
# 組成 (time, lat, lon) 的時間序列，之後依「船預計幾點到達該點」查對應時刻。
# ===============================
WEATHER_FORECAST_HOURS = 72          # 涵蓋未來 72 小時預報
WEATHER_STEP_HOURS = 1               # GFS/GFS-Wave 逐小時預報時效（f000, f001, f002, ... f120 內皆支援逐時）


def _find_latest_cycle():
    """找到目前可用的最新一組 GFS run (date, cycle)。"""
    for days_back in range(0, 4):
        check_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        d = check_date.strftime("%Y%m%d")
        for c in ["18", "12", "06", "00"]:
            url = (
                "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl?"
                f"file=gfswave.t{c}z.global.0p25.f000.grib2"
                "&var_HTSGW=on&lev_surface=on&subregion="
                "&leftlon=118&rightlon=124&toplat=26&bottomlat=21"
                f"&dir=%2Fgfs.{d}%2F{c}%2Fwave%2Fgridded"
            )
            try:
                r = requests.head(url, timeout=10)
                if r.status_code == 200:
                    return d, c
            except Exception:
                continue
    return None, None


def _fetch_wave_step(date_str, cycle, fhr):
    """抓單一預報時效（fhr, 小時）的波浪資料，失敗回傳 None。"""
    f_str = f"f{fhr:03d}"
    url = (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl?"
        f"file=gfswave.t{cycle}z.global.0p25.{f_str}.grib2"
        "&var_HTSGW=on&var_PERPW=on&var_DIRPW=on"
        "&lev_surface=on&subregion="
        "&leftlon=118&rightlon=124&toplat=26&bottomlat=21"
        f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fwave%2Fgridded"
    )
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or len(r.content) < 500:
            return None
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name
        ds_w = xr.open_dataset(
            tmp_path, engine="cfgrib",
            filter_by_keys={"stepRange": str(fhr), "typeOfLevel": "surface"}
        )
        swh_grid = ds_w["swh"].values
        wlats = ds_w["latitude"].values
        wlons = ds_w["longitude"].values
        try:
            dirpw_grid = ds_w["dirpw"].values
        except Exception:
            dirpw_grid = None
        ds_w.close()
        os.unlink(tmp_path)
        return {"swh_grid": swh_grid, "dirpw_grid": dirpw_grid, "lats": wlats, "lons": wlons}
    except Exception:
        return None


def _fetch_wind_step(date_str, cycle, fhr):
    """抓單一預報時效（fhr, 小時）的風場資料，失敗回傳 None。"""
    f_str = f"{fhr:03d}"
    url = (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
        f"file=gfs.t{cycle}z.pgrb2.0p25.f{f_str}"
        "&lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on"
        "&subregion=&leftlon=118&rightlon=124&toplat=26&bottomlat=21"
        f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"
    )
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or len(r.content) < 500:
            return None
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name
        ds_f = xr.open_dataset(
            tmp_path, engine="cfgrib",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10}
        )
        u = ds_f["u10"].values
        v = ds_f["v10"].values
        wlats = ds_f["latitude"].values
        wlons = ds_f["longitude"].values
        ds_f.close()
        os.unlink(tmp_path)
        return {"u": u, "v": v, "speed": np.sqrt(u**2 + v**2), "lats": wlats, "lons": wlons}
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_weather_series(forecast_hours=WEATHER_FORECAST_HOURS, step_hours=WEATHER_STEP_HOURS):
    """
    回傳風、浪的逐時（每 step_hours 小時一筆）時間序列。
    結構：
    {
      "date": date_str, "cycle": cycle,
      "times_rel": [0, 3, 6, ...],         # 相對於現在（run 起算）的小時數
      "wave_steps": [ {swh_grid, dirpw_grid, lats, lons} 或 None, ... ],
      "wind_steps": [ {u, v, speed, lats, lons} 或 None, ... ],
    }
    """
    date_str, cycle = _find_latest_cycle()
    if not date_str:
        return None

    fhrs = list(range(0, forecast_hours + 1, step_hours))
    wave_steps = []
    wind_steps = []
    for fhr in fhrs:
        wave_steps.append(_fetch_wave_step(date_str, cycle, fhr))
        wind_steps.append(_fetch_wind_step(date_str, cycle, fhr))

    # run 開始時刻（cycle 那一刻）相對於「現在」的小時差，
    # 讓 times_rel 對齊到跟海流一樣的「現在 = 0」基準
    run_start = pd.Timestamp(
        datetime.strptime(date_str + cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    )
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    run_offset_hours = (run_start - now_utc).total_seconds() / 3600.0
    times_rel = np.array([run_offset_hours + fhr for fhr in fhrs])
    times_abs = [run_start + pd.Timedelta(hours=fhr) for fhr in fhrs]

    return {
        "date": date_str,
        "cycle": cycle,
        "times_rel": times_rel,
        "times_abs": times_abs,
        "wave_steps": wave_steps,
        "wind_steps": wind_steps,
    }



# ===================== 轉接層: 讓規劃器可以查任意時間、任意位置的海況 =====================
def _asc(lats, lons, arr):
    """讓緯度、經度由小到大 (GFS 的緯度是由大到小)。arr 最後兩維為 (lat, lon)。"""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    if lats[0] > lats[-1]:
        lats, arr = lats[::-1], arr[..., ::-1, :]
    if lons[0] > lons[-1]:
        lons, arr = lons[::-1], arr[..., ::-1]
    return lats, lons, np.asarray(arr, float)


class TimeField:
    """(time, lat, lon) 的場; 空間雙線性、時間線性內插; 超出時間範圍時取最近一筆。"""

    def __init__(self, times_rel, lats, lons, arrs):
        order = np.argsort(times_rel)
        self.t = np.asarray(times_rel, float)[order]
        self.arrs = {k: np.asarray(v)[order] for k, v in arrs.items()}
        self.lats, self.lons = lats, lons

    def sample(self, names, hours, lat, lon):
        hours = np.broadcast_to(np.asarray(hours, float), np.shape(lat)).copy()
        n = len(self.t)
        i1 = np.clip(np.searchsorted(self.t, hours), 0, n - 1)
        i0 = np.clip(i1 - 1, 0, n - 1)
        span = np.where(i1 > i0, self.t[i1] - self.t[i0], 1.0)
        w = np.where(i1 > i0, np.clip((hours - self.t[i0]) / span, 0.0, 1.0), 0.0)
        out = {}
        for nm in names:
            a = self.arrs[nm]
            v = np.full(np.shape(lat), np.nan)
            for k0 in np.unique(i0):
                for k1 in np.unique(i1[i0 == k0]):
                    m = (i0 == k0) & (i1 == k1)
                    v0 = bilinear(a[k0], self.lats, self.lons, lat[m], lon[m])
                    v1 = v0 if k1 == k0 else bilinear(a[k1], self.lats, self.lons, lat[m], lon[m])
                    v[m] = (1 - w[m]) * v0 + w[m] * v1
            out[nm] = v
        return out


def _grid_at(field, name, hours):
    """某個時刻的整張場 (時間線性內插), 供底圖繪製。"""
    t = field.t
    i1 = int(np.clip(np.searchsorted(t, hours), 0, len(t) - 1))
    i0 = max(i1 - 1, 0)
    w = 0.0 if i1 == i0 else float(np.clip((hours - t[i0]) / (t[i1] - t[i0]), 0, 1))
    a = field.arrs[name]
    return (1 - w) * a[i0] + w * a[i1]


class LiveEnv:
    """把 HYCOM 時間序列與 GFS/GFS-Wave 預報包成 planner.py / fuel_model.py 要的 env.at(t, lat, lon)。
    t 為 datetime64 (UTC, 絕對時間); 「現在」= now_utc, 內部以相對於現在的小時數查表。"""

    def __init__(self, hycom, weather):
        lons, lats, _land, t_rel, t_abs, u_ts, v_ts = hycom
        # 「現在」= 資料抓取當下: 由 HYCOM 的絕對時間與相對小時反推 (不依賴呼叫端的時鐘)
        now = pd.DatetimeIndex(t_abs)[0] - pd.Timedelta(hours=float(np.asarray(t_rel, float)[0]))
        now = now.tz_convert("UTC").tz_localize(None) if now.tzinfo is not None else now
        self.now = now.to_datetime64().astype("datetime64[s]")
        la, lo, u = _asc(lats, lons, u_ts)
        _, _, v = _asc(lats, lons, v_ts)
        self.cur = TimeField(t_rel, la, lo, {"u": u, "v": v})
        self.wave = self.wind = None
        if weather:
            # 風浪各有自己的相對時間基準, 統一換成「相對於 self.now 的小時數」
            ts = np.array([
                (pd.Timestamp(t).tz_convert("UTC").tz_localize(None).to_datetime64().astype("datetime64[s]") - self.now)
                / np.timedelta64(1, "s") / 3600.0 for t in weather["times_abs"]])
            ok = [i for i, s in enumerate(weather["wave_steps"]) if s is not None]
            if ok:
                s0 = weather["wave_steps"][ok[0]]
                sw = np.stack([weather["wave_steps"][i]["swh_grid"] for i in ok])
                dirs = [weather["wave_steps"][i].get("dirpw_grid") for i in ok]
                la, lo, swh = _asc(s0["lats"], s0["lons"], sw)
                arrs = {"swh": swh}
                if all(d is not None for d in dirs):
                    d = np.stack(dirs)
                    _, _, ds = _asc(s0["lats"], s0["lons"], np.sin(np.radians(d)))
                    _, _, dc = _asc(s0["lats"], s0["lons"], np.cos(np.radians(d)))
                    arrs.update(dir_sin=ds, dir_cos=dc)
                self.wave = TimeField(ts[ok], la, lo, arrs)
            ok = [i for i, s in enumerate(weather["wind_steps"]) if s is not None]
            if ok:
                s0 = weather["wind_steps"][ok[0]]
                la, lo, uu = _asc(s0["lats"], s0["lons"], np.stack([weather["wind_steps"][i]["u"] for i in ok]))
                _, _, vv = _asc(s0["lats"], s0["lons"], np.stack([weather["wind_steps"][i]["v"] for i in ok]))
                self.wind = TimeField(ts[ok], la, lo, {"u": uu, "v": vv})

    def hours(self, t):
        return (np.asarray(t, dtype="datetime64[s]") - self.now.astype("datetime64[s]")).astype("timedelta64[s]").astype(float) / 3600.0

    def at(self, t, lat, lon):
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        h = self.hours(t)
        out = {}
        c = self.cur.sample(["u", "v"], h, lat, lon)
        out["u_cur"], out["v_cur"] = c["u"], c["v"]
        if self.wind is not None:
            w = self.wind.sample(["u", "v"], h, lat, lon)
            out["u10"], out["v10"] = w["u"], w["v"]
        if self.wave is not None:
            names = ["swh"] + (["dir_sin", "dir_cos"] if "dir_sin" in self.wave.arrs else [])
            w = self.wave.sample(names, h, lat, lon)
            out["swh"] = w["swh"]
            if "dir_sin" in w:
                out["mwd"] = (np.degrees(np.arctan2(w["dir_sin"], w["dir_cos"])) + 360.0) % 360.0
        return out

    def snapshot(self, t):
        """某個時刻的整張海流、風、浪場 (底圖用)。回傳 dict; 沒有的資料不放進去。"""
        h = float(self.hours(t))
        out = {"cur": (self.cur.lons, self.cur.lats, _grid_at(self.cur, "u", h), _grid_at(self.cur, "v", h))}
        if self.wind is not None:
            out["wind"] = (self.wind.lons, self.wind.lats, _grid_at(self.wind, "u", h), _grid_at(self.wind, "v", h))
        if self.wave is not None:
            out["wave"] = (self.wave.lons, self.wave.lats, _grid_at(self.wave, "swh", h))
        return out
