import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import requests
import tempfile
import os
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path
from datetime import datetime, timezone, timedelta

# ===============================
# Page Config
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS System")
st.title("🛰️ HELIOS System")

# ===============================
# No-Go Zones & Offshore Wind
# ===============================
NO_GO_ZONES = [
    [[22.953536, 120.171678], [22.934628, 120.175472], [22.933136, 120.170942], [22.95781, 120.16078]],
    [[22.943956, 120.172358], [22.939717, 120.173944], [22.928353, 120.157372], [22.936636, 120.153547]],
    [[22.933136, 120.170942], [22.924847, 120.172583], [22.915003, 120.159022], [22.931536, 120.155772]],
]

OFFSHORE_WIND = [
    [[24.18, 120.12], [24.22, 120.28], [24.05, 120.35], [24.00, 120.15]],
    [[24.00, 120.10], [24.05, 120.32], [23.90, 120.38], [23.85, 120.15]],
    [[23.88, 120.05], [23.92, 120.18], [23.75, 120.25], [23.70, 120.08]],
    [[23.68, 120.02], [23.72, 120.12], [23.58, 120.15], [23.55, 120.05]],
]

OFFSHORE_COST = 10

# ===============================
# HYCOM 海流：逐小時時間序列 (time, lat, lon)
# ===============================
HYCOM_FORECAST_HOURS = 72  # 抓未來72小時的預報，涵蓋大多數航程長度

@st.cache_data(ttl=3600)
def load_hycom_series(bbox=(21, 26, 118, 124), hours_ahead=HYCOM_FORECAST_HOURS):
    """
    回傳從「現在」開始，未來 hours_ahead 小時內的海流時間序列。
    times_rel: 各時間切片相對於「現在」的小時數
    u_ts, v_ts: shape = (T, lat, lon)
    """
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    lat_min, lat_max, lon_min, lon_max = bbox
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
    except Exception as e:
        st.error(f"無法連接到 HYCOM 數據庫: {e}")
        st.stop()

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
    if len(valid_idx) == 0:
        valid_idx = np.array([start_idx])
    idx_slice = valid_idx

    sub = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    lons = sub.lon.values
    lats = sub.lat.values

    u_ts = sub['ssu'].isel(time=idx_slice).values  # (T, lat, lon)
    v_ts = sub['ssv'].isel(time=idx_slice).values
    times_used = time_vals[idx_slice]
    times_rel = np.array([(t - now_utc).total_seconds() / 3600.0 for t in times_used])

    land_mask = np.all(np.isnan(u_ts), axis=0)

    return lons, lats, land_mask, times_rel, times_used, u_ts, v_ts


with st.spinner("載入 HYCOM 海流時間序列中..."):
    lons, lats, land_mask, hycom_times_rel, hycom_times_abs, hycom_u_ts, hycom_v_ts = load_hycom_series()

sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)

_now_idx = int(np.argmin(np.abs(hycom_times_rel)))
obs_time = hycom_times_abs[_now_idx]


def get_current_at(y, x, elapsed_hours):
    """依「從現在起算的航行時數」取得對應時間切片的海流值。"""
    idx = int(np.argmin(np.abs(hycom_times_rel - elapsed_hours)))
    return float(hycom_u_ts[idx, y, x]), float(hycom_v_ts[idx, y, x])


def get_current_snapshot(elapsed_hours):
    """取得離 elapsed_hours 最近的整張海流快照 (u, v, 對應時刻, 索引)。用於底圖顯示。"""
    idx = int(np.argmin(np.abs(hycom_times_rel - elapsed_hours)))
    return hycom_u_ts[idx], hycom_v_ts[idx], hycom_times_abs[idx], idx


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
WEATHER_STEP_HOURS = 3               # GFS/GFS-Wave 預報時效間隔（f000, f003, f006, ...）


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

    return {
        "date": date_str,
        "cycle": cycle,
        "times_rel": times_rel,
        "wave_steps": wave_steps,
        "wind_steps": wind_steps,
    }


with st.spinner("載入氣象（風/浪）時間序列中..."):
    weather_series = fetch_weather_series()


def _nearest_step_idx(times_rel, elapsed_hours):
    return int(np.argmin(np.abs(times_rel - elapsed_hours)))


def get_wave_at(elapsed_hours):
    """依「從現在起算的航行時數」取得對應時間切片的波浪資料（整張網格）。找不到回傳 None。"""
    if weather_series is None:
        return None
    idx = _nearest_step_idx(weather_series["times_rel"], elapsed_hours)
    # 若最接近的那筆抓取失敗，往前後找最近一筆有效資料
    n = len(weather_series["wave_steps"])
    order = sorted(range(n), key=lambda i: abs(i - idx))
    for i in order:
        if weather_series["wave_steps"][i] is not None:
            return weather_series["wave_steps"][i]
    return None


def get_wind_at(elapsed_hours):
    """依「從現在起算的航行時數」取得對應時間切片的風場資料（整張網格）。找不到回傳 None。"""
    if weather_series is None:
        return None
    idx = _nearest_step_idx(weather_series["times_rel"], elapsed_hours)
    n = len(weather_series["wind_steps"])
    order = sorted(range(n), key=lambda i: abs(i - idx))
    for i in order:
        if weather_series["wind_steps"][i] is not None:
            return weather_series["wind_steps"][i]
    return None


# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("Route Settings")
    # 起點：台中港 / 終點：蘇澳港
    s_lon = st.number_input("Start Lon", 118.0, 124.0, 120.4933)  # 台中港
    s_lat = st.number_input("Start Lat", 21.0, 26.0, 24.2833)     # 台中港
    e_lon = st.number_input("End Lon", 118.0, 124.0, 121.8617)    # 蘇澳港
    e_lat = st.number_input("End Lat", 21.0, 26.0, 24.5967)       # 蘇澳港
    ship_speed = st.number_input("Ship Speed (km/h)", 1.0, 60.0, 20.0)

    st.divider()
    progress_pct = st.slider("航行進度 (%)", 0, 100, 0, key="progress_slider")
    st.caption("拖動上方滑桿即可查看船舶航行至該進度時，對應時刻的風/浪/流底圖與剩餘資訊。")

    st.divider()
    st.subheader("🧬 船型自適應環境優化模式")

    ship_mode = st.radio(
        "選擇航行船舶類型 (Ship Profile):",
        options=["大型貨輪/油輪 (CargoTanker)", "小型漁船 (Fishing)", "其他公務/客輪 (Other)"],
        index=0,
        help="系統將根據不同船型的流體動力學與環境特徵，自動載入最優化權重矩陣。"
    )

    if "CargoTanker" in ship_mode:
        w_curr = 0.25
        w_wave = 0.35
        w_wind = 0.08
        ship_type_key = "CargoTanker"
    elif "Fishing" in ship_mode:
        w_curr = 0.18
        w_wave = 0.45
        w_wind = 0.15
        ship_type_key = "Fishing"
    else:  # Other
        w_curr = 0.22
        w_wave = 0.35
        w_wind = 0.14
        ship_type_key = "Other"

# ===============================
# 船型物理參數
# ===============================
SHIP_PARAMS = {
    'CargoTanker': {
        'distance_factor': 1.3, 'current_gain': 1.0,
        'wind_gain': 1.0, 'wind_speed_gain': 1.0,
        'time_weight': 2.5, 'fuel_weight': 0.20,
        'progress_weight': 1.2, 'wave_coef': 0.08,
    },
    'Fishing': {
        'distance_factor': 1.5, 'current_gain': 0.5,
        'wind_gain': 0.5, 'wind_speed_gain': 0.5,
        'time_weight': 1.8, 'fuel_weight': 0.15,
        'progress_weight': 2.5, 'wave_coef': 0.05,
    },
    'Other': {
        'distance_factor': 1.6, 'current_gain': 0.3,
        'wind_gain': 0.2, 'wind_speed_gain': 0.2,
        'time_weight': 1.5, 'fuel_weight': 0.08,
        'progress_weight': 2.0, 'wave_coef': 0.06,
    },
}
wave_severity = 3.0 if ship_type_key == 'CargoTanker' else 1.0

MAX_CURRENT_BONUS = 2.0
MAX_WIND_BONUS = 3.0

# ===============================
# Helpers
# ===============================
def nearest_cell(lon, lat):
    return (np.abs(lats - lat).argmin(), np.abs(lons - lon).argmin())

def offshore_penalty(y, x):
    for zone in OFFSHORE_WIND:
        if Path(zone).contains_point([lons[x], lats[y]]):
            return OFFSHORE_COST
    return 0

# ===============================
# 離岸安全距離：用實際公里數校正
# ===============================
COAST_SAFE_KM = 18
_lat_res = float(np.abs(np.diff(lats)).mean()) if len(lats) > 1 else 0.05
_lon_res = float(np.abs(np.diff(lons)).mean()) if len(lons) > 1 else 0.05
CELL_KM = ((_lat_res + _lon_res) / 2) * 111.0
COAST_SAFE_CELLS = max(3.0, COAST_SAFE_KM / max(CELL_KM, 1e-6))
COAST_PENALTY_WEIGHT = 30

def coast_penalty(y, x):
    d = dist_to_land[y, x]
    if d >= COAST_SAFE_CELLS:
        return 0
    return ((COAST_SAFE_CELLS - d) ** 3) * COAST_PENALTY_WEIGHT / (COAST_SAFE_CELLS ** 2)

def heuristic(y, x, goal):
    d = np.hypot(lats[y] - lats[goal[0]], lons[x] - lons[goal[1]]) * 111
    return d * SHIP_PARAMS[ship_type_key]['distance_factor'] * 0.9

# ===============================
# 綜合成本函數 — 風、浪、流皆依 elapsed_hours 查對應時間切片
# ===============================
def get_comprehensive_cost(y0, x0, y1, x1, goal, elapsed_hours):
    dlat = lats[y1] - lats[y0]
    dlon = lons[x1] - lons[x0]
    norm = np.hypot(dlat, dlon)
    if norm == 0:
        return 0, 0.0
    dir_lat = dlat / norm
    dir_lon = dlon / norm
    base_dist = norm * 111

    p = SHIP_PARAMS[ship_type_key]
    distance_factor = p['distance_factor']
    current_gain    = p['current_gain']
    wind_gain       = p['wind_gain']
    wind_speed_gain = p['wind_speed_gain']
    time_weight     = p['time_weight']
    fuel_weight     = p['fuel_weight']
    progress_weight = p['progress_weight']
    wave_coef       = p['wave_coef']

    progress_penalty = 0
    if goal is not None:
        d_before = np.hypot(lats[y0]-lats[goal[0]], lons[x0]-lons[goal[1]]) * 111
        d_after  = np.hypot(lats[y1]-lats[goal[0]], lons[x1]-lons[goal[1]]) * 111
        forward_progress = d_before - d_after
        progress_penalty = max(base_dist - forward_progress, 0) * progress_weight * 0.3

    # 海流：依 elapsed_hours 對應的時間切片
    current_proj = 0.0
    u_cur, v_cur = get_current_at(y0, x0, elapsed_hours)
    if not np.isnan(u_cur) and not np.isnan(v_cur):
        current_proj = (u_cur * dir_lon + v_cur * dir_lat) * 3.6
    current_bonus = min(max(current_proj, -MAX_CURRENT_BONUS), MAX_CURRENT_BONUS)
    current_cost = -current_bonus * current_gain

    # 🆕 風場：依 elapsed_hours 對應的時間切片
    wind_proj = 0.0
    wind_cost = 0.0
    wnd = get_wind_at(elapsed_hours)
    if wnd:
        wi = np.abs(wnd["lats"] - lats[y0]).argmin()
        wj = np.abs(wnd["lons"] - lons[x0]).argmin()
        wind_proj = (float(wnd["u"][wi, wj]) * dir_lon +
                     float(wnd["v"][wi, wj]) * dir_lat) * 3.6
        wind_bonus = min(max(wind_proj, -MAX_WIND_BONUS), MAX_WIND_BONUS)
        wind_cost = -wind_bonus * wind_gain

    # 🆕 波浪：依 elapsed_hours 對應的時間切片
    wave_cost = 0.0
    wave_slowdown = 1.0
    wave_proj = 0.0
    w = get_wave_at(elapsed_hours)
    if w:
        wi = np.abs(w["lats"] - lats[y0]).argmin()
        wj = np.abs(w["lons"] - lons[x0]).argmin()
        swh = w["swh_grid"][wi, wj]
        if not np.isnan(swh) and swh > 0:
            dirpw_grid = w.get("dirpw_grid")
            has_dir = dirpw_grid is not None and not np.isnan(dirpw_grid[wi, wj])
            if has_dir:
                wave_dir_rad = np.radians(dirpw_grid[wi, wj] + 180.0)
                wave_dir_lon = np.sin(wave_dir_rad)
                wave_dir_lat = np.cos(wave_dir_rad)
                wave_proj = wave_dir_lon * dir_lon + wave_dir_lat * dir_lat
                wave_slowdown = 1.0 - wave_proj * swh * wave_coef
                wave_slowdown = max(0.5, min(wave_slowdown, 1.5))
            else:
                wave_slowdown = 1.0 + swh * wave_coef
            wave_cost = swh ** 2 * wave_severity * max(
                1.0 - wave_proj if has_dir else 1.0, 0)

    effective_speed = (ship_speed + current_bonus * current_gain +
                        wind_proj * wind_speed_gain * 0.5) / wave_slowdown
    effective_speed = max(effective_speed, 2.0)

    segment_time_hours = base_dist / effective_speed
    time_cost = segment_time_hours * time_weight
    fuel_cost = (40 + 0.5 * effective_speed) * segment_time_hours * fuel_weight

    total_cost = (
        base_dist * distance_factor +
        current_cost * w_curr +
        wind_cost * w_wind +
        wave_cost * w_wave +
        time_cost + fuel_cost +
        progress_penalty +
        coast_penalty(y1, x1)
    )
    return max(total_cost, 0.05), segment_time_hours

# ===============================
# A* Pathfinding — 同步追蹤「累積航行時間」
# ===============================
dirs = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

def astar(start, goal):
    rows, cols = land_mask.shape
    pq = [(heuristic(start[0], start[1], goal), start)]
    came = {}
    cost = {start: 0}
    elapsed_time = {start: 0.0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            break
        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni, nj]:
                step_cost, seg_time = get_comprehensive_cost(
                    cur[0], cur[1], ni, nj, goal, elapsed_time[cur]
                )
                step_cost += offshore_penalty(ni, nj)

                new_g = cost[cur] + step_cost

                if (ni, nj) not in cost or new_g < cost[(ni, nj)]:
                    cost[(ni, nj)] = new_g
                    elapsed_time[(ni, nj)] = elapsed_time[cur] + seg_time
                    came[(ni, nj)] = cur
                    f = new_g + heuristic(ni, nj, goal)
                    heapq.heappush(pq, (f, (ni, nj)))

    path = []
    cur = goal
    while cur in came:
        path.append(cur)
        cur = came[cur]
    if path:
        path.append(start)
    return path[::-1]

# ===============================
# Route Logic
# ===============================
if "full_path" not in st.session_state:
    st.session_state.full_path = []
if "ship_step_idx" not in st.session_state:
    st.session_state.ship_step_idx = 0
if "route_key" not in st.session_state:
    st.session_state.route_key = None

start = nearest_cell(s_lon, s_lat)
goal  = nearest_cell(e_lon, e_lat)

route_key = (s_lon, s_lat, e_lon, e_lat, ship_mode)

if st.session_state.route_key != route_key:
    with st.spinner("HELIOS 尋路引擎正依據船型特徵與逐時風浪流變化進行最優解算..."):
        new_path = astar(start, goal)
    if len(new_path) == 0:
        st.error("❌ 無法在當前海況與船型設定下找到安全航線")
        st.stop()
    st.session_state.full_path = new_path
    st.session_state.ship_step_idx = 0
    st.session_state.route_key = route_key

path = st.session_state.full_path

if path:
    st.session_state.ship_step_idx = min(st.session_state.ship_step_idx, len(path)-1)

if path:
    slider_idx = int((progress_pct / 100) * (len(path) - 1))
    st.session_state.ship_step_idx = slider_idx

if not path:
    st.stop()

current_pos = path[st.session_state.ship_step_idx]

# ===============================
# Calculations — 剩餘距離/時間，逐段依對應時刻查風浪流
# ===============================
def calc_segment(y0, x0, y1, x1, elapsed_hours):
    p = SHIP_PARAMS[ship_type_key]
    current_gain    = p['current_gain']
    wind_speed_gain = p['wind_speed_gain']
    wave_coef       = p['wave_coef']

    seg_dist = np.hypot(lats[y1]-lats[y0], lons[x1]-lons[x0]) * 111
    dlat = lats[y1] - lats[y0]
    dlon = lons[x1] - lons[x0]
    norm = np.hypot(dlat, dlon)
    if norm == 0:
        return 0.0, 0.0

    dir_lat = dlat / norm
    dir_lon = dlon / norm

    current_proj = 0.0
    u_cur, v_cur = get_current_at(y0, x0, elapsed_hours)
    if not np.isnan(u_cur) and not np.isnan(v_cur):
        current_proj = (u_cur * dir_lon + v_cur * dir_lat) * 3.6
    current_bonus = min(max(current_proj, -MAX_CURRENT_BONUS), MAX_CURRENT_BONUS)

    wind_proj = 0.0
    wnd = get_wind_at(elapsed_hours)
    if wnd:
        wi = np.abs(wnd["lats"] - lats[y0]).argmin()
        wj = np.abs(wnd["lons"] - lons[x0]).argmin()
        wind_proj = (float(wnd["u"][wi, wj]) * dir_lon +
                     float(wnd["v"][wi, wj]) * dir_lat) * 3.6

    wave_slowdown = 1.0
    w = get_wave_at(elapsed_hours)
    if w:
        wi = np.abs(w["lats"] - lats[y0]).argmin()
        wj = np.abs(w["lons"] - lons[x0]).argmin()
        swh = w["swh_grid"][wi, wj]
        if not np.isnan(swh) and swh > 0:
            dirpw_grid = w.get("dirpw_grid")
            has_dir = dirpw_grid is not None and not np.isnan(dirpw_grid[wi, wj])
            if has_dir:
                wave_dir_rad = np.radians(dirpw_grid[wi, wj] + 180.0)
                wave_proj = (np.sin(wave_dir_rad) * dir_lon +
                             np.cos(wave_dir_rad) * dir_lat)
                wave_slowdown = 1.0 - wave_proj * swh * wave_coef
                wave_slowdown = max(0.5, min(wave_slowdown, 1.5))
            else:
                wave_slowdown = 1.0 + swh * wave_coef

    effective_speed = (ship_speed + current_bonus * current_gain +
                        wind_proj * wind_speed_gain * 0.5) / wave_slowdown
    effective_speed = max(effective_speed, 2.0)

    seg_time = seg_dist / effective_speed
    return seg_dist, seg_time


def calc_remaining(path, idx):
    elapsed = 0.0
    for i in range(0, idx):
        y0, x0 = path[i]
        y1, x1 = path[i + 1]
        _, seg_time = calc_segment(y0, x0, y1, x1, elapsed)
        elapsed += seg_time

    elapsed_at_current = elapsed  # 🆕 船抵達目前位置時，從現在算起已經過的小時數

    dist = 0.0
    total_time = 0.0
    for i in range(idx, len(path) - 1):
        y0, x0 = path[i]
        y1, x1 = path[i + 1]
        seg_dist, seg_time = calc_segment(y0, x0, y1, x1, elapsed)
        dist += seg_dist
        total_time += seg_time
        elapsed += seg_time

    if idx < len(path) - 1:
        y0, x0 = path[idx]
        y1, x1 = path[idx + 1]
        heading = np.degrees(np.arctan2(lats[y1]-lats[y0], lons[x1]-lons[x0]))
    else:
        heading = 0

    return dist, total_time, heading, elapsed_at_current

remaining_dist, remaining_time, heading, elapsed_at_current = calc_remaining(
    path, st.session_state.ship_step_idx
)

# ===============================
# 🆕 依「船目前預計時刻」取得對應的風/浪/流底圖資料
# 這樣拖動「航行進度」滑桿時，底圖會跟著切換到船屆時對應的風浪流狀況，
# 而不是永遠顯示「現在」這一份快照。
# ===============================
map_u, map_v, map_time, _map_idx = get_current_snapshot(elapsed_at_current)
map_wave = get_wave_at(elapsed_at_current)
map_wind = get_wind_at(elapsed_at_current)

# ===============================
# Dashboard — 第一排
# ===============================
st.subheader("Navigation Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Remaining Distance (km)", f"{remaining_dist:.2f}")
c2.metric("Remaining Time (hr)",     f"{remaining_time:.2f}")
c3.metric("Heading",                 f"{heading:.1f}°")
c4.metric("目前預計航行時數",         f"{elapsed_at_current:.2f} hr")

# 第二排 — 氣象資料（依目前位置對應的時刻）
w1, w2, w3 = st.columns(3)
if map_wave:
    wi = np.abs(map_wave["lats"] - lats[current_pos[0]]).argmin()
    wj = np.abs(map_wave["lons"] - lons[current_pos[1]]).argmin()
    swh_here = map_wave["swh_grid"][wi, wj]
    w1.metric("顯著波高（目前位置）", f"{swh_here:.2f} m" if not np.isnan(swh_here) else "N/A")
    if map_wave.get("dirpw_grid") is not None:
        dirpw_here = map_wave["dirpw_grid"][wi, wj]
        w2.metric("波浪方向（目前位置）", f"{dirpw_here:.1f}°" if not np.isnan(dirpw_here) else "N/A")
    else:
        w2.metric("波浪方向（目前位置）", "N/A")
else:
    w1.metric("顯著波高（目前位置）", "N/A")
    w2.metric("波浪方向（目前位置）", "N/A")

if map_wind:
    wi = np.abs(map_wind["lats"] - lats[current_pos[0]]).argmin()
    wj = np.abs(map_wind["lons"] - lons[current_pos[1]]).argmin()
    spd_here = float(map_wind["speed"][wi, wj])
    w3.metric("風速（目前位置）", f"{spd_here:.2f} m/s")
else:
    w3.metric("風速（目前位置）", "N/A")

st.caption(
    f"HYCOM/GFS 對應時刻（船抵達目前位置時預估）: {map_time} ｜ "
    f"已載入海流 {len(hycom_times_rel)} 個時間切片（涵蓋未來 {HYCOM_FORECAST_HOURS} 小時）、"
    f"風浪 {len(weather_series['times_rel']) if weather_series else 0} 個時間切片"
    f"（每 {WEATHER_STEP_HOURS} 小時一筆，涵蓋未來 {WEATHER_FORECAST_HOURS} 小時）"
)
st.caption(f"離岸安全距離：約 {COAST_SAFE_KM} km（換算約 {COAST_SAFE_CELLS:.1f} 個網格，每格約 {CELL_KM:.1f} km）")

# ===============================
# Map — 底圖依「船目前預計時刻」對應的風/浪/流資料繪製
# ===============================
fig = plt.figure(figsize=(10, 8))
ax  = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118, 124, 21, 26])
ax.add_feature(cfeature.LAND,       facecolor="#b0b0b0")
ax.add_feature(cfeature.COASTLINE)

# 海流（顯示「船目前預計時刻」對應的快照）
try:
    speed_cur = np.sqrt(map_u**2 + map_v**2)
    mesh = ax.pcolormesh(lons, lats, speed_cur,
                         cmap="Blues", shading="auto", vmin=0, vmax=1.6,
                         transform=ccrs.PlateCarree())
    fig.colorbar(mesh, ax=ax, label="Current Speed (m/s)")
except Exception:
    st.warning("Could not overlay current data.")

# 波浪等高線（同一對應時刻）
if map_wave:
    wlon_grid, wlat_grid = np.meshgrid(map_wave["lons"], map_wave["lats"])
    contour = ax.contour(
        wlon_grid, wlat_grid, map_wave["swh_grid"],
        levels=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        cmap="cool", linewidths=1.2,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(contour, fmt="%.1fm", fontsize=7, inline=True)

# 風場箭頭（同一對應時刻）
if map_wind:
    wlon_g, wlat_g = np.meshgrid(map_wind["lons"], map_wind["lats"])
    ax.quiver(wlon_g[::2, ::2], wlat_g[::2, ::2],
              map_wind["u"][::2, ::2], map_wind["v"][::2, ::2],
              scale=200, color="white", alpha=0.5,
              transform=ccrs.PlateCarree())

# 禁航區
for zone in NO_GO_ZONES:
    poly = np.array(zone)
    ax.fill(poly[:,1], poly[:,0], color="red",    alpha=0.4, transform=ccrs.PlateCarree())
for zone in OFFSHORE_WIND:
    poly = np.array(zone)
    ax.fill(poly[:,1], poly[:,0], color="yellow", alpha=0.4, transform=ccrs.PlateCarree())

# 路徑
full_lons = [lons[p[1]] for p in path]
full_lats = [lats[p[0]] for p in path]
ax.plot(full_lons, full_lats, color="pink",  linewidth=2, transform=ccrs.PlateCarree())

done_lons = full_lons[:st.session_state.ship_step_idx+1]
done_lats = full_lats[:st.session_state.ship_step_idx+1]
ax.plot(done_lons, done_lats, color="red", linewidth=2, transform=ccrs.PlateCarree())

ax.scatter(lons[current_pos[1]], lats[current_pos[0]],
           color="gray", marker="^", s=150, zorder=5, transform=ccrs.PlateCarree())
ax.scatter(s_lon, s_lat, color="#B15BFF", s=80,  edgecolors="black", transform=ccrs.PlateCarree())
ax.scatter(e_lon, e_lat, color="yellow",  marker="*", s=200, edgecolors="black", transform=ccrs.PlateCarree())

plt.title(f"HELIOS Navigation Map ｜ 對應時刻: {map_time}")
st.pyplot(fig)
