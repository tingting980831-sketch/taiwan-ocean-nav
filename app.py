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
# 🆕 Load HYCOM as a TIME SERIES (not a single snapshot)
#
# 問題背景：原本只抓「最接近現在」的單一時間切片，整個航程（可能好幾小時）
# 全部套用同一組海流資料。但船開 2 小時，第 2 小時經過的海域海流早就不是
# 「現在」的樣子了 —— 尤其台灣海峽潮流變化很快。
#
# 修正方式：一次抓未來 HYCOM_FORECAST_HOURS 小時內的所有時間切片，
# 回傳完整的 3D 陣列 (time, lat, lon)，之後在計算成本/剩餘時間時，
# 用「船預計幾點到達該點」去查對應時間的海流，而不是永遠查「現在」。
# ===============================
HYCOM_FORECAST_HOURS = 72  # 抓未來72小時的預報，涵蓋大多數航程長度

@st.cache_data(ttl=3600)
def load_hycom_series(bbox=(21, 26, 118, 124), hours_ahead=HYCOM_FORECAST_HOURS):
    """
    回傳從「現在」開始，未來 hours_ahead 小時內的海流時間序列。
    times_rel: 各時間切片相對於「現在」的小時數（可能含些微負值，
               因為會保留最接近現在的那一筆做為 t=0 起點）
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

    # 找到「最接近現在」的索引當作序列起點 t=0，
    # 然後往未來抓到 now + hours_ahead 為止
    deltas = (time_vals - now_utc).total_seconds() / 3600.0
    start_idx = int(np.argmin(np.abs(deltas)))
    end_mask = deltas <= hours_ahead
    # 確保至少從 start_idx 開始，且不超過資料集尾端
    valid_idx = np.where(end_mask)[0]
    valid_idx = valid_idx[valid_idx >= start_idx]
    if len(valid_idx) == 0:
        valid_idx = np.array([start_idx])
    idx_slice = valid_idx  # 已經是遞增排序的索引陣列

    sub = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    lons = sub.lon.values
    lats = sub.lat.values

    u_ts = sub['ssu'].isel(time=idx_slice).values  # (T, lat, lon)
    v_ts = sub['ssv'].isel(time=idx_slice).values
    times_used = time_vals[idx_slice]
    times_rel = np.array([(t - now_utc).total_seconds() / 3600.0 for t in times_used])

    # land_mask：用序列中「所有時間點皆為 NaN」的位置才視為陸地，
    # 避免單一時間切片偶發缺值就被誤判為陸地
    land_mask = np.all(np.isnan(u_ts), axis=0)

    return lons, lats, land_mask, times_rel, times_used, u_ts, v_ts


with st.spinner("載入 HYCOM 海流時間序列中..."):
    lons, lats, land_mask, hycom_times_rel, hycom_times_abs, hycom_u_ts, hycom_v_ts = load_hycom_series()

sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)

# 🆕 給地圖顯示與各種「現在這一刻」用途使用的索引（最接近 t=0 的切片）
_now_idx = int(np.argmin(np.abs(hycom_times_rel)))
obs_time = hycom_times_abs[_now_idx]
hycom_u = hycom_u_ts[_now_idx]  # 保留給地圖繪製用的「現在」快照
hycom_v = hycom_v_ts[_now_idx]


def get_current_at(y, x, elapsed_hours):
    """
    依「從現在起算的航行時數」取得對應時間切片的海流值。
    這是解決「開2小時、海流資料卻沒變」問題的核心函式。
    """
    idx = int(np.argmin(np.abs(hycom_times_rel - elapsed_hours)))
    return float(hycom_u_ts[idx, y, x]), float(hycom_v_ts[idx, y, x])


# ===============================
# 風場 + 波浪資料
# （目前仍為單一時間切片 f000，若航程很長，未來可比照海流做法
#   改抓多個預報時效 f000/f003/f006... 一併時間內插，先不在此次修正範圍）
# ===============================
@st.cache_data(ttl=3600)
def fetch_weather():
    date_str, cycle = None, None
    for days_back in range(0, 4):
        check_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        d = check_date.strftime("%Y%m%d")
        for c in ["00", "06", "12", "18"]:
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
                    date_str, cycle = d, c
                    break
            except:
                continue
        if date_str:
            break

    if not date_str:
        return None

    result = {"date": date_str, "cycle": cycle}

    # 波浪
    try:
        url = (
            "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl?"
            f"file=gfswave.t{cycle}z.global.0p25.f000.grib2"
            "&var_HTSGW=on&var_PERPW=on&var_DIRPW=on"
            "&lev_surface=on&subregion="
            "&leftlon=118&rightlon=124&toplat=26&bottomlat=21"
            f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fwave%2Fgridded"
        )
        r = requests.get(url, timeout=30)
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

        ds_w = xr.open_dataset(
            tmp_path, engine="cfgrib",
            filter_by_keys={"stepRange": "0", "typeOfLevel": "surface"}
        )
        swh_grid = ds_w["swh"].values
        wlats = ds_w["latitude"].values
        wlons = ds_w["longitude"].values

        try:
            dirpw_grid = ds_w["dirpw"].values
        except Exception:
            dirpw_grid = None

        lat_idx = np.abs(wlats - 24.0).argmin()
        lon_idx = np.abs(wlons - 122.0).argmin()
        found = False
        for radius in range(0, 6):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = lat_idx + di, lon_idx + dj
                    if 0 <= ni < len(wlats) and 0 <= nj < len(wlons):
                        if not np.isnan(swh_grid[ni, nj]):
                            lat_idx, lon_idx = ni, nj
                            found = True
                            break
                if found:
                    break
            if found:
                break

        result["wave"] = {
            "swh":        float(ds_w["swh"].isel(latitude=lat_idx, longitude=lon_idx).values),
            "dirpw":      float(ds_w["dirpw"].isel(latitude=lat_idx, longitude=lon_idx).values),
            "swh_grid":   swh_grid,
            "dirpw_grid": dirpw_grid,
            "lats":       wlats,
            "lons":       wlons,
        }
        ds_w.close()
        os.unlink(tmp_path)
    except Exception as e:
        result["wave"] = None
        st.warning(f"波浪資料載入失敗: {e}")

    # 風場
    try:
        url = (
            "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
            f"file=gfs.t{cycle}z.pgrb2.0p25.f000"
            "&lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on"
            "&subregion=&leftlon=118&rightlon=124&toplat=26&bottomlat=21"
            f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"
        )
        r = requests.get(url, timeout=30)
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

        ds_f = xr.open_dataset(
            tmp_path, engine="cfgrib",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10}
        )
        u = ds_f["u10"].values
        v = ds_f["v10"].values
        result["wind"] = {
            "u":     u,
            "v":     v,
            "speed": np.sqrt(u**2 + v**2),
            "lats":  ds_f["latitude"].values,
            "lons":  ds_f["longitude"].values,
        }
        ds_f.close()
        os.unlink(tmp_path)
    except Exception as e:
        result["wind"] = None
        st.warning(f"風場資料載入失敗: {e}")

    return result

with st.spinner("載入氣象資料中..."):
    weather = fetch_weather()

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("Route Settings")
    s_lon = st.number_input("Start Lon", 118.0, 124.0, 120.3)
    s_lat = st.number_input("Start Lat", 21.0, 26.0, 22.6)
    e_lon = st.number_input("End Lon", 118.0, 124.0, 122.0)
    e_lat = st.number_input("End Lat", 21.0, 26.0, 24.5)
    ship_speed = st.number_input("Ship Speed (km/h)", 1.0, 60.0, 20.0)

    st.divider()
    progress_pct = st.slider("航行進度 (%)", 0, 100, 0, key="progress_slider")
    st.button("Next Step (+1小時航程)", key="next_step")

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
# 🆕 離岸安全距離：改用「實際公里數」校正，不再假設固定網格解析度
#
# 問題背景：舊版本直接寫死「5個網格」當安全距離，並註解假設 1 格 ≈ 8-9km。
# 但如果 HYCOM 實際網格解析度不同（不同資料源/不同版本可能不一樣），
# 這個假設就會失準，安全距離可能遠小於 40 公里，導致路徑還是貼著海岸走。
#
# 修正：直接用經緯度陣列算出「每格實際大約多少公里」，再反推要多少格
# 才等於我們要的安全公里數。同時把懲罰權重拉高、指數拉高（平方->立方），
# 讓貼岸的代價曲線在接近岸邊時急遽升高。
# ===============================
COAST_SAFE_KM = 18  # 想要的離岸安全距離（公里），可依實際需求調整
_lat_res = float(np.abs(np.diff(lats)).mean()) if len(lats) > 1 else 0.05
_lon_res = float(np.abs(np.diff(lons)).mean()) if len(lons) > 1 else 0.05
CELL_KM = ((_lat_res + _lon_res) / 2) * 111.0
COAST_SAFE_CELLS = max(3.0, COAST_SAFE_KM / max(CELL_KM, 1e-6))
COAST_PENALTY_WEIGHT = 30

def coast_penalty(y, x):
    d = dist_to_land[y, x]
    if d >= COAST_SAFE_CELLS:
        return 0
    # 立方成長：越接近岸邊，代價升高得越陡，比平方更能壓制貼岸繞路
    return ((COAST_SAFE_CELLS - d) ** 3) * COAST_PENALTY_WEIGHT / (COAST_SAFE_CELLS ** 2)

def heuristic(y, x, goal):
    d = np.hypot(lats[y] - lats[goal[0]], lons[x] - lons[goal[1]]) * 111
    return d * SHIP_PARAMS[ship_type_key]['distance_factor'] * 0.9

# ===============================
# 🆕 綜合成本函數 — 加入 elapsed_hours 參數
# 現在會依「船預計幾點會走到這一步」去查對應時間的海流資料，
# 而不是永遠查「現在」這一份快照。
# 回傳 (cost, segment_time_hours)，segment_time_hours 用來累加下一步的 elapsed_hours。
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

    # 🆕 海流投影：改用 elapsed_hours 對應的時間切片，而非固定的「現在」
    current_proj = 0.0
    u_cur, v_cur = get_current_at(y0, x0, elapsed_hours)
    if not np.isnan(u_cur) and not np.isnan(v_cur):
        current_proj = (u_cur * dir_lon + v_cur * dir_lat) * 3.6
    current_bonus = min(max(current_proj, -MAX_CURRENT_BONUS), MAX_CURRENT_BONUS)
    current_cost = -current_bonus * current_gain

    wind_proj = 0.0
    wind_cost = 0.0
    if weather and weather.get("wind"):
        wnd = weather["wind"]
        wi = np.abs(wnd["lats"] - lats[y0]).argmin()
        wj = np.abs(wnd["lons"] - lons[x0]).argmin()
        wind_proj = (float(wnd["u"][wi, wj]) * dir_lon +
                     float(wnd["v"][wi, wj]) * dir_lat) * 3.6
        wind_bonus = min(max(wind_proj, -MAX_WIND_BONUS), MAX_WIND_BONUS)
        wind_cost = -wind_bonus * wind_gain

    wave_cost = 0.0
    wave_slowdown = 1.0
    wave_proj = 0.0
    if weather and weather.get("wave"):
        w = weather["wave"]
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
# 🆕 A* Pathfinding — 同步追蹤「累積航行時間」
# 這是時間相依最短路徑（time-dependent shortest path）的核心：
# 每次鬆弛（relax）節點時，不只更新累積成本 cost[node]，
# 也同步更新「船預計到達這個節點時，從現在起算過了幾小時」elapsed_time[node]，
# 下一步查海流資料時就用這個值去對應正確的時間切片。
# ===============================
dirs = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

def astar(start, goal):
    rows, cols = land_mask.shape
    pq = [(heuristic(start[0], start[1], goal), start)]
    came = {}
    cost = {start: 0}
    elapsed_time = {start: 0.0}  # 🆕 從現在起算的累積航行小時數

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
                    elapsed_time[(ni, nj)] = elapsed_time[cur] + seg_time  # 🆕
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
    with st.spinner("HELIOS 尋路引擎正依據船型特徵與逐時海流變化進行最優解算..."):
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

if st.session_state.get("next_step") and path and st.session_state.ship_step_idx < len(path) - 1:
    remaining_steps = len(path) - 1 - st.session_state.ship_step_idx
    approx_speed = max(ship_speed, 2.0)
    steps_per_hour = max(1, int(remaining_steps / max(1, int(approx_speed / 20))))
    st.session_state.ship_step_idx = min(
        st.session_state.ship_step_idx + steps_per_hour, len(path) - 1
    )

if not path:
    st.stop()

current_pos = path[st.session_state.ship_step_idx]

# ===============================
# 🆕 Calculations — 剩餘距離/時間計算也改用逐時海流
#
# 邏輯：先從路徑起點開始，依序累加每一段的航行時間，算出船抵達
# 目前位置（ship_step_idx）時，從現在算起已經過了幾小時；接著從
# 目前位置繼續往後累加剩餘路段，每一段都用「船屆時預計的時刻」
# 去查對應的海流資料，而不是整段航程都套用「現在」這一份。
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
    u_cur, v_cur = get_current_at(y0, x0, elapsed_hours)  # 🆕 依時間查海流
    if not np.isnan(u_cur) and not np.isnan(v_cur):
        current_proj = (u_cur * dir_lon + v_cur * dir_lat) * 3.6
    current_bonus = min(max(current_proj, -MAX_CURRENT_BONUS), MAX_CURRENT_BONUS)

    wind_proj = 0.0
    if weather and weather.get("wind"):
        wnd = weather["wind"]
        wi = np.abs(wnd["lats"] - lats[y0]).argmin()
        wj = np.abs(wnd["lons"] - lons[x0]).argmin()
        wind_proj = (float(wnd["u"][wi, wj]) * dir_lon +
                     float(wnd["v"][wi, wj]) * dir_lat) * 3.6

    wave_slowdown = 1.0
    if weather and weather.get("wave"):
        w = weather["wave"]
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
    # 先把起點到目前位置的累積航行時間算出來，
    # 這樣才知道「現在」船已經航行了幾小時，剩餘段落要從這個時間點繼續查海流
    elapsed = 0.0
    for i in range(0, idx):
        y0, x0 = path[i]
        y1, x1 = path[i + 1]
        _, seg_time = calc_segment(y0, x0, y1, x1, elapsed)
        elapsed += seg_time

    dist = 0.0
    total_time = 0.0
    for i in range(idx, len(path) - 1):
        y0, x0 = path[i]
        y1, x1 = path[i + 1]
        seg_dist, seg_time = calc_segment(y0, x0, y1, x1, elapsed)
        dist += seg_dist
        total_time += seg_time
        elapsed += seg_time  # 🆕 持續往前推進「船目前預計時刻」

    if idx < len(path) - 1:
        y0, x0 = path[idx]
        y1, x1 = path[idx + 1]
        heading = np.degrees(np.arctan2(lats[y1]-lats[y0], lons[x1]-lons[x0]))
    else:
        heading = 0

    return dist, total_time, heading

remaining_dist, remaining_time, heading = calc_remaining(path, st.session_state.ship_step_idx)

# ===============================
# Dashboard — 第一排
# ===============================
st.subheader("Navigation Dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("Remaining Distance (km)", f"{remaining_dist:.2f}")
c2.metric("Remaining Time (hr)",     f"{remaining_time:.2f}")
c3.metric("Heading",                 f"{heading:.1f}°")

# 第二排 — 氣象資料
w1, w2, w3 = st.columns(3)
if weather and weather.get("wave"):
    w = weather["wave"]
    w1.metric("顯著波高", f"{w['swh']:.2f} m")
    w2.metric("波浪方向", f"{w['dirpw']:.1f}°")
else:
    w1.metric("顯著波高", "N/A")
    w2.metric("波浪方向", "N/A")

if weather and weather.get("wind"):
    wnd = weather["wind"]
    ci = len(wnd["lats"]) // 2
    cj = len(wnd["lons"]) // 2
    spd = float(wnd["speed"][ci, cj])
    w3.metric("風速（中心）", f"{spd:.2f} m/s")
else:
    w3.metric("風速（中心）", "N/A")

st.caption(f"HYCOM observation time (t=0 快照): {obs_time} ｜ 已載入 {len(hycom_times_rel)} 個時間切片，涵蓋未來 {HYCOM_FORECAST_HOURS} 小時")
st.caption(f"離岸安全距離：約 {COAST_SAFE_KM} km（換算約 {COAST_SAFE_CELLS:.1f} 個網格，每格約 {CELL_KM:.1f} km）")

# ===============================
# Map
# ===============================
fig = plt.figure(figsize=(10, 8))
ax  = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118, 124, 21, 26])
ax.add_feature(cfeature.LAND,       facecolor="#b0b0b0")
ax.add_feature(cfeature.COASTLINE)

# 海流（顯示「現在」的快照）
try:
    speed_cur = np.sqrt(hycom_u**2 + hycom_v**2)
    mesh = ax.pcolormesh(lons, lats, speed_cur,
                         cmap="Blues", shading="auto", vmin=0, vmax=1.6,
                         transform=ccrs.PlateCarree())
    fig.colorbar(mesh, ax=ax, label="Current Speed (m/s)")
except:
    st.warning("Could not overlay current data.")

# 波浪等高線
if weather and weather.get("wave"):
    w = weather["wave"]
    wlon_grid, wlat_grid = np.meshgrid(w["lons"], w["lats"])
    contour = ax.contour(
        wlon_grid, wlat_grid, w["swh_grid"],
        levels=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        cmap="cool", linewidths=1.2,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(contour, fmt="%.1fm", fontsize=7, inline=True)

# 風場箭頭
if weather and weather.get("wind"):
    wnd = weather["wind"]
    wlon_g, wlat_g = np.meshgrid(wnd["lons"], wnd["lats"])
    ax.quiver(wlon_g[::2, ::2], wlat_g[::2, ::2],
              wnd["u"][::2, ::2], wnd["v"][::2, ::2],
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

plt.title("HELIOS Navigation Map")
st.pyplot(fig)
