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
# Load HYCOM
# ===============================
@st.cache_data(ttl=3600)
def load_hycom():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    try:
        ds = xr.open_dataset(url, decode_times=False)
    except Exception as e:
        st.error(f"無法連接到 HYCOM 數據庫: {e}")
        st.stop()

    if 'time_origin' in ds['time'].attrs:
        origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        obs_time = origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
    else:
        obs_time = pd.Timestamp.now()

    sub = ds.sel(lat=slice(21, 26), lon=slice(118, 124))
    lons = sub.lon.values
    lats = sub.lat.values

    # 預先加載最後一個時間步的海流數據，避免尋路時重複讀取網絡
    u_data = sub['ssu'].isel(time=-1).values
    v_data = sub['ssv'].isel(time=-1).values
    land_mask = np.isnan(u_data)

    return lons, lats, land_mask, obs_time, u_data, v_data

lons, lats, land_mask, obs_time, hycom_u, hycom_v = load_hycom()
sea_mask = ~land_mask
dist_to_land = distance_transform_edt(sea_mask)

# ===============================
# 風場 + 波浪資料
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

        # 🆕 完整波浪方向 grid（用於方向投影，取代單點 dirpw）
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
            "dirpw_grid": dirpw_grid,   # 🆕
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
    st.button("Next Step", key="next_step")

    st.divider()
    st.subheader("🧬 船型自適應環境優化模式")

    # 僅保留專業選項切換，後台自動注入黃金權重
    ship_mode = st.radio(
        "選擇航行船舶類型 (Ship Profile):",
        options=["大型貨輪/油輪 (CargoTanker)", "小型漁船 (Fishing)", "其他公務/客輪 (Other)"],
        index=0,
        help="系統將根據不同船型的流體動力學與環境特徵，自動載入最優化權重矩陣。"
    )

    # 後台靜態注入權重參數（不顯示於前端）
    if "CargoTanker" in ship_mode:
        w_curr = 0.452
        w_wave = 0.431
        w_wind = 0.117
        ship_type_key = "CargoTanker"
    elif "Fishing" in ship_mode:
        w_curr = 0.265
        w_wave = 0.513
        w_wind = 0.222
        ship_type_key = "Fishing"
    else:  # Other
        w_curr = 0.380
        w_wave = 0.410
        w_wind = 0.210
        ship_type_key = "Other"

# ===============================
# 🆕 船型物理參數（移植自文件1 SHIP_PARAMS）
# 註：船速本系統由使用者手動輸入(ship_speed)，故不覆寫 speed
# ===============================
SHIP_PARAMS = {
    'CargoTanker': {
        'distance_factor': 0.3, 'current_gain': 1.0,
        'wind_gain': 1.0, 'wind_speed_gain': 1.0,
        'time_weight': 5.0, 'fuel_weight': 0.20,
        'progress_weight': 0.6, 'wave_coef': 0.08,
    },
    'Fishing': {
        'distance_factor': 0.75, 'current_gain': 0.5,
        'wind_gain': 0.5, 'wind_speed_gain': 0.5,
        'time_weight': 3.0, 'fuel_weight': 0.15,
        'progress_weight': 2.0, 'wave_coef': 0.05,
    },
    'Other': {
        'distance_factor': 0.9, 'current_gain': 0.3,
        'wind_gain': 0.2, 'wind_speed_gain': 0.2,
        'time_weight': 2.0, 'fuel_weight': 0.08,
        'progress_weight': 1.5, 'wave_coef': 0.06,
    },
}
# 波浪嚴重度：貨輪對大浪更敏感（比照文件1 wave_severity 邏輯）
wave_severity = 3.0 if ship_type_key == 'CargoTanker' else 1.0

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

def coast_penalty(y, x):
    d = dist_to_land[y, x]
    return (2 - d) * 2 if d < 2 else 0

# 🆕 綜合成本函數（移植自文件1 get_comprehensive_cost）
# 涵蓋：實際距離(km)、進度懲罰、海流/風/波浪投影、effective_speed、時間成本、燃油成本
def get_comprehensive_cost(y0, x0, y1, x1, goal):
    dlat = lats[y1] - lats[y0]
    dlon = lons[x1] - lons[x0]
    norm = np.hypot(dlat, dlon)
    if norm == 0:
        return 0
    dir_lat = dlat / norm
    dir_lon = dlon / norm
    base_dist = norm * 111  # 度 -> km（近似）

    p = SHIP_PARAMS[ship_type_key]
    distance_factor = p['distance_factor']
    current_gain    = p['current_gain']
    wind_gain       = p['wind_gain']
    wind_speed_gain = p['wind_speed_gain']
    time_weight     = p['time_weight']
    fuel_weight     = p['fuel_weight']
    progress_weight = p['progress_weight']
    wave_coef       = p['wave_coef']

    # 進度懲罰：避免路徑繞遠離目標點
    progress_penalty = 0
    if goal is not None:
        d_before = np.hypot(lats[y0]-lats[goal[0]], lons[x0]-lons[goal[1]]) * 111
        d_after  = np.hypot(lats[y1]-lats[goal[0]], lons[x1]-lons[goal[1]]) * 111
        progress_penalty = max(d_after - d_before, 0) * progress_weight

    # 海流投影
    current_proj = 0.0
    u_cur = float(hycom_u[y0, x0])
    v_cur = float(hycom_v[y0, x0])
    if not np.isnan(u_cur) and not np.isnan(v_cur):
        current_proj = (u_cur * dir_lon + v_cur * dir_lat) * 3.6  # m/s -> km/h
    current_cost = -current_proj * current_gain

    # 風場投影
    wind_proj = 0.0
    wind_cost = 0.0
    if weather and weather.get("wind"):
        wnd = weather["wind"]
        wi = np.abs(wnd["lats"] - lats[y0]).argmin()
        wj = np.abs(wnd["lons"] - lons[x0]).argmin()
        wind_proj = (float(wnd["u"][wi, wj]) * dir_lon +
                     float(wnd["v"][wi, wj]) * dir_lat) * 3.6
        wind_cost = -wind_proj * wind_gain

    # 波浪（含方向投影）
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

    # effective_speed：綜合海流推力／風力／波浪減速
    effective_speed = (ship_speed + current_proj * current_gain +
                        wind_proj * wind_speed_gain * 0.5) / wave_slowdown
    effective_speed = max(effective_speed, 2.0)

    time_cost = (base_dist / effective_speed) * time_weight
    fuel_cost = (40 + 0.5 * effective_speed) * (base_dist / effective_speed) * fuel_weight

    total_cost = (
        base_dist * distance_factor +
        current_cost * w_curr +
        wind_cost * w_wind +
        wave_cost * w_wave +
        time_cost + fuel_cost +
        progress_penalty +
        coast_penalty(y1, x1)
    )
    return max(total_cost, 0.05)

# ===============================
# A* Pathfinding
# ===============================
dirs = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

def astar(start, goal):
    rows, cols = land_mask.shape
    pq = [(0, start)]
    came = {}
    cost = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            break
        for d in dirs:
            ni, nj = cur[0]+d[0], cur[1]+d[1]
            if 0 <= ni < rows and 0 <= nj < cols and not land_mask[ni, nj]:
                # 🆕 綜合成本(距離+進度懲罰+海流/風/波浪+時間/燃油) + 離岸風場懲罰
                step_cost = get_comprehensive_cost(cur[0], cur[1], ni, nj, goal) + offshore_penalty(ni, nj)

                new = cost[cur] + step_cost

                if (ni, nj) not in cost or new < cost[(ni, nj)]:
                    cost[(ni, nj)] = new
                    came[(ni, nj)] = cur
                    heapq.heappush(pq, (new, (ni, nj)))

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
    with st.spinner("HELIOS 尋路引擎正依據船型特徵進行最優解算..."):
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

if st.session_state.get("next_step"):
    if path and st.session_state.ship_step_idx < len(path)-1:
        st.session_state.ship_step_idx += 1

if not path:
    st.stop()

current_pos = path[st.session_state.ship_step_idx]

# ===============================
# Calculations
# 🆕 改用與 get_comprehensive_cost 一致的 effective_speed / wave_slowdown 邏輯
#    （波浪方向投影 + current_gain/wind_speed_gain，取代舊的門檻式減速）
# ===============================
def calc_remaining(path, idx):
    p = SHIP_PARAMS[ship_type_key]
    current_gain    = p['current_gain']
    wind_speed_gain = p['wind_speed_gain']
    wave_coef       = p['wave_coef']

    dist = 0
    total_time = 0

    for i in range(idx, len(path) - 1):
        y0, x0 = path[i]
        y1, x1 = path[i + 1]

        seg_dist = np.hypot(lats[y1]-lats[y0], lons[x1]-lons[x0]) * 111

        dlat = lats[y1] - lats[y0]
        dlon = lons[x1] - lons[x0]
        norm = np.hypot(dlat, dlon)
        if norm == 0:
            continue
        dir_lat = dlat / norm
        dir_lon = dlon / norm

        # 海流投影
        current_proj = 0.0
        u_cur = float(hycom_u[y0, x0])
        v_cur = float(hycom_v[y0, x0])
        if not np.isnan(u_cur) and not np.isnan(v_cur):
            current_proj = (u_cur * dir_lon + v_cur * dir_lat) * 3.6

        # 風場投影
        wind_proj = 0.0
        if weather and weather.get("wind"):
            wnd = weather["wind"]
            wi = np.abs(wnd["lats"] - lats[y0]).argmin()
            wj = np.abs(wnd["lons"] - lons[x0]).argmin()
            wind_proj = (float(wnd["u"][wi, wj]) * dir_lon +
                         float(wnd["v"][wi, wj]) * dir_lat) * 3.6

        # 波浪方向投影 -> 失速/加速倍率
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

        effective_speed = (ship_speed + current_proj * current_gain +
                            wind_proj * wind_speed_gain * 0.5) / wave_slowdown
        effective_speed = max(effective_speed, 2.0)

        dist += seg_dist
        total_time += seg_dist / effective_speed

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

st.caption(f"HYCOM observation time: {obs_time}")

# ===============================
# Map
# ===============================
fig = plt.figure(figsize=(10, 8))
ax  = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([118, 124, 21, 26])
ax.add_feature(cfeature.LAND,       facecolor="#b0b0b0")
ax.add_feature(cfeature.COASTLINE)

# 海流
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
