import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt
import heapq

st.set_page_config(layout="wide", page_title="HELIOS V7")
st.title("🛰️ HELIOS V7 智慧海象導航系統")

# ===============================
# 讀取 HYCOM 海流
# ===============================
@st.cache_data(ttl=3600)
def load_hycom_data():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds = xr.open_dataset(url, decode_times=False)

    # 處理 time_origin 可能不存在
    try:
        time_origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        latest_time = time_origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
    except KeyError:
        latest_time = pd.Timestamp.now()

    lat_slice = slice(21,26)
    lon_slice = slice(118,124)

    u_data = ds['ssu'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)
    v_data = ds['ssv'].sel(lat=lat_slice, lon=lon_slice).isel(time=-1)

    lons = u_data['lon'].values
    lats = u_data['lat'].values

    u_val = np.nan_to_num(u_data.values)
    v_val = np.nan_to_num(v_data.values)
    land_mask = np.isnan(u_data.values)

    return lons, lats, u_val, v_val, land_mask, latest_time

lons, lats, u, v, land_mask, obs_time = load_hycom_data()

# ===============================
# 禁航區
# ===============================
NO_GO_ZONES = [
    # 台南安平/四草
    [[22.953536, 120.171678], [22.934628, 120.175472], [22.933136, 120.170942], [22.957810, 120.160780]],
    [[22.943956, 120.172358], [22.939717, 120.173944], [22.928353, 120.157372], [22.936636, 120.153547]],
    [[22.88462, 120.17547], [22.91242, 120.17696], [22.91178, 120.17214], [22.933136, 120.170942]],
    # 屏東東港
    [[22.360902, 120.381109], [22.354722, 120.382139], [22.353191, 120.384728], [22.357684, 120.387818]],
    [[22.452778, 120.458611], [22.4475, 120.451389], [22.446111, 120.452222], [22.449167, 120.460556]],
    # 澎湖
    [[23.7885, 119.598368], [23.784251, 119.598368], [23.784251, 119.602022], [23.7885, 119.602022]],
    [[23.280833, 119.5], [23.280833, 119.509722], [23.274444, 119.509722], [23.274444, 119.5]],
    # 野柳/基隆/瑞芳
    [[25.231417, 121.648863], [25.226151, 121.651505], [25.233410, 121.642090], [25.242200, 121.634560]],
    [[25.135583, 121.826500], [25.128472, 121.834222], [25.128722, 121.818389], [25.118861, 121.824250]],
    # 東北角 B1-B4
    [[25.103033, 121.920683], [25.101992, 121.922256], [25.098672, 121.919836], [25.099569, 121.918122]],
    # 宜蘭外海
    [[24.8797, 121.8463], [24.8791, 121.8452], [24.8780, 121.8452]],
    # 旗津沿岸海域 A1-A8
    [[22.611569, 120.264303], [22.61055, 120.26386], [22.60902, 120.26382], [22.60799, 120.26434]],
    # 新竹/苗栗
    [[24.849621, 120.928948], [24.848034, 120.929797], [24.847862, 120.930568], [24.850101, 120.930086]],
    [[24.831074, 120.914995], [24.822774, 120.909618], [24.818237, 120.907696], [24.822791, 120.909332]],
]

# ===============================
# 離岸風場
# ===============================
OFFSHORE_WIND = [
    [[24.18, 120.12], [24.22, 120.28], [24.05, 120.35], [24.00, 120.15]],  # 彰化北
    [[24.00, 120.10], [24.05, 120.32], [23.90, 120.38], [23.85, 120.15]],  # 彰化中
    [[23.88, 120.05], [23.92, 120.18], [23.75, 120.25], [23.70, 120.08]],  # 彰化南
    [[23.68, 120.02], [23.72, 120.12], [23.58, 120.15], [23.55, 120.05]],  # 雲林允能
    [[24.75, 120.72], [24.80, 120.85], [24.65, 120.92], [24.60, 120.78]],  # 苗栗/新竹
    [[24.88, 120.85], [24.95, 120.95], [24.82, 121.02], [24.78, 120.90]],  # 新竹北
]

# ===============================
# 側邊欄設定
# ===============================
with st.sidebar:
    st.header("航點設定")
    s_lon = st.number_input("起點經度", 118.0, 124.0, 120.3)
    s_lat = st.number_input("起點緯度", 21.0, 26.0, 22.6)
    e_lon = st.number_input("終點經度", 118.0, 124.0, 122.0)
    e_lat = st.number_input("終點緯度", 21.0, 26.0, 24.5)
    ship_speed = st.number_input("船速 km/h", 1.0, 60.0, 20.0)

# ===============================
# 計算航線 (僅使用安全距離)
# ===============================
def nearest_ocean_cell(lon, lat, lons, lats, land_mask):
    lon_idx = np.abs(lons - lon).argmin()
    lat_idx = np.abs(lats - lat).argmin()
    if not land_mask[lat_idx, lon_idx]:
        return lat_idx, lon_idx
    ocean = np.where(~land_mask)
    dist = np.sqrt((lats[ocean[0]] - lat)**2 + (lons[ocean[1]] - lon)**2)
    i = dist.argmin()
    return ocean[0][i], ocean[1][i]

# ===============================
# 2D 地圖繪製
# ===============================
if lons is not None:

    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([118,124,21,26])
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)

    # 海流顏色層
    colors = ["#E5F0FF","#CCE0FF","#99C2FF","#66A3FF",
              "#3385FF","#0066FF","#0052CC","#003D99",
              "#002966","#001433","#000E24"]
    cmap = mcolors.LinearSegmentedColormap.from_list("flow", colors)
    total_factor = np.sqrt(u**2 + v**2)
    im = ax.pcolormesh(lons, lats, total_factor, cmap=cmap, shading='auto', alpha=0.8)
    plt.colorbar(im, ax=ax, label="海流強度")

    # 畫禁航區
    for zone in NO_GO_ZONES:
        poly = np.array(zone)
        ax.fill(poly[:,1], poly[:,0], color='red', alpha=0.5, zorder=5)

    # 畫離岸風場
    for zone in OFFSHORE_WIND:
        poly = np.array(zone)
        ax.fill(poly[:,1], poly[:,0], color='yellow', alpha=0.5, zorder=4)

    # 起點與終點
    ax.scatter(s_lon, s_lat, color='green', s=120, edgecolors='black', zorder=6)
    ax.scatter(e_lon, e_lat, color='yellow', marker='*', s=200, edgecolors='black', zorder=6)

    plt.title("HELIOS V7 - 2D 海流 + 禁航區 + 離岸風場")
    st.pyplot(fig)
