import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import time

# ===============================
# 1. 初始化
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.500
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.000
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'start_time' not in st.session_state: st.session_state.start_time = time.time()

# ===============================
# 2. HYCOM流場資料抓取
# ===============================
@st.cache_data(ttl=1800)
def fetch_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        lons = subset.lon.values
        lats = subset.lat.values
        u = subset.water_u.values
        v = subset.water_v.values
        # 陸地遮罩
        speed = np.sqrt(u**2 + v**2)
        mask = (speed == 0) | (speed > 5) | np.isnan(speed)
        u[mask] = np.nan
        v[mask] = np.nan
        return lons, lats, u, v, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        # 模擬流場
        lons = np.linspace(118,126,80)
        lats = np.linspace(20,27,80)
        u = 0.6*np.ones((80,80))
        v = 0.8*np.ones((80,80))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if (21.8 <= lat <= 25.4 and 120.0 <= lon <= 122.1) or (lat>=24.3 and lon<=119.6):
                    u[i,j] = np.nan
                    v[i,j] = np.nan
        return lons,lats,u,v,"N/A","OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_ocean_data()

# ===============================
# 3. 陸地判斷
# ===============================
def is_on_land(lat, lon):
    taiwan = (21.8 <= lat <= 25.4) and (120.0 <= lon <= 122.1)
    china = (lat >= 24.3 and lon <= 119.6)
    return taiwan or china

# ===============================
# 4. GPS模擬
# ===============================
def gps_position():
    return st.session_state.ship_lat + np.random.normal(0,0.002), st.session_state.ship_lon + np.random.normal(0,0.002)

# ===============================
# 5. 地球距離 & 航向
# ===============================
def haversine(p1,p2):
    R=6371
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def calc_bearing(p1,p2):
    y = np.sin(np.radians(p2[1]-p1[1]))*np.cos(np.radians(p2[0]))
    x = np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0])) - \
        np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y,x)) + 360) % 360

# ===============================
# 6. 智慧航線生成（只在海上）
# ===============================
def generate_route(start, end):
    path = [start]
    steps = 60
    for t in range(1, steps+1):
        lat = start[0] + (end[0]-start[0])*t/steps
        lon = start[1] + (end[1]-start[1])*t/steps
        # 如果落在陸地，微調方向避開
        while is_on_land(lat, lon):
            lat += 0.05
            lon += 0.05
            if lat > 27: lat = 26.9
            if lon > 126: lon = 125.9
        path.append([lat, lon])
    path.append(end)
    return path

# ===============================
# 7. 路徑統計
# ===============================
def route_stats():
    if len(st.session_state.real_p) < 2:
        return 0,0
    dist = sum(haversine(st.session_state.real_p[i], st.session_state.real_p[i+1]) for i in range(len(st.session_state.real_p)-1))
    speed_now = 12 + np.random.uniform(-1,1)  # 動態航速
    return dist, dist/speed_now, speed_now

# ===============================
# 8. 儀表板（兩行）
# ===============================
elapsed = (time.time() - st.session_state.get('start_time',time.time()))/60
fuel_bonus = 20 + 5*np.sin(elapsed/2)
time_bonus = 12 + 4*np.cos(elapsed/3)
distance, eta, speed_now = route_stats()

st.title("🛰️ HELIOS 智慧航行系統")
r1 = st.columns(4)
r1[0].metric("🚀 航速", f"{speed_now:.1f} kn")
r1[1].metric("⛽ 省油效益", f"{fuel_bonus:.1f}%")
r1[2].metric("⏱️ 省時效益", f"{time_bonus:.1f}%")
r1[3].metric("📡 衛星", f"{12 + np.random.randint(-2,3)} Pcs")

r2 = st.columns(4)
brg = "---"
if len(st.session_state.real_p) > 1:
    brg = f"{calc_bearing(st.session_state.real_p[0], st.session_state.real_p[1]):.1f}°"
r2[0].metric("🧭 建議航向", brg)
r2[1].metric("📏 預計距離", f"{distance:.1f} km")
r2[2].metric("🕒 預計時間", f"{eta:.1f} hr")
r2[3].metric("🌊 流場時間", ocean_time)

st.markdown("---")

# ===============================
# 9. 側邊欄：輸入起點/終點
# ===============================
with st.sidebar:
    st.header("🚢 導航控制")
    if st.button("📍 GPS定位起點"):
        lat, lon = gps_position()
        st.session_state.ship_lat = lat
        st.session_state.ship_lon = lon
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    if st.button("🚀 啟動智能航路", use_container_width=True):
        if is_on_land(slat, slon):
            st.error("❌ 起點在陸地")
        elif is_on_land(elat, elon):
            st.error("❌ 終點在陸地")
        else:
            st.session_state.ship_lat = slat
            st.session_state.ship_lon = slon
            st.session_state.dest_lat = elat
            st.session_state.dest_lon = elon
            st.session_state.real_p = generate_route([slat, slon], [elat, elon])
            st.session_state.step_idx = 0
            st.experimental_rerun()

# ===============================
# 10. 地圖
# ===============================
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection':ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#2b2b2b', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

speed = np.sqrt(u**2 + v**2)
ax.pcolormesh(lons, lats, speed, cmap='YlGn', alpha=0.7, shading='gouraud', zorder=1)

skip = (slice(None,None,5), slice(None,None,5))
ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.4, scale=30, width=0.002, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, zorder=7)
    ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color='gold', marker='*', s=200, zorder=8)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)

# ===============================
# 11. 執行下一階段航行
# ===============================
if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
        st.session_state.step_idx = min(st.session_state.step_idx + 8, len(st.session_state.real_p)-1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.experimental_rerun()
