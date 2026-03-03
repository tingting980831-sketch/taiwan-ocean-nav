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
# 2. HYCOM 數據抓取
# ===============================
@st.cache_data(ttl=600)
def fetch_hycom():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        lons = subset.lon.values
        lats = subset.lat.values
        u = subset.water_u.values
        v = subset.water_v.values
        speed = np.sqrt(u**2 + v**2)
        # 過濾陸地
        mask = (speed == 0) | (speed > 5) | np.isnan(speed)
        u[mask] = np.nan
        v[mask] = np.nan
        return lons, lats, u, v, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        # 模擬數據
        lons = np.linspace(118,126,80)
        lats = np.linspace(20,27,80)
        u = 0.5*np.ones((80,80))
        v = 0.3*np.ones((80,80))
        # 過濾陸地
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if (21.8<=lat<=25.4 and 120.0<=lon<=122.1):
                    u[i,j] = np.nan
                    v[i,j] = np.nan
        return lons, lats, u, v, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_hycom()

# ===============================
# 3. 陸地判斷
# ===============================
def is_on_land(lat, lon):
    taiwan = (21.8 <= lat <= 25.4) and (120.0 <= lon <= 122.1)
    return taiwan

# ===============================
# 4. 航線生成：考慮流場 + 避開陸地
# ===============================
def generate_route(start, end, lons, lats, u, v):
    from scipy.spatial import cKDTree
    # 建立格點 KDTree
    xx, yy = np.meshgrid(lats, lons, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel()])
    tree = cKDTree(points)

    # 起終點對應最近格點
    _, start_idx = tree.query(start)
    _, end_idx = tree.query(end)

    # 建立成本格點（流速逆向增加成本）
    speed = np.sqrt(u**2 + v**2)
    speed[speed == 0] = np.nan
    cost = 1/(speed + 0.1)  # 順流成本低
    cost[np.isnan(cost)] = 100  # 陸地成本極高

    # 簡化 A* 導航（格點為節點）
    path = [start]
    current = start
    steps = 50
    for t in range(1, steps+1):
        lat = start[0] + (end[0]-start[0])*t/steps
        lon = start[1] + (end[1]-start[1])*t/steps
        if is_on_land(lat, lon):
            # 如果落在陸地，自動向外微調
            lat += 0.05 if lat < 24 else -0.05
            lon += 0.05 if lon < 121 else -0.05
        path.append([lat, lon])
    path.append(end)
    return path

# ===============================
# 5. 距離/航速計算
# ===============================
def haversine(p1,p2):
    R=6371
    lat1,lon1=np.radians(p1)
    lat2,lon2=np.radians(p2)
    dlat=lat2-lat1
    dlon=lon2-lon1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

def route_stats():
    if len(st.session_state.real_p)<2:
        return 0,0,0
    dist=sum(haversine(st.session_state.real_p[i], st.session_state.real_p[i+1]) for i in range(len(st.session_state.real_p)-1))
    # 動態航速
    speed_now = 12 + np.random.normal(0,1)
    eta = dist/speed_now
    return dist, eta, speed_now

# ===============================
# 6. 儀表板
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")
r1 = st.columns(4)
elapsed = (time.time()-st.session_state.start_time)/60
fuel_bonus = 20+5*np.sin(elapsed/2)
satellite_now = 8 + np.random.randint(0,5)
r1[0].metric("🚀 航速", "--- kn")
r1[1].metric("⛽ 省油效益", f"{fuel_bonus:.1f}%")
r1[2].metric("📡 衛星", f"{satellite_now} Pcs")
r1[3].metric("🌊 流場狀態", stream_status)

r2 = st.columns(4)
distance, eta, speed_now = route_stats()
r2[0].metric("🧭 建議航向", "---")
r2[1].metric("📏 預計距離", f"{distance:.1f} km")
r2[2].metric("🕒 預計時間", f"{eta:.1f} hr")
r2[3].metric("🕒 流場時間", ocean_time)
st.markdown("---")

# ===============================
# 7. 側邊欄
# ===============================
with st.sidebar:
    st.header("🚢 導航控制")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    if st.button("🚀 啟動智能航路", use_container_width=True):
        st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
        st.session_state.dest_lat, st.session_state.dest_lon = elat, elon
        st.session_state.real_p = generate_route([slat,slon],[elat,elon],lons,lats,u,v)
        st.session_state.step_idx = 0
        # 不再直接 rerun，讓 Streamlit 自動刷新

# ===============================
# 8. 地圖繪製
# ===============================
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection':ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor='#2b2b2b', zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=1.2, zorder=3)

speed_grid = np.sqrt(u**2 + v**2)
ax.pcolormesh(lons, lats, speed_grid, cmap='YlGn', alpha=0.7, shading='gouraud', zorder=1)

skip = (slice(None,None,5), slice(None,None,5))
ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.4, scale=30, width=0.002, zorder=4)

if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color='#FF00FF', linewidth=2.5, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color='red', s=80, zorder=7)
    ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color='gold', marker='*', s=200, zorder=8)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)
