import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import heapq
import time

# ===============================
# 1. 初始化
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

# session state
if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.500
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.000
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'start_time' not in st.session_state: st.session_state.start_time = time.time()

# ===============================
# 2. 取得HYCOM流場資料
# ===============================
@st.cache_data(ttl=1800)
def fetch_ocean_data():
    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        subset = ds.sel(lat=slice(20.0, 27.0), lon=slice(118.0, 126.0), depth=0).isel(time=-1).load()

        lons = subset.lon.values
        lats = subset.lat.values
        u = subset.water_u.values
        v = subset.water_v.values
        speed = np.sqrt(u**2 + v**2)
        # 陸地遮罩
        mask = (speed==0) | (speed>5) | np.isnan(speed)
        u[mask] = np.nan
        v[mask] = np.nan

        return lons, lats, u, v, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        # 模擬流場資料
        lons = np.linspace(118, 126, 80)
        lats = np.linspace(20, 27, 80)
        u = 0.6*np.ones((80,80))
        v = 0.8*np.ones((80,80))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if (21.8<=lat<=25.4 and 120.0<=lon<=122.1) or (lat>=24.3 and lon<=119.6):
                    u[i,j]=np.nan
                    v[i,j]=np.nan
        return lons, lats, u, v, "N/A", "OFFLINE"

lons, lats, u, v, ocean_time, stream_status = fetch_ocean_data()
speed = np.sqrt(u**2 + v**2)

# ===============================
# 3. 陸地判斷
# ===============================
def is_on_land(lat, lon):
    taiwan = (21.8<=lat<=25.4) and (120.0<=lon<=122.1)
    china = (lat>=24.3 and lon<=119.6)
    return taiwan or china

# ===============================
# 4. 距離與航向
# ===============================
def haversine(p1,p2):
    R=6371
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlat = lat2-lat1
    dlon = lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def calc_bearing(p1,p2):
    y=np.sin(np.radians(p2[1]-p1[1]))*np.cos(np.radians(p2[0]))
    x=np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0]))-\
      np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y,x))+360)%360

# ===============================
# 5. A*智慧海上航線生成
# ===============================
def astar_route(start, end, lons, lats, speed):
    # 建立成本網格 (流速快 = 成本低)
    cost = np.zeros_like(speed)
    cost[:] = np.inf
    mask = ~np.isnan(speed)
    cost[mask] = 1/(speed[mask]+0.1) # 防止除零

    # 將經緯度映射到格點索引
    lon_idx = {lon:j for j, lon in enumerate(lons)}
    lat_idx = {lat:i for i, lat in enumerate(lats)}
    def nearest_idx(value, array):
        return np.argmin(np.abs(array-value))
    
    start_idx = (nearest_idx(start[0], lats), nearest_idx(start[1], lons))
    end_idx = (nearest_idx(end[0], lats), nearest_idx(end[1], lons))

    # A*演算法
    visited = np.full(cost.shape, False)
    heap = [(0, start_idx, [])]

    while heap:
        c, (i,j), path = heapq.heappop(heap)
        if visited[i,j]: continue
        visited[i,j]=True
        path = path+[(i,j)]
        if (i,j)==end_idx:
            # 轉換回經緯度
            return [[lats[ii], lons[jj]] for ii,jj in path]
        # 四鄰居
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                ni, nj = i+di, j+dj
                if 0<=ni<cost.shape[0] and 0<=nj<cost.shape[1]:
                    if not visited[ni,nj] and np.isfinite(cost[ni,nj]):
                        # 簡單啟發函數 (haversine)
                        h = haversine([lats[ni],lons[nj]],[lats[end_idx[0]],lons[end_idx[1]]])
                        heapq.heappush(heap,(c+cost[ni,nj]+h,(ni,nj),path))
    return None

# ===============================
# 6. 路徑統計
# ===============================
def route_stats():
    if len(st.session_state.real_p)<2: return 0,0,0
    dist = sum(haversine(st.session_state.real_p[i], st.session_state.real_p[i+1])
               for i in range(len(st.session_state.real_p)-1))
    # 動態航速
    speed_now = 12 + 2*np.sin((time.time()-st.session_state.start_time)/600)
    eta = dist/speed_now
    return dist, eta, speed_now

distance, eta, speed_now = route_stats()

# ===============================
# 7. 儀表板 (兩行)
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")
r1=st.columns(4)
r1[0].metric("🚀 航速", f"{speed_now:.1f} kn")
r1[1].metric("⛽ 省油效益", f"{20+5*np.sin((time.time()-st.session_state.start_time)/600):.1f}%")
r1[2].metric("📡 衛星", f"{8+4*np.cos((time.time()-st.session_state.start_time)/400):.0f} Pcs")
r1[3].metric("🌊 流場狀態", stream_status)

r2=st.columns(4)
brg="---"
if len(st.session_state.real_p)>1:
    brg=f"{calc_bearing(st.session_state.real_p[0],st.session_state.real_p[1]):.1f}°"
r2[0].metric("🧭 建議航向", brg)
r2[1].metric("📏 預計距離", f"{distance:.1f} km")
r2[2].metric("🕒 預計時間", f"{eta:.1f} hr")
r2[3].metric("🕒 流場時間", ocean_time)

st.markdown("---")

# ===============================
# 8. 側邊欄控制
# ===============================
with st.sidebar:
    st.header("🚢 導航控制中心")
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    if st.button("🚀 啟動智能航路", use_container_width=True):
        # 避免落在陸地
        if is_on_land(slat, slon):
            st.error("❌ 起點在陸地")
        elif is_on_land(elat, elon):
            st.error("❌ 終點在陸地")
        else:
            path = astar_route([slat, slon],[elat, elon], lons, lats, speed)
            if path is None:
                st.error("❌ 無法生成路徑")
            else:
                st.session_state.real_p = path
                st.session_state.ship_lat = slat
                st.session_state.ship_lon = slon
                st.session_state.dest_lat = elat
                st.session_state.dest_lon = elon
                st.session_state.step_idx = 0
                st.experimental_rerun()

# ===============================
# 9. 地圖繪製
# ===============================
fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection':ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, facecolor="#2b2b2b", zorder=2)
ax.add_feature(cfeature.COASTLINE, edgecolor="cyan", linewidth=1.2, zorder=3)

# 流場顯示
ax.pcolormesh(lons,lats,speed,cmap="YlGn", alpha=0.6, shading='gouraud', zorder=1)
skip = (slice(None,None,5), slice(None,None,5))
ax.quiver(lons[skip[1]], lats[skip[0]], u[skip], v[skip], color='white', alpha=0.4, scale=30, width=0.002, zorder=2)

# 航線顯示
if st.session_state.real_p:
    py, px = zip(*st.session_state.real_p)
    ax.plot(px, py, color="#FF00FF", linewidth=2.5, zorder=5)
    ax.scatter(st.session_state.ship_lon, st.session_state.ship_lat, color="red", s=80, zorder=6)
    ax.scatter(st.session_state.dest_lon, st.session_state.dest_lat, color="gold", marker="*", s=200, zorder=7)

ax.set_extent([118.5, 125.5, 20.5, 26.5])
st.pyplot(fig)

# ===============================
# 10. 執行下一階段
# ===============================
if st.button("🚢 執行下一階段航行", use_container_width=True):
    if st.session_state.real_p and st.session_state.step_idx < len(st.session_state.real_p)-1:
        st.session_state.step_idx = min(st.session_state.step_idx+5, len(st.session_state.real_p)-1)
        st.session_state.ship_lat, st.session_state.ship_lon = st.session_state.real_p[st.session_state.step_idx]
        st.experimental_rerun()
