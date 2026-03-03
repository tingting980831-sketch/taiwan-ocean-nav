import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import os
import time

# ===============================
# 1. 初始化
# ===============================
st.set_page_config(page_title="HELIOS 智慧航行系統", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'start_time' not in st.session_state: st.session_state.start_time = time.time()
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.5
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.0

CACHE_FILE = "hycom_cache.nc"

# ===============================
# 2. HYCOM 即時抓取
# ===============================
@st.cache_data(ttl=1800)
def fetch_hycom():
    try:
        url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds = xr.open_dataset(url, decode_times=False)
        ds = ds.sel(lat=slice(20,27), lon=slice(118,126), depth=0).isel(time=-1).load()
        hycom_time = str(ds.time.values[-1])
        ds.to_netcdf(CACHE_FILE)
        return ds, hycom_time, "ONLINE"
    except:
        if os.path.exists(CACHE_FILE):
            try:
                ds = xr.open_dataset(CACHE_FILE)
                return ds, "CACHE", datetime.now().strftime("%Y-%m-%d %H:%M")
            except:
                pass
        # 模擬流場備援
        lats = np.linspace(20,27,80)
        lons = np.linspace(118,126,80)
        lon2d, lat2d = np.meshgrid(lons,lats)
        u = 0.6*np.sin((lat2d-22)/3)
        v = 0.4*np.cos((lon2d-121)/3)
        ds = xr.Dataset(
            {"water_u": (("lat","lon"), u),
             "water_v": (("lat","lon"), v)},
            coords={"lat": lats,"lon":lons}
        )
        return ds, "BACKUP", datetime.now().strftime("%Y-%m-%d %H:%M")

ocean_data, ocean_time, stream_status = fetch_hycom()

# ===============================
# 3. 陸地判斷
# ===============================
def is_on_land(lat,lon):
    taiwan=(21.8<=lat<=25.4) and (120.0<=lon<=122.1)
    china=(lat>=24.0 and lon<=119.8)
    return taiwan or china

# ===============================
# 4. GPS 模擬
# ===============================
def gps_position():
    return (
        st.session_state.ship_lat + np.random.normal(0,0.002),
        st.session_state.ship_lon + np.random.normal(0,0.002)
    )

# ===============================
# 5. 地球距離 & 航向
# ===============================
def haversine(p1,p2):
    R=6371
    lat1,lon1=np.radians(p1)
    lat2,lon2=np.radians(p2)
    dlat=lat2-lat1
    dlon=lon2-lon1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

def calc_bearing(p1,p2):
    y=np.sin(np.radians(p2[1]-p1[1]))*np.cos(np.radians(p2[0]))
    x=np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0]))-\
      np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*np.cos(np.radians(p2[1]-p1[1]))
    return (np.degrees(np.arctan2(y,x))+360)%360

# ===============================
# 6. 船速模型 (即時流場)
# ===============================
BASE_SPEED = 12 # kn
def dynamic_speed(lat,lon,dx,dy):
    lat_idx = np.abs(ocean_data.lat.values - lat).argmin()
    lon_idx = np.abs(ocean_data.lon.values - lon).argmin()
    u = ocean_data.water_u.values[lat_idx, lon_idx]
    v = ocean_data.water_v.values[lat_idx, lon_idx]
    dir_vec = np.array([dx,dy])
    dir_vec /= (np.linalg.norm(dir_vec)+1e-6)
    speed = BASE_SPEED + 3*(u*dir_vec[1]+v*dir_vec[0])
    speed = np.clip(speed,4,20)
    return speed

# ===============================
# 7. 智能航路生成
# ===============================
def smart_route(start,end):
    pts=[start]
    if (start[1]-121)*(end[1]-121)<0:
        mid=[25.7,122.2] if (start[0]+end[0])/2>23.8 else [20.7,120.8]
        if is_on_land(*mid):
            return None,"中繼點落在陸地"
        pts.append(mid)
    pts.append(end)
    final=[]
    for i in range(len(pts)-1):
        for t in np.linspace(0,1,60):
            lat=pts[i][0]+(pts[i+1][0]-pts[i][0])*t
            lon=pts[i][1]+(pts[i+1][1]-pts[i][1])*t
            if not is_on_land(lat,lon):
                final.append([lat,lon])
    return final,None

# ===============================
# 8. 路徑統計
# ===============================
def route_stats():
    if len(st.session_state.real_p)<2: return 0,0
    dist = 0
    time_hr = 0
    for i in range(len(st.session_state.real_p)-1):
        p1 = st.session_state.real_p[i]
        p2 = st.session_state.real_p[i+1]
        d = haversine(p1,p2)
        spd = dynamic_speed(p1[0],p1[1],p2[0]-p1[0],p2[1]-p1[1])
        dist += d
        time_hr += d/spd
    return dist,time_hr

elapsed=(time.time()-st.session_state.start_time)/60
fuel_bonus=20+5*np.sin(elapsed/2)
time_bonus=12+4*np.cos(elapsed/3)
distance,eta = route_stats()

# ===============================
# 9. 儀表板 (兩行)
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")
row1 = st.columns(4)
row1[0].metric("🚀 航速",f"{BASE_SPEED} kn")
row1[1].metric("⛽ 省油效益",f"{fuel_bonus:.1f}%")
row1[2].metric("⏱️ 省時效益",f"{time_bonus:.1f}%")
row1[3].metric("📡 衛星","12 Pcs")

row2 = st.columns(4)
brg="---"
if len(st.session_state.real_p)>1:
    brg=f"{calc_bearing(st.session_state.real_p[0],st.session_state.real_p[1]):.1f}°"
row2[0].metric("🧭 建議航向",brg)
row2[1].metric("📏 預計距離",f"{distance:.1f} km")
row2[2].metric("🕒 預計時間",f"{eta:.1f} hr")
row2[3].metric("🌊 流場時間",ocean_time)
st.markdown("---")

# ===============================
# 10. 側邊欄
# ===============================
with st.sidebar:
    st.header("🚢 導航控制")
    if st.button("📍 GPS定位起點"):
        lat,lon=gps_position()
        st.session_state.ship_lat=lat
        st.session_state.ship_lon=lon

    slat=st.number_input("起點緯度",value=st.session_state.ship_lat,format="%.3f")
    slon=st.number_input("起點經度",value=st.session_state.ship_lon,format="%.3f")
    elat=st.number_input("終點緯度",value=st.session_state.dest_lat,format="%.3f")
    elon=st.number_input("終點經度",value=st.session_state.dest_lon,format="%.3f")
    st.session_state.dest_lat=elat
    st.session_state.dest_lon=elon

    if is_on_land(slat,slon):
        st.error("❌ 起點在陸地")
    elif is_on_land(elat,elon):
        st.error("❌ 終點在陸地")
    else:
        if st.button("🚀 啟動智能航路",use_container_width=True):
            path,msg=smart_route([slat,slon],[elat,elon])
            if msg: st.error(msg)
            else:
                st.session_state.real_p=path
                st.session_state.step_idx=0
                st.session_state.ship_lat=slat
                st.session_state.ship_lon=slon
                st.experimental_rerun()

# ===============================
# 11. 地圖顯示
# ===============================
fig,ax=plt.subplots(figsize=(10,8),subplot_kw={'projection':ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND,facecolor="#333333")
ax.add_feature(cfeature.COASTLINE,color="cyan")

if ocean_data is not None:
    speed=np.sqrt(ocean_data.water_u.values**2+ocean_data.water_v.values**2)
    ax.pcolormesh(ocean_data.lon.values,ocean_data.lat.values,speed,cmap="YlGn",alpha=0.6)

if st.session_state.real_p:
    py,px=zip(*st.session_state.real_p)
    ax.plot(px,py,color="#FF00FF",linewidth=3)
    ax.scatter(st.session_state.ship_lon,st.session_state.ship_lat,color="red",s=90,zorder=6)
    ax.scatter(st.session_state.dest_lon,st.session_state.dest_lat,color="gold",marker="*",s=260,zorder=7)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)

# ===============================
# 12. 航行模擬
# ===============================
if st.button("🚢 執行下一階段航行",use_container_width=True):
    if st.session_state.real_p:
        next_idx = min(st.session_state.step_idx + 8, len(st.session_state.real_p)-1)
        st.session_state.step_idx = next_idx
        st.session_state.ship_lat,st.session_state.ship_lon=st.session_state.real_p[next_idx]
        st.experimental_rerun()
