import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from datetime import datetime
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

# ===============================
# 2. 載入真實世界陸地資料
# ===============================
@st.cache_resource
def load_land_geometry():

    land_shp = shpreader.natural_earth(
        resolution='10m',
        category='physical',
        name='land'
    )

    reader = shpreader.Reader(land_shp)
    geoms = list(reader.geometries())

    return unary_union(geoms)

LAND = load_land_geometry()

# ===============================
# 3. 陸地判斷
# ===============================
def is_on_land(lat, lon):
    point = Point(lon, lat)
    return LAND.contains(point)

# ===============================
# 4. HYCOM 即時資料
# ===============================
@st.cache_data(ttl=1800)
def fetch_hycom():

    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest"

        ds = xr.open_dataset(url)

        subset = ds.sel(
            lat=slice(20,27),
            lon=slice(118,126),
            depth=0
        ).isel(time=0).load()

        hycom_time = str(ds.time.values[0])

        return subset, hycom_time, "ONLINE"

    except:

        return None, "N/A", "OFFLINE"

ocean_data, ocean_time, stream_status = fetch_hycom()

# ===============================
# 5. GPS 模擬
# ===============================
def gps_position():

    return (
        st.session_state.ship_lat + np.random.normal(0,0.002),
        st.session_state.ship_lon + np.random.normal(0,0.002)
    )

# ===============================
# 6. 地球距離
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
      np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*\
      np.cos(np.radians(p2[1]-p1[1]))

    return (np.degrees(np.arctan2(y,x))+360)%360

# ===============================
# 7. 智能航線（避免陸地）
# ===============================
def smart_route(start,end):

    line = LineString([
        (start[1],start[0]),
        (end[1],end[0])
    ])

    if line.intersects(LAND):
        return None,"航線會穿越陸地，請重新設定"

    final=[]

    for t in np.linspace(0,1,120):

        lat=start[0]+(end[0]-start[0])*t
        lon=start[1]+(end[1]-start[1])*t

        final.append([lat,lon])

    return final,None

# ===============================
# 8. 路徑統計
# ===============================
def route_stats():

    if len(st.session_state.real_p)<2:
        return 0,0

    dist=sum(
        haversine(st.session_state.real_p[i],
                  st.session_state.real_p[i+1])
        for i in range(len(st.session_state.real_p)-1)
    )

    speed=15.8

    return dist,dist/speed

distance,eta=route_stats()

# ===============================
# 9. 儀表板
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")

r1=st.columns(4)

r1[0].metric("🚀 航速","15.8 kn")
r1[1].metric("📏 預計距離",f"{distance:.1f} km")
r1[2].metric("🕒 預計時間",f"{eta:.1f} hr")
r1[3].metric("🌊 HYCOM時間",ocean_time)

st.markdown(f"資料狀態：{stream_status}")

st.markdown("---")

# ===============================
# 10. Sidebar
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

            if msg:

                st.error(msg)

            else:

                st.session_state.real_p=path
                st.session_state.step_idx=0
                st.session_state.ship_lat=slat
                st.session_state.ship_lon=slon

                st.rerun()

# ===============================
# 11. 地圖
# ===============================
fig,ax=plt.subplots(figsize=(10,8),
        subplot_kw={'projection':ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND,facecolor="#333333")
ax.add_feature(cfeature.COASTLINE,color="cyan")

if ocean_data is not None:

    speed=np.sqrt(ocean_data.water_u**2+ocean_data.water_v**2)

    ax.pcolormesh(
        ocean_data.lon,
        ocean_data.lat,
        speed,
        cmap="YlGn",
        alpha=0.6
    )

if st.session_state.real_p:

    py,px=zip(*st.session_state.real_p)

    ax.plot(px,py,color="#FF00FF",linewidth=3)

    ax.scatter(
        st.session_state.ship_lon,
        st.session_state.ship_lat,
        color="red",
        s=90,
        zorder=6
    )

    ax.scatter(
        st.session_state.dest_lon,
        st.session_state.dest_lat,
        color="gold",
        marker="*",
        s=260,
        zorder=7
    )

ax.set_extent([118.5,125.5,20.5,26.5])

st.pyplot(fig)

# ===============================
# 12. 航行模擬
# ===============================
if st.button("🚢 執行下一階段航行",use_container_width=True):

    if st.session_state.real_p and \
       st.session_state.step_idx<len(st.session_state.real_p)-1:

        st.session_state.step_idx+=8

        st.session_state.ship_lat,\
        st.session_state.ship_lon=\
            st.session_state.real_p[st.session_state.step_idx]

        st.rerun()
