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
# 2. 真實陸地幾何
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

LAND_GEOM = load_land_geometry()

def is_on_land(lat,lon):
    return LAND_GEOM.contains(Point(lon,lat))

# ===============================
# 3. HYCOM 最新流場（時間正確解析）
# ===============================
@st.cache_data(ttl=900)
def fetch_hycom():

    try:

        url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest"

        ds=xr.open_dataset(url,decode_times=True)

        subset=ds.sel(
            lat=slice(20,27),
            lon=slice(118,126),
            depth=0
        ).isel(time=-1).load()

        latest_time = ds.time.values[-1]

        # 轉成 Python datetime
        hycom_time = np.datetime_as_string(latest_time, unit='m')

        st.session_state["backup_ocean"]=subset
        st.session_state["backup_time"]=hycom_time

        return subset,hycom_time,"ONLINE"

    except:

        if "backup_ocean" in st.session_state:

            return (
                st.session_state["backup_ocean"],
                st.session_state["backup_time"],
                "BACKUP"
            )

        # 建立安全假流場
        lat=np.linspace(20,27,40)
        lon=np.linspace(118,126,60)

        fake=xr.Dataset(
            {
                "water_u": (("lat","lon"),np.zeros((40,60))),
                "water_v": (("lat","lon"),np.zeros((40,60))),
            },
            coords={"lat":lat,"lon":lon}
        )

        now=datetime.utcnow().strftime("%Y-%m-%d %H:%M")

        return fake,now,"SIMULATED"

ocean_data,ocean_time,stream_status=fetch_hycom()

# ===============================
# 4. 地球距離
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
# 5. 航線（禁止穿越陸地）
# ===============================
def smart_route(start,end):

    line=LineString([(start[1],start[0]),(end[1],end[0])])

    if line.intersects(LAND_GEOM):
        return None,"航線會穿越陸地"

    final=[]
    for t in np.linspace(0,1,120):
        lat=start[0]+(end[0]-start[0])*t
        lon=start[1]+(end[1]-start[1])*t
        final.append([lat,lon])

    return final,None

# ===============================
# 6. 路徑統計
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
# 7. 儀表板（完全沒動）
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")

r1=st.columns(4)
r1[0].metric("🚀 航速","15.8 kn")
r1[1].metric("📏 預計距離",f"{distance:.1f} km")
r1[2].metric("🕒 預計時間",f"{eta:.1f} hr")
r1[3].metric("🌊 流場時間",ocean_time)

st.markdown("---")

# ===============================
# 8. Sidebar
# ===============================
with st.sidebar:

    st.header("🚢 導航控制")

    slat=st.number_input("起點緯度",value=st.session_state.ship_lat,format="%.3f")
    slon=st.number_input("起點經度",value=st.session_state.ship_lon,format="%.3f")

    elat=st.number_input("終點緯度",value=st.session_state.dest_lat,format="%.3f")
    elon=st.number_input("終點經度",value=st.session_state.dest_lon,format="%.3f")

    st.session_state.dest_lat=elat
    st.session_state.dest_lon=elon

    if st.button("🚀 啟動智能航路",use_container_width=True):

        path,msg=smart_route([slat,slon],[elat,elon])

        if msg:
            st.error(msg)
        else:
            st.session_state.real_p=path
            st.session_state.ship_lat=slat
            st.session_state.ship_lon=slon
            st.rerun()

# ===============================
# 9. 地圖
# ===============================
fig,ax=plt.subplots(figsize=(10,8),
        subplot_kw={'projection':ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND,facecolor="#333333")
ax.add_feature(cfeature.COASTLINE,color="cyan")

if ocean_data is not None:

    speed=np.sqrt(ocean_data.water_u**2+ocean_data.water_v**2)

    ax.pcolormesh(ocean_data.lon,ocean_data.lat,speed,cmap="YlGn",alpha=0.6)

if st.session_state.real_p:

    py,px=zip(*st.session_state.real_p)

    ax.plot(px,py,color="#FF00FF",linewidth=3)

    ax.scatter(st.session_state.ship_lon,
               st.session_state.ship_lat,
               color="red",s=90,zorder=6)

    ax.scatter(st.session_state.dest_lon,
               st.session_state.dest_lat,
               color="gold",marker="*",
               s=260,zorder=7)

ax.set_extent([118.5,125.5,20.5,26.5])

st.pyplot(fig)
