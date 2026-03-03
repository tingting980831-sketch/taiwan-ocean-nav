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
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'start_time' not in st.session_state: st.session_state.start_time = time.time()
if 'dest_lat' not in st.session_state: st.session_state.dest_lat = 22.5
if 'dest_lon' not in st.session_state: st.session_state.dest_lon = 120.0

# ===============================
# 2. HYCOM
# ===============================
@st.cache_data(ttl=1800)
def fetch_hycom():
    try:
        url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds=xr.open_dataset(url,decode_times=False)

        subset=ds.sel(
            lat=slice(20,27),
            lon=slice(118,126),
            depth=0
        ).isel(time=-1).load()

        # 取得HYCOM時間
        hycom_time=str(ds.time.values[-1])

        return subset, hycom_time, "ONLINE"

    except:
        return None,"N/A","OFFLINE"

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
# 5. 地球距離
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
# 6. 航線生成
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
# 7. 路徑統計
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

elapsed=(time.time()-st.session_state.start_time)/60
fuel_bonus=20+5*np.sin(elapsed/2)
time_bonus=12+4*np.cos(elapsed/3)

distance,eta=route_stats()

# ===============================
# 8. 儀表板（兩行）
# ===============================
st.title("🛰️ HELIOS 智慧航行系統")

r1=st.columns(4)
r1[0].metric("🚀 航速","15.8 kn")
r1[1].metric("⛽ 省油效益",f"{fuel_bonus:.1f}%")
r1[2].metric("⏱️ 省時效益",f"{time_bonus:.1f}%")
r1[3].metric("📡 衛星","12 Pcs")

r2=st.columns(4)

brg="---"
if len(st.session_state.real_p)>1:
    brg=f"{calc_bearing(st.session_state.real_p[0],st.session_state.real_p[1]):.1f}°"

r2[0].metric("🧭 建議航向",brg)
r2[1].metric("📏 預計距離",f"{distance:.1f} km")
r2[2].metric("🕒 預計時間",f"{eta:.1f} hr")
r2[3].metric("🌊 流場時間",ocean_time)

st.markdown("---")

# ===============================
# 9. Sidebar
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
# 10. 地圖
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

    # 船
    ax.scatter(st.session_state.ship_lon,
               st.session_state.ship_lat,
               color="red",s=90,zorder=6)

    # ⭐ 終點星星（修復）
    ax.scatter(st.session_state.dest_lon,
               st.session_state.dest_lat,
               color="gold",marker="*",
               s=260,zorder=7)

ax.set_extent([118.5,125.5,20.5,26.5])
st.pyplot(fig)

# ===============================
# 11. 航行模擬
# ===============================
if st.button("🚢 執行下一階段航行",use_container_width=True):

    if st.session_state.real_p and \
       st.session_state.step_idx<len(st.session_state.real_p)-1:

        st.session_state.step_idx+=8
        st.session_state.ship_lat,\
        st.session_state.ship_lon=\
            st.session_state.real_p[st.session_state.step_idx]

        st.rerun()
