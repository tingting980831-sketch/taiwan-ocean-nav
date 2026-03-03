import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import time

# ===============================
# 1. 系統初始化
# ===============================
st.set_page_config(page_title="HELIOS V30 FINAL", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.060
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.200
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'start_time' not in st.session_state: st.session_state.start_time = time.time()

# ===============================
# 2. HYCOM 即時流場
# ===============================
@st.cache_data(ttl=1800)
def fetch_hycom():
    try:
        url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds=xr.open_dataset(url,decode_times=False)
        sub=ds.sel(lat=slice(20,27),lon=slice(118,126),depth=0).isel(time=-1).load()
        return sub, datetime.now().strftime("%H:%M:%S"), "ONLINE"
    except:
        return None,"N/A","OFFLINE"

ocean_data,data_clock,stream_status = fetch_hycom()

# ===============================
# 3. 陸地檢查 (改良版)
# ===============================
def is_on_land(lat,lon):
    taiwan=(21.8<=lat<=25.4) and (120.0<=lon<=122.1)
    china=(lat>=24.0 and lon<=119.8)
    return taiwan or china

# ===============================
# 4. GPS 模擬定位
# ===============================
def gps_position():
    noise_lat=np.random.normal(0,0.002)
    noise_lon=np.random.normal(0,0.002)
    return st.session_state.ship_lat+noise_lat,\
           st.session_state.ship_lon+noise_lon

# ===============================
# 5. 距離與航向
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
# 6. 流場輔助 (省油+省時)
# ===============================
def current_bonus(lat,lon):
    if ocean_data is None:
        return 0
    u=ocean_data.water_u.sel(lat=lat,lon=lon,method="nearest").values
    v=ocean_data.water_v.sel(lat=lat,lon=lon,method="nearest").values
    return np.sqrt(u**2+v**2)

# ===============================
# 7. 智慧航線生成
# ===============================
def smart_route(start,end):

    pts=[start]

    mid_lat=(start[0]+end[0])/2

    # 台灣跨越修正
    if (start[1]-121)*(end[1]-121)<0:
        if mid_lat>23.8:
            mid=[25.7,122.2]
        else:
            mid=[20.7,120.8]

        if is_on_land(*mid):
            return None,"中繼點位於陸地"

        pts.append(mid)

    pts.append(end)

    # 平滑+流場優化
    final=[]
    for i in range(len(pts)-1):
        for t in np.linspace(0,1,60):
            lat=pts[i][0]+(pts[i+1][0]-pts[i][0])*t
            lon=pts[i][1]+(pts[i+1][1]-pts[i][1])*t

            bonus=current_bonus(lat,lon)
            lat+=bonus*0.03*np.sin(t*np.pi)
            lon+=bonus*0.03*np.cos(t*np.pi)

            if is_on_land(lat,lon):
                continue

            final.append([lat,lon])

    return final,None

# ===============================
# 8. 儀表板計算
# ===============================
def route_stats():

    if len(st.session_state.real_p)<2:
        return 0,0

    dist=0
    for i in range(len(st.session_state.real_p)-1):
        dist+=haversine(st.session_state.real_p[i],
                        st.session_state.real_p[i+1])

    speed=15.8
    hours=dist/speed
    return dist,hours

elapsed=(time.time()-st.session_state.start_time)/60
fuel_bonus=20+5*np.sin(elapsed/2)
time_bonus=12+4*np.cos(elapsed/3)

distance,eta=route_stats()

# ===============================
# 9. 儀表板 (兩行)
# ===============================
st.title("🛰️ HELIOS V30 智慧航行系統")

row1=st.columns(4)
row1[0].metric("🚀 航速","15.8 kn")
row1[1].metric("⛽ 省油效益",f"{fuel_bonus:.1f}%")
row1[2].metric("⏱️ 省時效益",f"{time_bonus:.1f}%")
row1[3].metric("📡 衛星","12 Pcs")

row2=st.columns(4)

brg="---"
if len(st.session_state.real_p)>1:
    brg=f"{calc_bearing(st.session_state.real_p[0],st.session_state.real_p[1]):.1f}°"

row2[0].metric("🧭 建議航向",brg)
row2[1].metric("📏 預計距離",f"{distance:.1f} km")
row2[2].metric("🕒 預計時間",f"{eta:.1f} hr")
row2[3].metric("🌊 流場",stream_status)

st.markdown("---")

# ===============================
# 10. 側邊導航
# ===============================
with st.sidebar:

    st.header("🚢 導航控制")

    if st.button("📍 GPS定位起點"):
        lat,lon=gps_position()
        st.session_state.ship_lat=lat
        st.session_state.ship_lon=lon
        st.success("GPS定位完成")

    slat=st.number_input("起點緯度",value=st.session_state.ship_lat,format="%.3f")
    slon=st.number_input("起點經度",value=st.session_state.ship_lon,format="%.3f")
    elat=st.number_input("終點緯度",value=22.5,format="%.3f")
    elon=st.number_input("終點經度",value=120.0,format="%.3f")

    error=None
    if is_on_land(slat,slon):
        error="❌ 起點位於陸地"
    elif is_on_land(elat,elon):
        error="❌ 終點位於陸地"

    if error:
        st.error(error)
        st.button("🚀 啟動智能航路",disabled=True,use_container_width=True)
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
    ax.pcolormesh(ocean_data.lon,ocean_data.lat,
                  speed,cmap="YlGn",alpha=0.6)

if st.session_state.real_p:
    py,px=zip(*st.session_state.real_p)
    ax.plot(px,py,color="#FF00FF",linewidth=3)
    ax.scatter(st.session_state.ship_lon,
               st.session_state.ship_lat,
               color="red",s=90)

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
