import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from heapq import heappush, heappop
from scipy.ndimage import distance_transform_edt
import time

# ===============================
# 系統初始化
# ===============================
st.set_page_config(page_title="HELIOS V29 智慧航運版", layout="wide")

if 'ship_lat' not in st.session_state: st.session_state.ship_lat = 25.06
if 'ship_lon' not in st.session_state: st.session_state.ship_lon = 122.2
if 'real_p' not in st.session_state: st.session_state.real_p = []
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0


# ===============================
# HYCOM資料
# ===============================
@st.cache_data(ttl=3600)
def fetch_hycom():
    try:
        url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
        ds=xr.open_dataset(url,decode_times=False)

        subset=ds.sel(
            lat=slice(20,27),
            lon=slice(118,126),
            depth=0
        ).isel(time=-1).load()

        return subset,datetime.now().strftime("%H:%M:%S"),"ONLINE"
    except:
        return None,"N/A","OFFLINE"

ocean_data,data_clock,stream_status=fetch_hycom()


# ===============================
# 建立避障成本圖
# ===============================
def build_cost_map(ocean_data):

    speed=np.sqrt(
        ocean_data.water_u.values**2+
        ocean_data.water_v.values**2
    )

    ocean_mask=np.isfinite(speed)
    dist=distance_transform_edt(ocean_mask)

    cost=np.ones_like(dist,float)

    # 靠岸懲罰
    cost+=np.exp(-dist/3)*40

    cost[~ocean_mask]=np.inf

    return cost,ocean_data.lat.values,ocean_data.lon.values


# ===============================
# 時間成本（省油+省時核心）
# ===============================
def time_cost(i1,j1,i2,j2,cost_map,ocean_data,lats,lons):

    if np.isinf(cost_map[i2,j2]):
        return np.inf

    dy=(lats[i2]-lats[i1])*111
    dx=(lons[j2]-lons[j1])*111*np.cos(np.radians(lats[i1]))

    dist=np.hypot(dx,dy)

    u=ocean_data.water_u.values[i1,j1]
    v=ocean_data.water_v.values[i1,j1]

    norm=np.hypot(dx,dy)+1e-6
    dirx=dx/norm
    diry=dy/norm

    current_along=u*dirx+v*diry

    ship_speed=8.0
    ground_speed=max(1.0,ship_speed+current_along)

    time_cost=dist/ground_speed

    return time_cost*cost_map[i2,j2]


# ===============================
# A* 導航
# ===============================
def astar_route(start,goal,cost,ocean_data,lats,lons):

    h,w=cost.shape

    def heuristic(a,b):
        return np.hypot(a[0]-b[0],a[1]-b[1])

    open_set=[]
    heappush(open_set,(0,start))

    came={}
    g={start:0}

    dirs=[(1,0),(-1,0),(0,1),(0,-1),
          (1,1),(1,-1),(-1,1),(-1,-1)]

    while open_set:

        _,cur=heappop(open_set)

        if cur==goal:
            path=[cur]
            while cur in came:
                cur=came[cur]
                path.append(cur)
            return path[::-1]

        for d in dirs:

            nx,ny=cur[0]+d[0],cur[1]+d[1]

            if nx<0 or ny<0 or nx>=h or ny>=w:
                continue

            step=time_cost(cur[0],cur[1],nx,ny,
                           cost,ocean_data,lats,lons)

            if np.isinf(step): continue

            new_g=g[cur]+step

            if (nx,ny) not in g or new_g<g[(nx,ny)]:
                g[(nx,ny)]=new_g
                f=new_g+heuristic((nx,ny),goal)
                heappush(open_set,(f,(nx,ny)))
                came[(nx,ny)]=cur

    return []


# ===============================
# 路徑平滑
# ===============================
def smooth_path(path,iterations=3):

    pts=np.array(path)

    for _ in range(iterations):
        new=[pts[0]]

        for i in range(len(pts)-1):
            p,q=pts[i],pts[i+1]
            new.append(0.75*p+0.25*q)
            new.append(0.25*p+0.75*q)

        new.append(pts[-1])
        pts=np.array(new)

    return pts.tolist()


# ===============================
# 船舶動態儀表板
# ===============================
def get_current_vector(lat,lon):

    lats=ocean_data.lat.values
    lons=ocean_data.lon.values

    iy=np.abs(lats-lat).argmin()
    ix=np.abs(lons-lon).argmin()

    return float(ocean_data.water_u.values[iy,ix]),\
           float(ocean_data.water_v.values[iy,ix])


def compute_ship_metrics():

    if not st.session_state.real_p:
        return 0,0,0

    base_speed=8.0

    lat=st.session_state.ship_lat
    lon=st.session_state.ship_lon

    u,v=get_current_vector(lat,lon)

    cur=np.hypot(u,v)

    ship_speed=base_speed+cur*0.6
    energy=max(0,(ship_speed-base_speed)/base_speed*100)

    sats=10+int(3*np.sin(st.session_state.step_idx/6))

    return ship_speed*1.94,energy,sats


# ===============================
# 航向
# ===============================
def calc_bearing(p1,p2):

    y=np.sin(np.radians(p2[1]-p1[1]))*np.cos(np.radians(p2[0]))
    x=np.cos(np.radians(p1[0]))*np.sin(np.radians(p2[0]))-\
      np.sin(np.radians(p1[0]))*np.cos(np.radians(p2[0]))*\
      np.cos(np.radians(p2[1]-p1[1]))

    return (np.degrees(np.arctan2(y,x))+360)%360


# ===============================
# 儀表板
# ===============================
spd,energy,sats=compute_ship_metrics()

st.title("🛰️ HELIOS V29 智慧航運系統")

c1,c2,c3,c4,c5=st.columns(5)

c1.metric("🚀 航速",f"{spd:.1f} kn")
c2.metric("⛽ 能源紅利",f"{energy:.1f}%")

brg="---"
if st.session_state.real_p and st.session_state.step_idx<len(st.session_state.real_p)-1:
    brg=f"{calc_bearing(st.session_state.real_p[st.session_state.step_idx],st.session_state.real_p[st.session_state.step_idx+1]):.1f}°"

c3.metric("🧭 建議航向",brg)
c4.metric("📡 衛星",f"{sats} Pcs")
c5.metric("🕒 數據時標",data_clock)

st.markdown("---")

# ===============================
# Sidebar
# ===============================
with st.sidebar:

    st.header("🚢 導航控制器")

    slat=st.number_input("起點緯度",value=st.session_state.ship_lat,format="%.3f")
    slon=st.number_input("起點經度",value=st.session_state.ship_lon,format="%.3f")

    elat=st.number_input("終點緯度",value=22.5,format="%.3f")
    elon=st.number_input("終點經度",value=120.0,format="%.3f")

    if st.button("🚀 啟動智能航路",use_container_width=True):

        cost,lats,lons=build_cost_map(ocean_data)

        def nearest(lat,lon):
            return np.abs(lats-lat).argmin(),np.abs(lons-lon).argmin()

        start=nearest(slat,slon)
        goal=nearest(elat,elon)

        path_idx=astar_route(start,goal,cost,ocean_data,lats,lons)

        if len(path_idx)==0:
            st.error("找不到安全航路")
        else:
            raw=[[lats[i],lons[j]] for i,j in path_idx]
            final_p=smooth_path(raw)

            st.session_state.real_p=final_p
            st.session_state.ship_lat=slat
            st.session_state.ship_lon=slon
            st.session_state.step_idx=0
            st.rerun()


# ===============================
# 地圖
# ===============================
fig,ax=plt.subplots(figsize=(10,8),
    subplot_kw={'projection':ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND,facecolor='#333333')
ax.add_feature(cfeature.COASTLINE,edgecolor='cyan')

if ocean_data is not None:

    lons=ocean_data.lon
    lats=ocean_data.lat

    speed=np.sqrt(ocean_data.water_u**2+ocean_data.water_v**2)

    ax.pcolormesh(lons,lats,speed,cmap='YlGn',alpha=0.6)

if st.session_state.real_p:

    py,px=zip(*st.session_state.real_p)
    ax.plot(px,py,color='#FF00FF',linewidth=3)

    ax.scatter(st.session_state.ship_lon,
               st.session_state.ship_lat,
               color='red',s=80)

    ax.scatter(elon,elat,color='gold',marker='*',s=250)

ax.set_extent([118.5,125.5,20.5,26.5])

st.pyplot(fig)


# ===============================
# 航行動畫
# ===============================
if st.button("🚢 執行下一階段航行",use_container_width=True):

    if st.session_state.real_p and \
       st.session_state.step_idx<len(st.session_state.real_p)-1:

        st.session_state.step_idx=min(
            st.session_state.step_idx+12,
            len(st.session_state.real_p)-1
        )

        st.session_state.ship_lat,\
        st.session_state.ship_lon=\
            st.session_state.real_p[st.session_state.step_idx]

        st.rerun()

time.sleep(0.05)
