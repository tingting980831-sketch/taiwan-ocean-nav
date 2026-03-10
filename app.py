import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import distance_transform_edt

# =====================================================
# PAGE
# =====================================================
st.set_page_config(layout="wide", page_title="HELIOS Navigation")
st.title("🛰️ HELIOS Smart Ocean Navigation")

SIMULATION_DT = 1  # hour per step
MPS_TO_DEG_PER_HOUR = 3600 / 111000

# =====================================================
# OFFSHORE WIND
# =====================================================
OFFSHORE_WIND = [
    [[24.18,120.12],[24.22,120.28],[24.05,120.35],[24.00,120.15]],
    [[24.00,120.10],[24.05,120.32],[23.90,120.38],[23.85,120.15]],
]

# =====================================================
# HYCOM LOAD (MULTI TIME)
# =====================================================
@st.cache_data(ttl=3600)
def load_hycom_all():
    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    ds=xr.open_dataset(url,decode_times=False)

    u=ds['ssu'].sel(lat=slice(21,26),lon=slice(118,123))
    v=ds['ssv'].sel(lat=slice(21,26),lon=slice(118,123))

    lons=u.lon.values
    lats=u.lat.values

    return lons,lats,u.values,v.values

lons,lats,U_all,V_all=load_hycom_all()

# =====================================================
# LAND + COAST BUFFER
# =====================================================
land_mask=np.isnan(U_all[0])

dist=distance_transform_edt(~land_mask)
coast_penalty=dist<3

# =====================================================
# SESSION STATE
# =====================================================
if "step" not in st.session_state:
    st.session_state.step=0

if "ship_pos" not in st.session_state:
    st.session_state.ship_pos=np.array([120.3,22.6])

if "track" not in st.session_state:
    st.session_state.track=[st.session_state.ship_pos.copy()]

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("Navigation")

    goal_lon=st.number_input("Goal Lon",118.0,123.0,122.0)
    goal_lat=st.number_input("Goal Lat",21.0,26.0,24.5)

    ship_speed=st.number_input("Ship Speed (km/h)",1.0,60.0,20.0)

    if st.button("Next Step (1 hr)"):
        st.session_state.step+=1

# =====================================================
# CURRENT FIELD
# =====================================================
time_idx=st.session_state.step % U_all.shape[0]
u=U_all[time_idx]
v=V_all[time_idx]

# =====================================================
# HELPERS
# =====================================================
def sample_current(lon,lat):
    i=np.abs(lats-lat).argmin()
    j=np.abs(lons-lon).argmin()
    return u[i,j],v[i,j]

def inside_wind(lon,lat):
    for zone in OFFSHORE_WIND:
        poly=np.array(zone)
        if (poly[:,1].min()<lon<poly[:,1].max() and
            poly[:,0].min()<lat<poly[:,0].max()):
            return True
    return False

# =====================================================
# DYNAMIC NAVIGATION AI
# =====================================================
ship=st.session_state.ship_pos
goal=np.array([goal_lon,goal_lat])

vec_goal=goal-ship
dist_goal=np.linalg.norm(vec_goal)

heading=vec_goal/dist_goal

# sample flow
cu,cv=sample_current(ship[0],ship[1])
current_vec=np.array([cu,cv])*MPS_TO_DEG_PER_HOUR

# avoidance behaviour
if inside_wind(ship[0],ship[1]):
    heading+=np.array([0.5,0.5])

# coastline avoidance
lat_i=np.abs(lats-ship[1]).argmin()
lon_i=np.abs(lons-ship[0]).argmin()

if coast_penalty[lat_i,lon_i]:
    heading+=np.random.randn(2)*0.3

heading=heading/np.linalg.norm(heading)

ship_vec=heading*(ship_speed/111)*SIMULATION_DT

new_pos=ship+ship_vec+current_vec*SIMULATION_DT

st.session_state.ship_pos=new_pos
st.session_state.track.append(new_pos.copy())

# =====================================================
# STATS
# =====================================================
remain_dist=np.linalg.norm(goal-new_pos)*111
remain_time=remain_dist/ship_speed

bearing=np.degrees(np.arctan2(heading[0],heading[1]))%360

satellites_visible=6

# =====================================================
# DASHBOARD
# =====================================================
st.subheader("Navigation Dashboard")

c1,c2,c3,c4=st.columns(4)
c1.metric("Remaining Distance (km)",f"{remain_dist:.1f}")
c2.metric("Remaining Time (hr)",f"{remain_time:.1f}")
c3.metric("Heading (deg)",f"{bearing:.0f}")
c4.metric("Satellite Link","OK" if satellites_visible>3 else "Weak")

# =====================================================
# MAP
# =====================================================
fig=plt.figure(figsize=(10,8))
ax=plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([118,123,21,26])

ax.add_feature(cfeature.LAND,facecolor="#bfbfbf")
ax.add_feature(cfeature.COASTLINE,linewidth=0.6)

speed=np.sqrt(u**2+v**2)

mesh=ax.pcolormesh(lons,lats,speed,
                   cmap="Blues",
                   shading="auto")

fig.colorbar(mesh,ax=ax,label="Current Speed (m/s)")

# offshore wind
for zone in OFFSHORE_WIND:
    poly=np.array(zone)
    ax.fill(poly[:,1],poly[:,0],color="yellow",alpha=0.35)

# full track
track=np.array(st.session_state.track)
ax.plot(track[:,0],track[:,1],
        color="#ff4da6",
        linewidth=2)

# ship icon (ROTATING TRIANGLE)
angle=np.radians(bearing)

ship_shape=np.array([
    [0,0.05],
    [-0.02,-0.02],
    [0.02,-0.02]
])

R=np.array([
    [np.cos(angle),-np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])

rot=(ship_shape@R.T)+new_pos
ax.fill(rot[:,0],rot[:,1],color="gray")

# start
ax.scatter(track[0,0],track[0,1],
           color="#B15BFF",
           s=50,
           edgecolors="black")

# goal
ax.scatter(goal[0],goal[1],
           color="yellow",
           marker="*",
           s=180,
           edgecolors="black")

plt.title(f"HELIOS Dynamic Navigation | Hour {st.session_state.step}")
st.pyplot(fig)
