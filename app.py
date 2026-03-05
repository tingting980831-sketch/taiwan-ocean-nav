import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr

# ===============================
# 1. 讀取 HYCOM 即時流場
# ===============================

@st.cache_data(ttl=600)
def load_hycom():
    url = "https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"
    ds = xr.open_dataset(url, decode_times=False, engine="netcdf4")
    latest = -1

    subset = ds.isel(time=latest).sel(
        depth=0,
        lon=slice(117.2, 124.8),
        lat=slice(20.2, 26.8)
    ).load()

    lat = subset.lat.values
    lon = subset.lon.values
    u = np.nan_to_num(subset.water_u.values)
    v = np.nan_to_num(subset.water_v.values)
    time_label = str(ds.time.values[latest])

    return lat, lon, u, v, time_label

# ===============================
# 2. 距離計算函數
# ===============================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# ===============================
# 3. A* 演算法
# ===============================
def astar(start, goal, lat, lon, u, v, forbidden):
    open_set=[]
    heapq.heappush(open_set,(0,start))
    came = {}
    g = {start:0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path=[]
            while current in came:
                path.append(current)
                current = came[current]
            path.append(start)
            path.reverse()
            return path

        for d in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ni, nj = current[0]+d[0], current[1]+d[1]
            if ni<0 or nj<0 or ni>=len(lat) or nj>=len(lon):
                continue
            if forbidden[ni, nj]:
                continue

            dist = haversine(lat[current[0]], lon[current[1]], lat[ni], lon[nj])
            # 目前只用距離作 cost，不計算順流
            tentative = g[current]+dist

            if (ni,nj) not in g or tentative<g[(ni,nj)]:
                g[(ni,nj)] = tentative
                priority = tentative + np.hypot(ni-goal[0], nj-goal[1])
                heapq.heappush(open_set,(priority,(ni,nj)))
                came[(ni,nj)] = current
    return None

# ===============================
# 4. Streamlit UI
# ===============================
st.set_page_config(layout="wide")
st.title("HELIOS 即時海象導航系統 (流線 + 航線)")

lat, lon, u, v, time_label = load_hycom()
st.success(f"HYCOM 最新資料時間: {time_label}")

LON, LAT = np.meshgrid(lon, lat)

# 建立台灣與澎湖避開區
mask_tw = (((LAT-23.75)/1.85)**2 + ((LON-121.0)/0.75)**2) < 1
mask_is = (((LAT-23.5)/0.3)**2 + ((LON-119.6)/0.3)**2) < 1
forbidden = mask_tw | mask_is

with st.sidebar:
    st.header("導航設定")
    s_lat = st.number_input("起點緯度", value=22.3)
    s_lon = st.number_input("起點經度", value=120.1)
    e_lat = st.number_input("終點緯度", value=25.2)
    e_lon = st.number_input("終點經度", value=122.0)
    run = st.button("計算最佳路徑")

# 找最近格點
def get_idx(lat0, lon0):
    i = np.argmin(np.abs(lat-lat0))
    j = np.argmin(np.abs(lon-lon0))
    return i,j

path = None
if run:
    s = get_idx(s_lat, s_lon)
    g = get_idx(e_lat, e_lon)
    if forbidden[s] or forbidden[g]:
        st.error("起點或終點在陸地")
    else:
        path = astar(s,g,lat,lon,u,v,forbidden)

# ===============================
# 5. 地圖繪製
# ===============================
fig, ax = plt.subplots(figsize=(11,8), subplot_kw={'projection':ccrs.PlateCarree()})
ax.set_extent([117.2,124.8,20.2,26.8])

# 流速熱圖
speed = np.sqrt(u**2 + v**2)
mesh = ax.pcolormesh(lon, lat, speed, cmap="turbo", shading="auto", alpha=0.8)

# 流線
ax.streamplot(lon, lat, u, v, density=2, color=speed, linewidth=1, cmap="viridis", transform=ccrs.PlateCarree(), zorder=3)

# 地形
ax.add_feature(cfeature.LAND, facecolor="black")
ax.add_feature(cfeature.COASTLINE, color="white", linewidth=0.8)

# 畫航線
if path:
    path_lat = [lat[p[0]] for p in path]
    path_lon = [lon[p[1]] for p in path]
    ax.plot(path_lon, path_lat, color="red", linewidth=3, transform=ccrs.PlateCarree(), label="A* Route")
    ax.scatter(s_lon, s_lat, color="lime", s=120, transform=ccrs.PlateCarree(), label="Start")
    ax.scatter(e_lon, e_lat, color="yellow", s=120, transform=ccrs.PlateCarree(), label="End")

ax.legend(loc="upper right")
plt.colorbar(mesh, ax=ax, label="流速 m/s", fraction=0.03, pad=0.04)

st.pyplot(fig)
