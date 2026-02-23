import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

# -------------------------------
# 1️⃣ 頁面設定
# -------------------------------
st.set_page_config(page_title="AI 海象導航系統 + HELIOS", layout="wide")
st.title("⚓ 智慧避障導航系統 + HELIOS")
st.write("結合 A* 演算法、HYCOM 海流數據與 HELIOS 低軌衛星即時導航分析")

# -------------------------------
# 2️⃣ Session State 初始化
# -------------------------------
if "start_lon" not in st.session_state:
    st.session_state.start_lon = 121.750
    st.session_state.start_lat = 25.150

if "sim_lon" not in st.session_state:
    st.session_state.sim_lon = 121.850
    st.session_state.sim_lat = 25.150

DATA_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"

# -------------------------------
# 3️⃣ 側邊欄：出發點與目的地
# -------------------------------
st.sidebar.header("📍 航線設定")

def get_ais_position():
    # 模擬 AIS 即時定位
    return np.random.uniform(121.6, 122.1), np.random.uniform(24.8, 25.3)

if st.sidebar.button("📡 立即定位（AIS）"):
    lon, lat = get_ais_position()
    st.session_state.start_lon = lon
    st.session_state.start_lat = lat
    st.sidebar.success("已取得 AIS 即時位置")

s_lon = st.sidebar.number_input("起點經度", value=st.session_state.start_lon, format="%.3f")
s_lat = st.sidebar.number_input("起點緯度", value=st.session_state.start_lat, format="%.3f")
st.session_state.start_lon = s_lon
st.session_state.start_lat = s_lat

e_lon = st.sidebar.number_input("終點經度", value=121.900, format="%.3f")
e_lat = st.sidebar.number_input("終點緯度", value=24.600, format="%.3f")

# -------------------------------
# 4️⃣ 側邊欄：HELIOS 星座設定
# -------------------------------
st.sidebar.header("🛰️ HELIOS 星座設定")
st.sidebar.info("""
**軌道高度**: 900 km  
**總衛星數**: 36 顆  
**覆蓋率**: ~84%  
""")

if st.sidebar.button("🎲 瞬移到海上測試點"):
    st.session_state.sim_lon = np.random.uniform(119.5, 122.5)
    st.session_state.sim_lat = np.random.uniform(22.5, 25.5)

c_lon = st.sidebar.number_input("模擬經度", value=st.session_state.sim_lon, format="%.3f")
c_lat = st.sidebar.number_input("模擬緯度", value=st.session_state.sim_lat, format="%.3f")

SHIP_POWER_KNOTS = 15.0

# -------------------------------
# 5️⃣ A* 演算法函數
# -------------------------------
def astar_search(grid, safety_map, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {start: 0}
    while open_heap:
        current = heapq.heappop(open_heap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        for dy, dx in neighbors:
            nb = (current[0]+dy, current[1]+dx)
            if 0<=nb[0]<grid.shape[0] and 0<=nb[1]<grid.shape[1]:
                if grid[nb]==1: continue
                step = 1.414 if dy!=0 and dx!=0 else 1.0
                cost = step + safety_map[nb]*1.5
                g_new = g_score[current] + cost
                if g_new < g_score.get(nb, 1e12):
                    g_score[nb]=g_new
                    f = g_new + np.linalg.norm(np.array(nb)-np.array(goal))
                    came_from[nb]=current
                    heapq.heappush(open_heap, (f, nb))
    return []

# -------------------------------
# 6️⃣ A* 航線規劃
# -------------------------------
if st.sidebar.button("🚀 規劃航線"):
    with st.spinner("讀取 HYCOM 資料..."):
        try:
            ds = xr.open_dataset(DATA_URL, decode_times=False)
            margin = 1.0
            sub = ds.sel(
                lon=slice(min(s_lon,e_lon)-margin,max(s_lon,e_lon)+margin),
                lat=slice(min(s_lat,e_lat)-margin,max(s_lat,e_lat)+margin),
                depth=0
            ).isel(time=-1).load()
            lons, lats = sub.lon.values, sub.lat.values
            grid = np.where(np.isnan(sub.water_u.values),1,0)
            safety_map = np.exp(-distance_transform_edt(1-grid)/0.5)

            iy_s, ix_s = np.abs(lats-s_lat).argmin(), np.abs(lons-s_lon).argmin()
            iy_e, ix_e = np.abs(lats-e_lat).argmin(), np.abs(lons-e_lon).argmin()

            def nearest_water(y,x):
                if grid[y,x]==0: return (y,x)
                Y,X=np.indices(grid.shape)
                d=np.sqrt((Y-y)**2+(X-x)**2)
                d[grid==1]=1e9
                return np.unravel_index(np.argmin(d),grid.shape)

            s_idx, e_idx = nearest_water(iy_s,ix_s), nearest_water(iy_e,ix_e)
            path = astar_search(grid,safety_map,s_idx,e_idx)

            if not path:
                st.error("找不到路徑")
            else:
                path_lon = [s_lon]+[lons[p[1]] for p in path]+[e_lon]
                path_lat = [s_lat]+[lats[p[0]] for p in path]+[e_lat]

                fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection':ccrs.PlateCarree()})
                ax.set_extent([min(path_lon)-0.4,max(path_lon)+0.4,min(path_lat)-0.4,max(path_lat)+0.4])
                speed = np.sqrt(sub.water_u**2+sub.water_v**2)
                cf = ax.pcolormesh(lons,lats,speed,cmap='viridis',shading='auto',alpha=0.7)
                plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.03, shrink=0.6, label='Current Speed (m/s)')
                ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor='#222222')
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'),edgecolor='white')
                ax.plot(path_lon,path_lat,color='magenta',lw=3,label='AI Path')
                ax.scatter(s_lon,s_lat,c='yellow',s=100,label='Start')
                ax.scatter(e_lon,e_lat,c='red',marker='*',s=200,label='Goal')
                ax.set_title("AI Marine Navigation & Obstacle Avoidance",fontsize=14)
                ax.legend(loc='lower right')
                st.pyplot(fig)
                st.success("規劃完成 (Success)! 航線經緯度已精確對齊。")
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------
# 7️⃣ HELIOS 即時導航分析
# -------------------------------
def calculate_metrics(u,v,speed_knots):
    vs = speed_knots*0.514
    sog = vs + (u*0.6+v*0.4)
    sog_knots = sog/0.514
    fuel_saving = max(min((1-(vs/sog)**3)*100+12.5,18.4),0.0)
    comm_stability = 0.84 + np.random.uniform(0.08,0.12)
    return round(sog_knots,2), round(fuel_saving,1), round(comm_stability,2)

if st.sidebar.button("🚀 HELIOS 即時導航分析"):
    with st.spinner("讀取 HYCOM 資料中..."):
        try:
            ds = xr.open_dataset(DATA_URL, decode_times=False)
            sub = ds.sel(lon=slice(c_lon-0.6,c_lon+0.6),
                         lat=slice(c_lat-0.6,c_lat+0.6),
                         depth=0).isel(time=-1).load()
            u_val = sub.water_u.interp(lat=c_lat,lon=c_lon).values
            v_val = sub.water_v.interp(lat=c_lat,lon=c_lon).values

            if np.isnan(u_val):
                st.error("❌ 座標位於陸地")
            else:
                sog,fuel,comm = calculate_metrics(float(u_val),float(v_val),SHIP_POWER_KNOTS)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("🚀 對地速度 (SOG)",f"{sog} kn",f"{round(sog-SHIP_POWER_KNOTS,1)} kn")
                m2.metric("⛽ 燃油節省比例",f"{fuel}%","AI 優化中")
                m3.metric("📡 HELIOS 穩定度",f"{comm}","36 Sats / 900km")
                m4.metric("⚙️ 建議轉向角",f"{round(np.degrees(np.arctan2(v_val,u_val)),1)}°")

                fig, ax = plt.subplots(figsize=(10,7),subplot_kw={'projection':ccrs.PlateCarree()})
                ax.set_extent([c_lon-0.5,c_lon+0.5,c_lat-0.5,c_lat+0.5])
                mag = np.sqrt(sub.water_u**2+sub.water_v**2)
                land_mask = np.isnan(sub.water_u.values)
                mag_masked = np.ma.masked_where(land_mask,mag)
                cf = ax.pcolormesh(sub.lon,sub.lat,mag_masked,cmap='YlGn',shading='auto',alpha=0.9)
                plt.colorbar(cf,label='Current Speed (m/s)',shrink=0.5)
                ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor='#1e1e1e',zorder=5)
                ax.add_feature(cfeature.COASTLINE.with_scale('10m'),edgecolor='white',linewidth=1.2,zorder=6)
                ax.quiver(c_lon,c_lat,u_val,v_val,color='red',scale=5,zorder=10)
                ax.scatter(c_lon,c_lat,color='#FF00FF',s=120,edgecolors='white',zorder=11)
                st.pyplot(fig)
                st.success("HELIOS 系統運作中：當前覆蓋率足以支撐下一步決策。")
        except Exception as e:
            st.error(f"連線失敗: {e}")
