import streamlit as st
import numpy as np
import requests
import pandas as pd
import heapq
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.title("Ocean Route Planner")

############################
# 海象 API
############################

@st.cache_data
def get_realtime_marine_data(lat, lon):

    url = "https://marine-api.open-meteo.com/v1/marine"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height,wind_speed_10m,wind_direction_10m",
        "forecast_days": 1,
        "timezone": "Asia/Taipei"
    }

    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    wave = float(df["wave_height"].iloc[0])
    wind_speed = float(df["wind_speed_10m"].iloc[0])
    wind_dir = float(df["wind_direction_10m"].iloc[0])

    return wave, wind_speed, wind_dir


############################
# 風向轉換
############################

def wind_to_uv(speed, direction):

    rad = np.deg2rad(direction)

    u = speed * np.sin(rad)
    v = speed * np.cos(rad)

    return u, v


############################
# A* 演算法
############################

def heuristic(a, b):

    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def astar(start, goal, wave_grid, wind_grid):

    rows, cols = wave_grid.shape

    pq = []
    heapq.heappush(pq,(0,start))

    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    moves = [(-1,0),(1,0),(0,-1),(0,1)]

    while pq:

        _,current = heapq.heappop(pq)

        if current == goal:
            break

        for dx,dy in moves:

            nx = current[0]+dx
            ny = current[1]+dy

            if 0<=nx<rows and 0<=ny<cols:

                wave = wave_grid[nx,ny]
                wind = wind_grid[nx,ny]

                new_cost = cost_so_far[current] + 1 + wave*2 + wind*0.5

                if (nx,ny) not in cost_so_far or new_cost < cost_so_far[(nx,ny)]:

                    cost_so_far[(nx,ny)] = new_cost

                    priority = new_cost + heuristic(goal,(nx,ny))

                    heapq.heappush(pq,(priority,(nx,ny)))

                    came_from[(nx,ny)] = current

    path = []
    node = goal

    while node != start:

        path.append(node)
        node = came_from[node]

    path.append(start)
    path.reverse()

    return path


############################
# 建立海域 grid
############################

lat_range = np.linspace(22, 26, 20)
lon_range = np.linspace(120, 124, 20)

lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

wave_grid = np.zeros_like(lat_grid)
wind_grid = np.zeros_like(lat_grid)
wind_u_grid = np.zeros_like(lat_grid)
wind_v_grid = np.zeros_like(lat_grid)

############################
# 抓海象
############################

with st.spinner("Loading marine data..."):

    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):

            lat = lat_grid[i,j]
            lon = lon_grid[i,j]

            wave, wind_speed, wind_dir = get_realtime_marine_data(lat,lon)

            u,v = wind_to_uv(wind_speed,wind_dir)

            wave_grid[i,j] = wave
            wind_grid[i,j] = wind_speed
            wind_u_grid[i,j] = u
            wind_v_grid[i,j] = v


############################
# 航線
############################

start = (2,2)
goal = (17,17)

path = astar(start,goal,wave_grid,wind_grid)

path_lats = [lat_grid[p] for p in path]
path_lons = [lon_grid[p] for p in path]


############################
# 2D 地圖
############################

st.subheader("2D Wave Map")

fig,ax = plt.subplots()

c = ax.contourf(lon_grid,lat_grid,wave_grid,cmap="Blues")

ax.plot(path_lons,path_lats,"r",linewidth=2)

plt.colorbar(c,label="Wave Height (m)")

st.pyplot(fig)


############################
# 3D 海象
############################

st.subheader("3D Ocean Model")

fig3d = go.Figure()

fig3d.add_trace(go.Surface(
    x=lon_grid,
    y=lat_grid,
    z=wave_grid,
    colorscale="Viridis",
    opacity=0.8,
    name="Wave Height"
))

skip=3

fig3d.add_trace(go.Cone(

    x=lon_grid[::skip,::skip].flatten(),
    y=lat_grid[::skip,::skip].flatten(),
    z=wave_grid[::skip,::skip].flatten()+0.5,

    u=wind_u_grid[::skip,::skip].flatten(),
    v=wind_v_grid[::skip,::skip].flatten(),
    w=np.zeros_like(wind_u_grid[::skip,::skip].flatten()),

    sizemode="scaled",
    sizeref=2,
    showscale=False,
    name="Wind"
))

fig3d.add_trace(go.Scatter3d(

    x=path_lons,
    y=path_lats,
    z=np.ones(len(path_lats))*np.max(wave_grid)+1,

    mode="lines",
    line=dict(color="red",width=6),
    name="Route"
))

fig3d.update_layout(

    scene=dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis_title="Wave Height"
    ),

    height=700
)

st.plotly_chart(fig3d,use_container_width=True)
