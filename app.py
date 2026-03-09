import streamlit as st
import numpy as np
import requests
import pandas as pd
import heapq
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.title("Ocean Route Planner")

####################################
# 取得海象（只抓一次）
####################################

@st.cache_data
def get_marine():

    url = "https://marine-api.open-meteo.com/v1/marine"

    params = {
        "latitude": 23.5,
        "longitude": 121,
        "hourly": "wave_height,wind_speed_10m,wind_direction_10m",
        "forecast_days": 1,
        "timezone": "Asia/Taipei"
    }

    try:

        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        df = pd.DataFrame(data["hourly"])

        wave = float(df["wave_height"].iloc[0])
        wind_speed = float(df["wind_speed_10m"].iloc[0])
        wind_dir = float(df["wind_direction_10m"].iloc[0])

        return wave, wind_speed, wind_dir

    except:

        return 1.0, 5.0, 90.0


####################################
# 風向轉向量
####################################

def wind_to_uv(speed, direction):

    rad = np.deg2rad(direction)

    u = speed * np.sin(rad)
    v = speed * np.cos(rad)

    return u, v


####################################
# A* 演算法
####################################

def heuristic(a,b):

    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def astar(start,goal,wave_grid,wind_grid):

    rows,cols = wave_grid.shape

    pq=[]
    heapq.heappush(pq,(0,start))

    came={}
    cost={}

    came[start]=None
    cost[start]=0

    moves=[(-1,0),(1,0),(0,-1),(0,1)]

    while pq:

        _,cur=heapq.heappop(pq)

        if cur==goal:
            break

        for dx,dy in moves:

            nx=cur[0]+dx
            ny=cur[1]+dy

            if 0<=nx<rows and 0<=ny<cols:

                wave=wave_grid[nx,ny]
                wind=wind_grid[nx,ny]

                new_cost=cost[cur]+1+wave*2+wind*0.3

                if (nx,ny) not in cost or new_cost<cost[(nx,ny)]:

                    cost[(nx,ny)]=new_cost

                    priority=new_cost+heuristic(goal,(nx,ny))

                    heapq.heappush(pq,(priority,(nx,ny)))

                    came[(nx,ny)]=cur

    path=[]
    node=goal

    while node!=start:

        path.append(node)
        node=came[node]

    path.append(start)
    path.reverse()

    return path


####################################
# 生成海域 grid
####################################

lat_range=np.linspace(22,26,30)
lon_range=np.linspace(120,124,30)

lon_grid,lat_grid=np.meshgrid(lon_range,lat_range)


####################################
# 抓海象
####################################

wave,wind_speed,wind_dir=get_marine()

u,v=wind_to_uv(wind_speed,wind_dir)


####################################
# 建立海象場（隨機微變化）
####################################

wave_grid=wave+np.random.normal(0,0.2,lat_grid.shape)

wind_grid=wind_speed+np.random.normal(0,1,lat_grid.shape)

wind_u_grid=np.full(lat_grid.shape,u)
wind_v_grid=np.full(lat_grid.shape,v)


####################################
# 航線
####################################

start=(3,3)
goal=(25,25)

path=astar(start,goal,wave_grid,wind_grid)

path_lats=[lat_grid[p] for p in path]
path_lons=[lon_grid[p] for p in path]


####################################
# 2D 波高圖
####################################

st.subheader("2D Wave Map")

fig,ax=plt.subplots()

c=ax.contourf(lon_grid,lat_grid,wave_grid,cmap="Blues")

ax.plot(path_lons,path_lats,"r",linewidth=2)

plt.colorbar(c,label="Wave Height (m)")

st.pyplot(fig)


####################################
# 3D 海象
####################################

st.subheader("3D Ocean Model")

fig3d=go.Figure()

fig3d.add_trace(go.Surface(

    x=lon_grid,
    y=lat_grid,
    z=wave_grid,

    colorscale="Viridis",
    opacity=0.8,
    name="Wave Height"
))


skip=4

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
