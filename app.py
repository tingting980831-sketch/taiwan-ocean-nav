import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
from scipy.ndimage import distance_transform_edt

############################################
# 1 讀取 HYCOM
############################################

def load_hycom():

    url="https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"

    ds=xr.open_dataset(url,decode_times=False)

    origin=pd.to_datetime(ds["time"].attrs["time_origin"])
    times=origin+pd.to_timedelta(ds["time"].values,unit="h")

    lat_slice=slice(20,26)
    lon_slice=slice(119,123)

    ssu=ds["ssu"].sel(lat=lat_slice,lon=lon_slice)
    ssv=ds["ssv"].sel(lat=lat_slice,lon=lon_slice)

    ssu=ssu.isel(time=-1)
    ssv=ssv.isel(time=-1)

    u=ssu.values
    v=ssv.values

    lats=ssu["lat"].values
    lons=ssu["lon"].values

    land_mask=np.isnan(u)

    return lons,lats,u,v,land_mask


############################################
# 2 找最近海格點
############################################

def nearest_ocean_cell(lon,lat,lons,lats,land_mask):

    lon_idx=np.abs(lons-lon).argmin()
    lat_idx=np.abs(lats-lat).argmin()

    if not land_mask[lat_idx,lon_idx]:
        return (lat_idx,lon_idx)

    ocean=np.where(~land_mask)

    dist=np.sqrt(
        (lats[ocean[0]]-lat)**2+
        (lons[ocean[1]]-lon)**2
    )

    i=dist.argmin()

    return (ocean[0][i],ocean[1][i])


############################################
# 3 安全距離
############################################

def compute_safety(land_mask):

    ocean=~land_mask
    dist=distance_transform_edt(ocean)

    return dist


############################################
# 4 A*
############################################

def astar(start,goal,u,v,land_mask,safety):

    rows,cols=land_mask.shape

    dirs=[
        (1,0),(-1,0),(0,1),(0,-1),
        (1,1),(1,-1),(-1,1),(-1,-1)
    ]

    pq=[]
    heapq.heappush(pq,(0,start))

    came_from={}
    cost={start:0}

    while pq:

        _,cur=heapq.heappop(pq)

        if cur==goal:
            break

        for d in dirs:

            ni=cur[0]+d[0]
            nj=cur[1]+d[1]

            if ni<0 or nj<0 or ni>=rows or nj>=cols:
                continue

            if land_mask[ni,nj]:
                continue

            dist=np.sqrt(d[0]**2+d[1]**2)

            flow=u[cur]*d[1]+v[cur]*d[0]

            land_penalty=1/(safety[ni,nj]+1)

            new_cost=cost[cur]+dist+land_penalty-0.5*flow

            if (ni,nj) not in cost or new_cost<cost[(ni,nj)]:

                cost[(ni,nj)]=new_cost

                heapq.heappush(pq,(new_cost,(ni,nj)))

                came_from[(ni,nj)]=cur

    if goal not in came_from:
        print("找不到路徑")
        return []

    path=[]
    node=goal

    while node!=start:
        path.append(node)
        node=came_from[node]

    path.append(start)
    path.reverse()

    return path


############################################
# 5 繪圖
############################################

def plot_map(lons,lats,u,v,path,start_lon,start_lat,goal_lon,goal_lat):

    fig=plt.figure(figsize=(8,8))
    ax=plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([119,123,20,26])

    ax.add_feature(cfeature.LAND,facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE)

    ax.quiver(lons,lats,u,v,scale=3)

    if len(path)>0:

        path_lats=[lats[p[0]] for p in path]
        path_lons=[lons[p[1]] for p in path]

        ax.plot(path_lons,path_lats,color="red",linewidth=2,label="AI Route")

    ax.scatter(start_lon,start_lat,color="green",s=80,label="Start")
    ax.scatter(goal_lon,goal_lat,color="blue",s=80,label="Goal")

    plt.legend()
    plt.title("Ocean Current Navigation")

    plt.show()


############################################
# 主程式
############################################

print("下載 HYCOM 海流資料...")

lons,lats,u,v,land_mask=load_hycom()

print("計算安全距離...")

safety=compute_safety(land_mask)

# 起點 高雄
start_lon=120.3
start_lat=22.6

# 終點 外海測試
goal_lon=122.5
goal_lat=24

start=nearest_ocean_cell(start_lon,start_lat,lons,lats,land_mask)
goal=nearest_ocean_cell(goal_lon,goal_lat,lons,lats,land_mask)

print("A* 搜尋路徑...")

path=astar(start,goal,u,v,land_mask,safety)

print("完成")

plot_map(lons,lats,u,v,path,start_lon,start_lat,goal_lon,goal_lat)
