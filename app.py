import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

CACHE_FILE = "hycom_cache.nc"


# ===============================
# 建立備援流場（永不空白）
# ===============================
def create_backup_ocean():

    lats = np.linspace(20, 27, 80)
    lons = np.linspace(118, 126, 80)

    lon2d, lat2d = np.meshgrid(lons, lats)

    # 模擬黑潮流場
    u = 0.6 * np.sin((lat2d - 22) / 3)
    v = 0.4 * np.cos((lon2d - 121) / 3)

    ds = xr.Dataset(
        {
            "water_u": (("lat", "lon"), u),
            "water_v": (("lat", "lon"), v),
        },
        coords={"lat": lats, "lon": lons},
    )

    return ds, "BACKUP", datetime.now()


# ===============================
# 嘗試下載 HYCOM
# ===============================
def try_fetch_hycom():

    try:
        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"

        ds = xr.open_dataset(
            url,
            decode_times=False,
        )

        ds = ds.sel(
            lat=slice(20, 27),
            lon=slice(118, 126),
            depth=0
        ).isel(time=-1)

        # 儲存本地快取（成功才存）
        ds.to_netcdf(CACHE_FILE)

        time_value = ds["time"].values.item()
        base_time = datetime(2000, 1, 1)
        flow_time = base_time + timedelta(hours=int(time_value))

        return ds, "ONLINE", flow_time

    except Exception as e:
        return None, "FAILED", None


# ===============================
# 主載入（永不當機）
# ===============================
@st.cache_data(ttl=1800)
def load_ocean_data():

    ds, status, flow_time = try_fetch_hycom()

    # ✅ 成功
    if ds is not None:
        return ds, status, flow_time

    # ✅ 使用本地 cache
    if os.path.exists(CACHE_FILE):
        try:
            ds = xr.open_dataset(CACHE_FILE)
            return ds, "CACHE", datetime.now()
        except:
            pass

    # ✅ 最後備援
    return create_backup_ocean()
