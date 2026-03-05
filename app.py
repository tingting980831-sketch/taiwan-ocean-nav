import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import heapq
import xarray as xr
import pandas as pd
from datetime import datetime, timezone

# ===============================
# 1. 數據源嫁接：ESPC-D-V02 穩定版
# ===============================
@st.cache_data(ttl=600)
def get_espc_safe_data():
    # 改用 FMRC 'Best' 路徑，這對 OPeNDAP 連線最穩定
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/FMRC/runs/ESPC-D-V02_RUN_best.ncd"
    
    try:
        # 加上 timeout 參數與強制使用 netcdf4 引擎
        ds = xr.open_dataset(url, decode_times=False, engine='netcdf4')
        
        # 取得基準時間與現在時間點
        units = ds.time.units
        base_date = pd.to_datetime(units.split("since ")[1].split(".")[0]).replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        
        # 計算偏移並對齊索引
        target_h = (now_utc - base_date).total_seconds() / 3600
        time_vals = ds.time.values
        idx = np.abs(time_vals - target_h).argmin()
        
        # 關鍵：先 sel 再 load，減少網路傳輸量以避免連線中斷
        subset = ds.isel(time=idx).sel(
            depth=0, 
            lon=slice(117.2, 124.8), 
            lat=slice(20.2, 26.8)
        ).load()
        
        u = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values).astype(np.float32)
        
        sync_label = (base_date + pd.Timedelta(hours=float(time_vals[idx]))).strftime('%Y-%m-%d %H:%M UTC')
        return subset.lat.values, subset.lon.values, u, v, sync_label
    except Exception as e:
        # 如果還是失敗，顯示更詳細的引導
        st.error(f"⚠️ 同步失敗：{e}")
        st.info("提示：這通常是 HYCOM 伺服器端連線過多。請嘗試重新整理網頁，或點擊右上角選單中的 'Clear Cache'。")
        return None, None, None, None, "Retry Required"

# (後續的 A* 演算法與地圖顯示部分保持不變)
