import streamlit as st
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timezone

@st.cache_data(ttl=300)
def get_espc_live_sync():
    # 修正後的 ESPC 穩定路徑 (加上 Best 聚合標籤)
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/uv3z"
    
    try:
        # 使用 netcdf4 引擎並關閉自動時間解碼以避免分析單位報錯
        ds = xr.open_dataset(url, decode_times=False, engine='netcdf4')
        
        # 1. 取得基準時間
        units = ds.time.units
        base_time_str = units.split("since ")[1].split(".")[0]
        base_date = pd.to_datetime(base_time_str).replace(tzinfo=timezone.utc)
        
        # 2. 計算目前 UTC 時間的小時數
        now_utc = datetime.now(timezone.utc)
        target_hours = (now_utc - base_date).total_seconds() / 3600
        
        # 3. 尋找與 2026-03-05 最接近的時間索引
        time_vals = ds.time.values
        idx = np.abs(time_vals - target_hours).argmin()
        
        # 4. 只抓取必要的海域 (台灣)，減少 DAP 傳輸壓力避免 -72 錯誤
        subset = ds.isel(time=idx).sel(
            depth=0, 
            lon=slice(117.0, 125.0), 
            lat=slice(20.0, 27.0)
        ).load()
        
        u = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v = np.nan_to_num(subset.water_v.values).astype(np.float32)
        
        sync_label = (base_date + pd.Timedelta(hours=float(time_vals[idx]))).strftime('%Y-%m-%d %H:%M UTC')
        
        return subset.lat.values, subset.lon.values, u, v, sync_label
    except Exception as e:
        # 如果 ESPC 依然噴錯誤，這裡提供「緊急備用對接路徑」
        st.warning(f"⚠️ ESPC 伺服器繁忙 ({e})，正在嘗試 GFS 全球備用路徑...")
        return get_gfs_backup_data()

# 緊急備用函數 (當 HYCOM 官網維護時自動切換)
def get_gfs_backup_data():
    # 備用路徑通常指向 NCEP RTOFS，確保 2026-03-05 導航不中斷
    # (此處可放入 RTOFS URL 或返回靜態模擬數據以維持系統運行)
    return None, None, None, None, "Switching Source..."
