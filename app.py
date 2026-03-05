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
# 1. 數據暴力對接 (強制同步 NODASS 時間)
# ===============================
@st.cache_data(ttl=300) # 每 5 分鐘強制清除快取
def get_live_nodass_sync():
    # 這是 HYCOM 目前最穩定的 2026 數據對接點
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    try:
        # 開啟數據集，不解碼時間以避免報錯
        ds = xr.open_dataset(url, decode_times=False)
        
        # 解析基準時間 (hours since ...)
        units = ds.time.units
        base_time_str = units.split("since ")[1].split(".")[0]
        base_date = pd.to_datetime(base_time_str).replace(tzinfo=timezone.utc)
        
        # 獲取「現在 UTC 時間」
        now_utc = datetime.now(timezone.utc)
        # 計算從基準點到現在的小時數
        target_hours = (now_utc - base_date).total_seconds() / 3600
        
        # 【關鍵補丁】從時間軸中找出最接近「現在」的索引
        time_vals = ds.time.values
        # 找出距離現在最近的格點索引
        idx = np.abs(time_vals - target_hours).argmin()
        
        # 讀取該時間點資料
        subset = ds.isel(time=idx).sel(
            depth=0, 
            lon=slice(117.0, 125.0), 
            lat=slice(20.0, 27.0)
        ).load()
        
        u_val = np.nan_to_num(subset.water_u.values).astype(np.float32)
        v_val = np.nan_to_num(subset.water_v.values).astype(np.float32)
        
        # 產出與 NODASS 同步的時間標籤
        actual_time = (base_date + pd.Timedelta(hours=float(time_vals[idx])))
        sync_label = actual_time.strftime('%Y-%m-%d %H:%M UTC')
        
        return subset.lat.values.astype(np.float32), subset.lon.values.astype(np.float32), u_val, v_val, sync_label
    except Exception as e:
        st.error(f"數據對接失敗: {e}")
        return None, None, None, None, "連線中斷"

# ... (haversine 與 A* 演算法保持不變) ...

# ===============================
# 2. 介面與顯示 (與截圖同步)
# ===============================
st.set_page_config(layout="wide", page_title="HELIOS V13 NODASS Sync")
st.title("🛰️ HELIOS 智慧導航 (NODASS 實時同步版)")

lat, lon, u_raw, v_raw, live_time = get_live_nodass_sync()

if lat is not None:
    # 顯示現在的時間點
    st.success(f"✅ 已成功對接實時流場：{live_time}")
    
    with st.sidebar:
        st.header("📍 任務參數")
        # 您原本的輸入介面...
        base_speed = st.slider("巡航航速 (kn)", 10.0, 25.0, 15.0)
        run_btn = st.button("🚀 執行即時優化導航", use_container_width=True)

    # ... (中間 A* 邏輯保持不變) ...

    # ===============================
    # 3. 地圖視覺化 (仿 NODASS 色彩)
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([117.5, 124.5, 20.5, 26.5]) 
    
    # 計算流速
    speed = np.sqrt(u_raw**2 + v_raw**2)
    # 使用與 NODASS 相似的 'jet' 或 'turbo' 色彩，並調整分佈
    mesh = ax.pcolormesh(lon, lat, speed, cmap='turbo', shading='auto', alpha=0.9, zorder=0)
    plt.colorbar(mesh, ax=ax, label='流速 (m/s)', fraction=0.03, pad=0.04)

    ax.add_feature(cfeature.LAND, facecolor='#1a1a1a', zorder=5) # 深色陸地
    ax.add_feature(cfeature.COASTLINE, edgecolor='cyan', linewidth=0.8, zorder=6) # 亮色海岸線

    # 如果有路徑就畫出來
    # ... 繪圖邏輯 ...
    st.pyplot(fig)
