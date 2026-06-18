import time  # 確保在檔案頂部引入

# ===============================
# Load HYCOM (已加入防崩潰自動重試機制)
# ===============================
@st.cache_data(ttl=3600)
def load_hycom():
    url = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/ice/2026"
    
    max_retries = 4       # 最大重試次數
    retry_delay = 15      # 每次失敗後等待 15 秒再試
    ds = None

    for attempt in range(max_retries):
        try:
            # 使用 xarray 開啟遠端數據集，並加上基本的範疇提示
            ds = xr.open_dataset(url, decode_times=False)
            break  # 成功連線則跳出重試迴圈
        except Exception as e:
            if attempt < max_retries - 1:
                # 在 Streamlit 畫面上提示使用者正在重新嘗試
                st.warning(f"⚠️ HYCOM 伺服器連線失敗 (嘗試 {attempt + 1}/{max_retries}): {e}。將於 {retry_delay} 秒後自動重試...")
                time.sleep(retry_delay)
            else:
                st.error(f"❌ 嚴重錯誤：已達到最大重試次數，無法連接到 HYCOM 數據庫。\n錯誤細節: {e}")
                st.info("💡 提示：這通常是因為 HYCOM 伺服器正在進行每日例行維護（美東時間 02:00-03:00 / 12:00-13:00），請稍後再試。")
                st.stop()

    # 以下維持原本的資料處理解析邏輯
    if 'time_origin' in ds['time'].attrs:
        origin = pd.to_datetime(ds['time'].attrs['time_origin'])
        obs_time = origin + pd.to_timedelta(ds['time'].values[-1], unit='h')
    else:
        obs_time = pd.Timestamp.now()

    # 進行區域切片
    sub = ds.sel(lat=slice(21, 26), lon=slice(118, 124))
    lons = sub.lon.values
    lats = sub.lat.values
    
    # 預先加載最後一個時間步的海流數據
    u_data = sub['ssu'].isel(time=-1).values
    v_data = sub['ssv'].isel(time=-1).values
    land_mask = np.isnan(u_data)

    # 關閉 dataset 釋放資源
    ds.close()

    return lons, lats, land_mask, obs_time, u_data, v_data
