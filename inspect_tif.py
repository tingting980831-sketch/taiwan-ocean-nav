"""
inspect_nc.py -- 快速檢查 NetCDF (ERA5 風/浪、HYCOM 海流等) 的內容

用法:
    python inspect_nc.py C:\\Users\\tingt\\ERA5\\era5_2023_03.nc
    python inspect_nc.py C:\\Users\\tingt\\ERA5_WAVE_2023\\wave_2023_03.nc

會印出: 維度、座標範圍與間距、時間範圍與間隔、每個變數的單位與名稱、
        第一個時間切片的缺值比例 (判斷陸地遮罩/資料完整度)。
"""
import os
import sys

if len(sys.argv) < 2:
    sys.exit("用法: python inspect_nc.py <檔案.nc>")

path = sys.argv[1]
print(f"檔案: {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")

# --- 先看檔頭, 判斷這到底是什麼格式 (只用標準函式庫) ---
with open(path, "rb") as f:
    head = f.read(16)
if head[:4] == b"\x89HDF":
    kind = "NetCDF4 / HDF5 (正常, 需要 netCDF4 或 h5netcdf 套件)"
elif head[:3] == b"CDF":
    kind = "NetCDF3 classic (正常)"
elif head[:5].lower() == b"<html" or head[:5].lower() == b"<?xml" or head[:9].lower() == b"<!doctype":
    kind = "網頁/XML 文字, 不是資料檔 (可能是下載失敗存成的錯誤頁)"
elif head[:4] in (b"GRIB", b"PK\x03\x04"):
    kind = "GRIB 或 ZIP 壓縮檔, 不是 NetCDF (需先解壓/轉檔)"
else:
    kind = "無法辨識"
print(f"檔頭位元組: {head!r}\n判斷: {kind}")

import numpy as np
import xarray as xr

ds = None
errors = {}
for eng in (None, "netcdf4", "h5netcdf", "scipy"):
    try:
        ds = xr.open_dataset(path, engine=eng) if eng else xr.open_dataset(path)
        print(f"讀取成功 (engine={eng or '自動'})")
        break
    except Exception as e:                                   # noqa: BLE001
        errors[eng or "自動"] = f"{type(e).__name__}: {str(e).strip()[:300]}"
if ds is None and any("decode time" in v or "decode_times" in v for v in errors.values()):
    for eng in ("netcdf4", None):
        try:
            ds = (xr.open_dataset(path, engine=eng, decode_times=False) if eng
                  else xr.open_dataset(path, decode_times=False))
            print("\n[注意] 檔內時間單位無法標準解碼, 已改用 decode_times=False 讀取 "
                  "(時間座標會顯示為原始數字)")
            break
        except Exception as e:                               # noqa: BLE001
            errors[f"decode_times=False/{eng or '自動'}"] = f"{type(e).__name__}: {str(e).strip()[:300]}"
if ds is None:
    print("\n每個讀取引擎的錯誤:")
    for k, v in errors.items():
        print(f"  [{k}] {v}")
    try:
        import netCDF4
        print(f"\n已安裝 netCDF4 {netCDF4.__version__}; 直接用它打開:")
        d0 = netCDF4.Dataset(path)
        print(d0)
    except Exception as e:                                   # noqa: BLE001
        print(f"  netCDF4 直接開啟也失敗: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n[維度]", dict(ds.sizes))

print("\n[座標]")
for name, c in ds.coords.items():
    v = c.values
    line = f"  {name:12s} shape={v.shape}"
    if np.issubdtype(v.dtype, np.number) and v.size > 1:
        d = np.diff(np.sort(v.astype(float)))
        d = d[d != 0]
        line += f"  範圍 {float(v.min()):.4g} ~ {float(v.max()):.4g}"
        if len(d):
            line += f"  間距(中位數) {float(np.median(d)):.4g}"
    elif np.issubdtype(v.dtype, np.datetime64) and v.size > 1:
        t = np.sort(v)
        dt = np.diff(t).astype("timedelta64[m]").astype(float)
        line += f"  {str(t[0])[:16]} ~ {str(t[-1])[:16]}  間隔(分鐘) 中位數 {np.median(dt):.0f}, 最大 {dt.max():.0f}"
    if name in ("depth", "lev", "level") and v.size > 1 and np.issubdtype(v.dtype, np.number):
        line += "  前 5 個: " + ", ".join(f"{float(x):g}" for x in v[:5])
    print(line)

print("\n[變數]")
tdim = next((d for d in ("time", "valid_time") if d in ds.dims), None)
for name, v in ds.data_vars.items():
    attrs = v.attrs
    print(f"  {name:10s} dims={v.dims} shape={v.shape}")
    print(f"             long_name={attrs.get('long_name', '')!r}  units={attrs.get('units', '')!r}")
    enc = v.encoding
    if enc.get("scale_factor") is not None or enc.get("_FillValue") is not None:
        print(f"             壓縮參數: scale_factor={enc.get('scale_factor')} add_offset={enc.get('add_offset')} "
              f"_FillValue={enc.get('_FillValue')} dtype={enc.get('dtype')}")
    try:
        sl = v.isel({tdim: 0}) if tdim and tdim in v.dims else v
        arr = np.asarray(sl.values, dtype=float)
        print(f"             第一個時間切片: 缺值 {np.isnan(arr).mean():.1%}; "
              f"最小 {np.nanmin(arr):.4g}, 最大 {np.nanmax(arr):.4g}")
    except Exception as e:                                   # noqa: BLE001
        print(f"             (無法統計: {e})")

print("\n[全域屬性(節錄)]")
for k in list(ds.attrs)[:8]:
    print(f"  {k}: {str(ds.attrs[k])[:100]}")
