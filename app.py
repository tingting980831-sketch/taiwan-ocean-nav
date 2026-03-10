import streamlit as st
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------
# 1. Page setup
# -----------------------------
st.set_page_config(page_title="AI Ocean Navigation Dashboard", layout="wide")
st.title("⚓ Smart Obstacle Avoidance & Ocean Data")

# -----------------------------
# 2. Load HYCOM data (currents + wind)
# -----------------------------
@st.cache_data
def load_hycom_data():
    HYCOM_URL = "your_hycom_dataset_url_here"  # replace with actual
    ds = xr.open_dataset(HYCOM_URL)

    # Ocean currents
    u = ds["water_u"].isel(time=-1)
    v = ds["water_v"].isel(time=-1)

    # Wind field
    wind_u = ds["wind_u"].isel(time=-1)
    wind_v = ds["wind_v"].isel(time=-1)

    obs_time = str(ds["time"].values[-1])

    wind_ok = (
        wind_u is not None
        and wind_v is not None
        and not np.isnan(wind_u.values).all()
    )

    return u, v, wind_u, wind_v, obs_time, wind_ok

u, v, wind_u, wind_v, obs_time, wind_ok = load_hycom_data()

# -----------------------------
# 3. Small caption for HYCOM + wind status
# -----------------------------
wind_status = "Loaded" if wind_ok else "Not Connected"
st.caption(f"HYCOM observation time: {obs_time} | Wind data: {wind_status}")

# -----------------------------
# 4. Plot setup
# -----------------------------
fig = plt.figure(figsize=(12,6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Coastlines
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, linestyle=':')

# -----------------------------
# 5. Plot ocean current magnitude
# -----------------------------
speed = np.sqrt(u**2 + v**2)
pcm = ax.pcolormesh(
    u["lon"], u["lat"], speed,
    cmap="viridis", shading="auto"
)
cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', label='Current speed (m/s)')

# -----------------------------
# 6. Plot wind vectors (yellow)
# -----------------------------
stride = 5  # reduce arrow density
ax.quiver(
    wind_u["lon"][::stride], wind_u["lat"][::stride],
    wind_u.values[::stride, ::stride],
    wind_v.values[::stride, ::stride],
    color='yellow', scale=50, width=0.003,
    label="Wind field"
)

# -----------------------------
# 7. Layout info (horizontal bar)
# -----------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Estimated Distance (nm)", "120")
col2.metric("Estimated Time (h)", "8.5")
col3.metric("Route Status", "Safe")

# -----------------------------
# 8. Show figure
# -----------------------------
st.pyplot(fig)
