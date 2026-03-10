import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="AI 海象導航系統", layout="wide")
st.title("⚓ AI 海象導航系統")

# ===============================
# 即時風資料（Open-Meteo）
# ===============================
@st.cache_data(ttl=1800)
def get_wind_data(lat, lon):

    url = "https://marine-api.open-meteo.com/v1/marine"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "wind_speed_10m",
            "wind_direction_10m"
        ],
        "timezone": "Asia/Taipei",
        "forecast_days": 1
    }

    res = requests.get(url, params=params, timeout=30)
    data = res.json()

    hourly = data.get("hourly", {})

    wind_speed = hourly.get("wind_speed_10m", [0])[0]
    wind_dir = hourly.get("wind_direction_10m", [0])[0]

    return wind_speed, wind_dir


# ===============================
# 完全禁止海域（半透明紅）
# ===============================
restricted_zones = [

    [
        (120.171678,22.953536),
        (120.175472,22.934628),
        (120.170942,22.933136),
        (120.160780,22.957810)
    ],

    [
        (120.172358,22.943956),
        (120.173944,22.939717),
        (120.157372,22.928353),
        (120.153547,22.936636)
    ],

    [
        (120.17547,22.88462),
        (120.17696,22.91242),
        (120.17214,22.91178),
        (120.170942,22.933136)
    ],

    [
        (119.5,23.280833),
        (119.509722,23.280833),
        (119.509722,23.274444),
        (119.5,23.274444)
    ],

    [
        (119.492222,23.3),
        (119.689444,23.3),
        (119.689444,23.225),
        (119.49222,23.225)
    ],

    [
        (121.872265,24.583736),
        (121.874346,24.585287),
        (121.872801,24.583521),
        (121.874067,24.584590)
    ],

    [
        (120.58407,24.38367),
        (120.5833427,24.38367),
        (120.5842898,24.383464),
        (120.5849261,24.384821)
    ],

    [
        (120.856069,24.6995),
        (120.847477,24.689371),
        (120.838557,24.698345),
        (120.846797,24.705380)
    ]
]


# ===============================
# 地圖繪製
# ===============================
def draw_map():

    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 範圍（台灣）
    ax.set_extent([118, 123, 21, 26])

    # 原本底圖（不改配色）
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # ===============================
    # 畫禁止海域（透明紅）
    # ===============================
    for zone in restricted_zones:

        poly = Polygon(
            zone,
            closed=True,
            facecolor="red",
            edgecolor="darkred",
            alpha=0.35,
            transform=ccrs.PlateCarree(),
            zorder=5
        )

        ax.add_patch(poly)

    return fig


# ===============================
# 主程式
# ===============================

lat = st.sidebar.number_input("緯度", value=24.2)
lon = st.sidebar.number_input("經度", value=120.4)

wind_speed, wind_dir = get_wind_data(lat, lon)

st.sidebar.write(f"💨 風速：{wind_speed} m/s")
st.sidebar.write(f"🧭 風向：{wind_dir}°")

fig = draw_map()
st.pyplot(fig)
