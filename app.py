@st.cache_data(ttl=1800)
def get_realtime_marine_data(lat, lon):

    # ===== 波高 API =====
    marine_url = "https://marine-api.open-meteo.com/v1/marine"

    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height",
        "timezone": "Asia/Taipei"
    }

    marine = requests.get(marine_url, params=marine_params).json()

    wave = None

    try:
        wave = marine["hourly"]["wave_height"][0]
    except:
        pass

    # 如果沒有波高 -> 往海上偏移再抓
    if wave is None:

        marine_params["latitude"] = lat + 0.2
        marine_params["longitude"] = lon + 0.2

        marine = requests.get(marine_url, params=marine_params).json()

        try:
            wave = marine["hourly"]["wave_height"][0]
        except:
            wave = None

    # ===== 風速 API =====
    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Taipei"
    }

    weather = requests.get(weather_url, params=weather_params).json()

    try:
        wind_speed = float(weather["hourly"]["wind_speed_10m"][0])
        wind_dir = float(weather["hourly"]["wind_direction_10m"][0])
    except:
        wind_speed = None
        wind_dir = None

    return wave, wind_speed, wind_dir
