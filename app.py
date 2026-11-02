"""
app.py -- HELIOS System (Streamlit 介面)

與舊版相同的部分: Streamlit 介面、HYCOM / GFS / GFS-Wave 資料抓取 (data_live.py), 畫面配置。
改掉的部分:
  - 尋路核心: 改用 baseline.py + planner.py + fuel_model.py, 與說明書的節能評估是同一套演算法
  - 船型: 13 類船舶參數庫 (ship_profiles_v2.csv), 並可輸入吃水 (空載 / 半載 / 滿載或自訂)
  - 成本: 不再用手設的環境權重; 油耗由物理模型 (風、浪、流的阻力與功率) 計算

執行:  streamlit run app.py
需要同資料夾: ship_profiles_v2.csv、fuel_model.py、baseline.py、planner.py、bathymetry.py、
              analyze_speed.py、voyages.py、monthly_corr.py、ais_profile.py、env_fields.py、
              data_live.py、helios_core.py、helios_map.py   (水深限制另需 *MDTM*.tif)
"""
import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bathymetry as BY          # noqa: E402
import data_live as DL           # noqa: E402
import fuel_model as FM          # noqa: E402
import helios_core as HC         # noqa: E402
import helios_map as HM          # noqa: E402
from analyze_speed import build_dist_grid, load_land   # noqa: E402
from voyages import PORTS        # noqa: E402

st.set_page_config(layout="wide", page_title="HELIOS System")
st.title("🛰️ HELIOS System")

# ===============================
# 禁航區與離岸風場 (沿用舊版座標; 現在直接當作不可通行)
# ===============================
NO_GO_ZONES = [
    [[22.953536, 120.171678], [22.934628, 120.175472], [22.933136, 120.170942], [22.95781, 120.16078]],
    [[22.943956, 120.172358], [22.939717, 120.173944], [22.928353, 120.157372], [22.936636, 120.153547]],
    [[22.933136, 120.170942], [22.924847, 120.172583], [22.915003, 120.159022], [22.931536, 120.155772]],
]
OFFSHORE_WIND = [
    [[24.18, 120.12], [24.22, 120.28], [24.05, 120.35], [24.00, 120.15]],
    [[24.00, 120.10], [24.05, 120.32], [23.90, 120.38], [23.85, 120.15]],
    [[23.88, 120.05], [23.92, 120.18], [23.75, 120.25], [23.70, 120.08]],
    [[23.68, 120.02], [23.72, 120.12], [23.58, 120.15], [23.55, 120.05]],
]
BUFFER_KM = 5.0          # 與評估一致: 基準路徑離岸 >= 5 km


# ===============================
# 載入 (有快取)
# ===============================
@st.cache_resource
def get_dist():
    return build_dist_grid(load_land(None))


@st.cache_resource
def get_router(depth_on, tif_path):
    lons, lats = HC.grid_axes()
    blocked = HC.polygons_mask(lons, lats, NO_GO_ZONES + OFFSHORE_WIND)
    depth = BY.load_depth_grid(tif_path, lons, lats) if (depth_on and tif_path) else None
    return HC.Router(get_dist(), BUFFER_KM, blocked=blocked, depth=depth)


profile_path = os.path.join(HERE, "ship_profiles_v2.csv")
if not os.path.exists(profile_path):
    st.error("找不到 ship_profiles_v2.csv (船舶參數庫), 請放在 app.py 同一個資料夾。")
    st.stop()
PROFILES = HC.load_profiles(profile_path)
CLASS_KEYS = [k for k in HC.CLASS_LABELS if k in PROFILES]
tif_files = glob.glob(os.path.join(HERE, "*MDTM*.tif"))
TIF = tif_files[0] if tif_files else None

with st.spinner("載入 HYCOM 海流時間序列中..."):
    hycom = DL.load_hycom_series()
with st.spinner("載入氣象（風/浪）時間序列中..."):
    weather = DL.fetch_weather_series()
env = DL.LiveEnv(hycom, weather)
T0 = env.now                                   # 出發時間 = 資料抓取當下 (UTC)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("Route Settings")
    port_names = ["自訂座標"] + list(PORTS)
    sp = st.selectbox("起點", port_names, index=port_names.index("Taichung"))
    if sp == "自訂座標":
        s_lon = st.number_input("Start Lon", 118.0, 124.0, 120.49)
        s_lat = st.number_input("Start Lat", 21.0, 26.0, 24.28)
    else:
        s_lon, s_lat = PORTS[sp][0], PORTS[sp][1]
    ep = st.selectbox("終點", port_names, index=port_names.index("Suao"))
    if ep == "自訂座標":
        e_lon = st.number_input("End Lon", 118.0, 124.0, 121.89)
        e_lat = st.number_input("End Lat", 21.0, 26.0, 24.59)
    else:
        e_lon, e_lat = PORTS[ep][0], PORTS[ep][1]

    st.divider()
    st.subheader("船型與載重")
    ship_key = st.selectbox("船型 (13 類船舶參數庫)", CLASS_KEYS, format_func=lambda k: HC.CLASS_LABELS[k],
                            index=CLASS_KEYS.index("cargo_m") if "cargo_m" in CLASS_KEYS else 0)
    prof = PROFILES[ship_key]
    st.caption(f"典型船長 {prof['loa']:.0f} m、船寬 {prof['beam']:.1f} m、服務航速 {prof['vs']:.1f} 節")
    if ship_key in HC.DRAFT_VARIES:
        load = st.radio("載重狀態 (吃水)", ["空載 (P10)", "半載 (P50)", "滿載 (P90)", "自訂"], index=1, horizontal=True)
        draft = {"空載 (P10)": prof["d10"], "半載 (P50)": prof["d50"], "滿載 (P90)": prof["d90"]}.get(load)
        if draft is None:
            draft = st.number_input("吃水 (m)", 1.0, 25.0, float(prof["d50"]), 0.1)
        st.caption(f"吃水 {draft:.1f} m (此船型同一艘船的吃水會隨裝載明顯變動)")
    else:
        draft = st.number_input("吃水 (m)", 0.5, 25.0, float(prof["d50"]), 0.1)
        st.caption("此船型的 AIS 吃水多半是預設值、不會隨載重更新, 此欄位僅作估計。")
    speed_kn = st.number_input("目標航速 (節)", 3.0, 35.0, float(round(prof["vs"], 1)), 0.5,
                               help="決定要求的到達時間; 實際對水速度由規劃器解出『剛好準時』的速度。")

    st.divider()
    corridor = st.slider("規劃搜尋範圍 (離基準路徑 km)", 20, 100, 60, 10)
    depth_on = False
    if TIF:
        depth_on = st.checkbox("套用水深限制 (依吃水)", value=False,
                               help="用海底地形資料排除對這個吃水太淺的海域; 解析度粗, 港口附近不可靠。")
    progress_pct = st.slider("航行進度 (%)", 0, 100, 0, key="progress_slider")
    st.caption("拖動滑桿可查看船舶航行至該進度時，對應時刻的風/浪/流底圖與剩餘資訊。")

# ===============================
# 規劃 (輸入改變才重算)
# ===============================
ship = HC.make_ship(ship_key, PROFILES, draft, speed_kn)
router = get_router(depth_on, TIF)
req_depth = BY.required_depth_m(ship, hs=1.5) if depth_on else None
route_key = (round(s_lon, 4), round(s_lat, 4), round(e_lon, 4), round(e_lat, 4), ship_key, round(draft, 2),
             round(speed_kn, 2), corridor, depth_on, str(T0))
if st.session_state.get("route_key") != route_key:
    with st.spinner("HELIOS 尋路引擎正依據船型、吃水與預報風浪流進行規劃..."):
        st.session_state.route = HC.route_plan(router, env, ship, (s_lon, s_lat), (e_lon, e_lat), speed_kn,
                                               T0, corridor, req_depth=req_depth)
    st.session_state.route_key = route_key
R = st.session_state.route
if R["error"]:
    st.error("❌ " + R["error"])
    st.stop()

tl = R["timeline"]
res = R["chosen"]["res"]
base_res = R["base"]["res"]
n = len(tl["lon"])
idx = int(round(progress_pct / 100 * (n - 1)))
elapsed_h = float(tl["cum_h"][idx])
remaining_km = float(tl["cum_km"][-1] - tl["cum_km"][idx])
remaining_h = float(tl["cum_h"][-1] - tl["cum_h"][idx])
heading = float(tl["heading"][idx])
seg_env = tl["seg_env"][min(idx, len(tl["seg_env"]) - 1)]
t_map = T0 + np.timedelta64(int(round(elapsed_h * 3600)), "s")

# ===============================
# Dashboard
# ===============================
st.subheader("Navigation Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Remaining Distance (km)", f"{remaining_km:.1f}")
c2.metric("Remaining Time (hr)", f"{remaining_h:.2f}")
c3.metric("Heading", f"{heading:.0f}°")
c4.metric("對水速度 (節)", f"{res['V_kn']:.1f}")

f1, f2, f3, f4 = st.columns(4)
f1.metric("預估燃油 (噸)", f"{res['fuel_t']:.2f}")
f2.metric("基準路徑燃油 (噸)", f"{base_res['fuel_t']:.2f}")
f3.metric("相對基準節省", f"{R['saving_pct']:.1f} %")
f4.metric("總航程 (km)", f"{res['dist_km']:.0f}")
if not R["used_plan"]:
    st.info("在目前的預報海況下，規劃路徑沒有比基準路徑省油，因此沿用基準路徑。")
if R["stretched"] > 1.0:
    st.warning(f"目標航速在預報海況下趕不上，已把要求的航行時間放寬 {100 * (R['stretched'] - 1):.0f}%。")

w1, w2, w3 = st.columns(3)
swh = seg_env.get("swh", np.nan)
w1.metric("顯著波高（目前位置）", "—" if not np.isfinite(swh) else f"{swh:.2f} m")
wsp = float(np.hypot(seg_env.get("u10", np.nan), seg_env.get("v10", np.nan)))
w2.metric("風速（目前位置）", "—" if not np.isfinite(wsp) else f"{wsp:.1f} m/s")
csp = float(np.hypot(seg_env.get("u_cur", np.nan), seg_env.get("v_cur", np.nan)))
w3.metric("海流（目前位置）", "—" if not np.isfinite(csp) else f"{csp:.2f} m/s")


def _ok(x):
    return "✅ 已連接" if x else "❌ 未連接"


st.caption(f"資料連接狀態｜HYCOM 海流: {_ok(True)} ｜ 波浪: {_ok(env.wave is not None)} ｜ 風場: {_ok(env.wind is not None)}"
           f"　｜　出發時間 (UTC) {str(T0)[:16]}　｜　底圖時刻 {str(t_map)[:16]}")

# ===============================
# Map
# ===============================
snap = env.snapshot(t_map)
fig = HM.draw_map(
    snap,
    ((R["base"]["lon"], R["base"]["lat"]), (tl["lon"], tl["lat"])),
    [("red", NO_GO_ZONES), ("yellow", OFFSHORE_WIND)],
    (tl["lon"][idx], tl["lat"][idx]), idx, (s_lon, s_lat), (e_lon, e_lat))
st.pyplot(fig)

with st.expander("這條航線怎麼算的 / 限制"):
    st.markdown(
        f"""
- **基準路徑**：離岸 ≥ {BUFFER_KM:g} km 的海上最短路徑 (避開禁航區與離岸風場{'，並依吃水排除過淺海域' if depth_on else ''})。
- **規劃**：在基準路徑兩側 {corridor} km 內，依預報的海流、風、浪計算每一段的油耗，找出「油耗 + λ × 時間」最小的路線。
- **公平比較**：基準與規劃路徑在**同樣的出發與到達時間**下各自解出剛好準時的對水速度，再比燃油。
- **船**：{HC.CLASS_LABELS[ship_key]}，船長 {ship.loa:.0f} m、船寬 {ship.beam:.1f} m、吃水 {ship.draft:.1f} m、
  排水量約 {ship.disp_t:,.0f} t、服務功率約 {ship.ps_kw:,.0f} kW。
- **限制**：全程固定對水速度；海況取名目時間；風 0.25°、浪 0.5° 解析度在規劃範圍內變化很小，效益主要來自海流；
  預報有誤差；阻力模型係數 (k_aw、cx、受風面積、主機功率上限) 是假設值，未用文獻校準。
"""
    )
