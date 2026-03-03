# ===============================
# 9. 側邊欄操作 (起點終點即時更新航線)
# ===============================
with st.sidebar:
    st.header("🚢 導航控制")
    
    # GPS 起點
    if st.button("📍 GPS定位起點"):
        lat, lon = gps_position()
        st.session_state.ship_lat = lat
        st.session_state.ship_lon = lon
    
    # 起點終點輸入
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")
    
    # 每次改動都更新 session_state
    st.session_state.ship_lat, st.session_state.ship_lon = slat, slon
    st.session_state.dest_lat, st.session_state.dest_lon = elat, elon
    
    # 檢查陸地
    if is_on_land(slat, slon):
        st.error("❌ 起點在陸地，請重新選擇！")
        st.session_state.real_p = []
    elif is_on_land(elat, elon):
        st.error("❌ 終點在陸地，請重新選擇！")
        st.session_state.real_p = []
    else:
        # 自動生成智慧航線
        st.session_state.real_p = smart_route(
            [slat, slon],
            [elat, elon],
            u, v, lons, lats
        )
        st.session_state.step_idx = 0
        st.session_state.rerun_flag = True
