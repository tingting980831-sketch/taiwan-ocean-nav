with st.sidebar:
    st.header("🚢 導航控制")

    # 起點輸入
    slat = st.number_input("起點緯度", value=st.session_state.ship_lat, format="%.3f")
    slon = st.number_input("起點經度", value=st.session_state.ship_lon, format="%.3f")

    # 定位目前船位
    if st.button("📍 定位目前位置"):
        # 假設這裡可以從 GPS 或模擬取得座標，先用隨機或預設值模擬
        st.session_state.ship_lat = 25.060   # 模擬目前緯度
        st.session_state.ship_lon = 122.200  # 模擬目前經度
        slat, slon = st.session_state.ship_lat, st.session_state.ship_lon
        st.success("已更新為目前位置！")
        st.experimental_rerun()  # 更新 sidebar 顯示

    # 終點輸入
    elat = st.number_input("終點緯度", value=st.session_state.dest_lat, format="%.3f")
    elon = st.number_input("終點經度", value=st.session_state.dest_lon, format="%.3f")

    if st.button("🚀 啟動智慧航線", use_container_width=True):

        if is_land(slat, slon):
            st.error("起點在陸地！")
        elif is_land(elat, elon):
            st.error("終點在陸地！")
        else:
            path = smart_route([slat, slon], [elat, elon])

            st.session_state.ship_lat = slat
            st.session_state.ship_lon = slon
            st.session_state.dest_lat = elat
            st.session_state.dest_lon = elon
            st.session_state.real_p = path
            st.session_state.step_idx = 0

            st.experimental_rerun()
