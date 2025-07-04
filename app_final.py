import streamlit as st
import os
import cv2
from ultralytics import YOLO
import time
import datetime
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
import paho.mqtt.client as mqtt
from streamlit_js_eval import streamlit_js_eval

# --- StreamlitのUI設定 ---
st.set_page_config(page_title="外来種キャッチャー", layout="wide")
st.title("🐟 外来種キャチャー（外来種判別アプリ）")

# --- 設定項目 ---
SERVICE_ACCOUNT_KEY_PATH = "serviceAccountKey.json"
TRAINED_MODEL_PATH = 'best(1).pt'
MQTT_BROKER = 'broker.hivemq.com'
MQTT_PORT = 1883
MQTT_TOPIC = 'otunagi/gate/control'
# -----------------

# --- 設定ファイルの読み込み ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("設定ファイル 'config.yaml' が見つかりません。")
    st.stop()

# --- Firebaseの初期化 ---
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    st.error(f"Firebaseの初期化エラー: {e}")
    st.stop()

# --- MQTTクライアントの初期化 ---
@st.cache_resource
def init_mqtt_client():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTTブローカーとの接続に成功しました。")
        else:
            print(f"❌ MQTTブローカーとの接続に失敗しました: rc={rc}")
    
    client = mqtt.Client()
    client.on_connect = on_connect
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        st.error(f"MQTTブローカーへの接続中にエラー: {e}")
        return None
    return client

mqtt_client = init_mqtt_client()

# --- モデルの読み込み ---
@st.cache_resource
def load_yolo_model():
    path = TRAINED_MODEL_PATH
    if not os.path.exists(path):
        st.error(f"モデルファイル '{path}' が見つかりません。")
        return None
    return YOLO(path)

# --- アプリのコア機能（関数定義）---
def get_cameras_for_user(user_id):
    try:
        cameras_ref = db.collection('users').document(user_id).collection('cameras').stream()
        cameras = {}
        for doc in cameras_ref:
            camera_data = doc.to_dict()
            camera_data['id'] = doc.id
            cameras[camera_data.get('location', "名称未設定")] = camera_data
        return cameras
    except Exception as e:
        st.error(f"カメラ情報の取得中にエラーが発生しました: {e}")
        return {}

def add_camera_for_user(user_id, location, video_source, threshold, coordinates):
    try:
        cameras_ref = db.collection('users').document(user_id).collection('cameras')
        cameras_ref.add({
            'location': location,
            'video_source': video_source,
            'notification_threshold': threshold,
            'coordinates': firestore.GeoPoint(coordinates['lat'], coordinates['lng'])
        })
        st.success(f"新しいカメラ「{location}」を登録しました！")
        time.sleep(1) 
        st.rerun() 
    except Exception as e:
        st.error(f"カメラの登録中にエラーが発生しました: {e}")

def log_detection(camera_id, detected_count):
    try:
        today_str = datetime.date.today().isoformat()
        summary_ref = db.collection('daily_summary').document(camera_id).collection('dates').document(today_str)
        
        @firestore.transactional
        def update_in_transaction(transaction, doc_ref, count):
            snapshot = doc_ref.get(transaction=transaction)
            new_count = count
            if snapshot.exists:
                new_count += snapshot.to_dict().get('total_count', 0)
            transaction.set(doc_ref, {'total_count': new_count}, merge=True)
            return new_count
        
        transaction = db.transaction()
        total_today = update_in_transaction(transaction, summary_ref, detected_count)
        return total_today
    except Exception as e:
        st.error(f"ログ記録中にエラー: {e}")
        return None

def get_weekly_detection_history(camera_id):
    history = {}
    today = datetime.date.today()
    for i in range(6, -1, -1):
        target_date = today - timedelta(days=i)
        date_str = target_date.isoformat()
        doc_ref = db.collection('daily_summary').document(camera_id).collection('dates').document(date_str)
        doc = doc_ref.get()
        display_date = target_date.strftime('%m/%d')
        history[display_date] = doc.to_dict().get('total_count', 0) if doc.exists else 0
    return history

# --- ナビゲーション画面の描画関数 ---
def render_navigation_view():
    st.header(f"ナビゲーション: {st.session_state.navigate_to['location']}")

    if st.button("⬅️ ダッシュボードに戻る"):
        st.session_state.view = 'main'
        st.session_state.user_location = None
        st.rerun()

    if 'user_location' not in st.session_state:
        st.session_state.user_location = None

    if st.button("現在地を取得して経路を検索", use_container_width=True):
        with st.spinner("現在地を検索中..."):
            js_code = "new Promise((resolve, reject) => { navigator.geolocation.getCurrentPosition(resolve, reject, {enableHighAccuracy: true}); }).then(pos => ({lat: pos.coords.latitude, lng: pos.coords.longitude})).catch(err => null);"
            user_loc = streamlit_js_eval(js_expressions=js_code, key="GET_NAV_LOCATION")
            if user_loc:
                st.session_state.user_location = user_loc
            else:
                st.error("現在地の取得に失敗しました。ブラウザの許可設定を確認してください。")

    if st.session_state.user_location:
        user_lat, user_lng = st.session_state.user_location['lat'], st.session_state.user_location['lng']
        cam_info = st.session_state.navigate_to
        cam_coords = cam_info['coordinates']
        cam_lat, cam_lng = cam_coords.latitude, cam_coords.longitude

        distance_km = great_circle((user_lat, user_lng), (cam_lat, cam_lng)).kilometers
        st.metric(label="カメラまでの直線距離", value=f"{distance_km:.2f} km")
        st.divider()

        st.subheader("経路オプション")
        direct_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lng}&destination={cam_lat},{cam_lng}"
        st.link_button("➡️ Googleマップで経路を開く", direct_url, use_container_width=True)
        
        st.divider()

        map_center_lat, map_center_lng = (user_lat + cam_lat) / 2, (user_lng + cam_lng) / 2
        m = folium.Map(location=[map_center_lat, map_center_lng], zoom_start=12)
        
        folium.Marker([user_lat, user_lng], popup="現在地", icon=folium.Icon(color='blue', icon='user')).add_to(m)
        folium.Marker([cam_lat, cam_lng], popup=cam_info['location'], icon=folium.Icon(color='red', icon='camera')).add_to(m)
        
        folium.PolyLine(locations=[[user_lat, user_lng], [cam_lat, cam_lng]], color='blue').add_to(m)
        
        st_folium(m, width=725, height=500, key="nav_map")

# --- ログイン認証 ---
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
name, authentication_status, username = authenticator.login("Login", "main")

# --- メインロジック ---
if st.session_state["authentication_status"]:
    user_id = username
    if 'view' not in st.session_state:
        st.session_state.view = 'main'

    if st.session_state.view == 'navigation':
        render_navigation_view()
    
    else:
        with st.sidebar:
            st.success(f"ようこそ **{name}** さん")
            authenticator.logout('ログアウト')

        model = load_yolo_model()
        cameras = get_cameras_for_user(user_id)

        with st.sidebar:
            st.divider()
            st.header("カメラの新規登録")
            
            col1, col2 = st.columns(2)
            with col1:
                search_query = st.text_input("場所検索", placeholder="例: 琵琶湖")
            with col2:
                if st.button("検索", use_container_width=True):
                    geolocator = Nominatim(user_agent=f"otunagi-app-{user_id}")
                    try:
                        location = geolocator.geocode(search_query, timeout=10)
                        if location:
                            st.session_state.map_center = [location.latitude, location.longitude]
                        else:
                            st.warning("場所が見つかりませんでした。")
                    except Exception as e:
                        st.error(f"検索中にエラー: {e}")
            
            if st.button("現在地を取得して中心に設定", use_container_width=True):
                loc = streamlit_js_eval(js_expressions='(async () => { const pos = await new Promise((resolve, reject) => navigator.geolocation.getCurrentPosition(resolve, reject)); return {"lat": pos.coords.latitude, "lon": pos.coords.longitude} })()', key="geolocation")
                if isinstance(loc, dict):
                    st.session_state.map_center = [loc['lat'], loc['lon']]
                    st.success("現在地を中心に設定しました。")
            
            st.write("下の地図を動かし、中央のピンを設置場所に合わせ登録してください。")
            
            if 'map_center' not in st.session_state:
                st.session_state.map_center = [35.004, 135.862] 

            m_reg = folium.Map(location=st.session_state.map_center, zoom_start=15)
            folium.Marker(st.session_state.map_center, icon=folium.Icon(color='red', icon='camera')).add_to(m_reg)
            map_data_reg = st_folium(m_reg, key="reg_map", width=300, height=250)

            if map_data_reg and map_data_reg.get("center"):
                new_center = [map_data_reg["center"]["lat"], map_data_reg["center"]["lng"]]
                if st.session_state.map_center != new_center:
                    st.session_state.map_center = new_center
                    st.rerun()

            with st.form("new_camera_form_with_map", clear_on_submit=True):
                coords = st.session_state.map_center
                st.write(f"**選択中の座標**: `緯度 {coords[0]:.5f}, 経度 {coords[1]:.5f}`")
                new_location = st.text_input("この場所の名前 *", placeholder="例: 矢橋帰帆島公園")
                new_video_source = st.text_input("動画パスまたはカメラ番号 *", placeholder="例: my_video.mp4 or 0")
                new_threshold = st.number_input("通知しきい値（匹）", min_value=1, value=5)
                
                submitted = st.form_submit_button("この場所でカメラを登録する")
                if submitted and new_location and new_video_source:
                    add_camera_for_user(user_id, new_location, new_video_source, new_threshold, {'lat': coords[0], 'lng': coords[1]})

        if not model:
            st.error("AIモデルの読み込みに失敗しました。")
        elif not cameras:
            st.warning(f"カメラが登録されていません。サイドバーから新しいカメラを登録してください。")
        else:
            st.sidebar.divider()
            st.sidebar.header("カメラの選択と操作")
            selected_location = st.sidebar.selectbox("カメラを選択してください", options=list(cameras.keys()))
            selected_camera_info = cameras[selected_location]
            
            if st.sidebar.button("カメラを回収しに行く", use_container_width=True, type="primary"):
                st.session_state.view = 'navigation'
                st.session_state.navigate_to = selected_camera_info
                st.rerun()

            confidence_threshold = st.sidebar.slider('信頼度のしきい値', 0.0, 1.0, 0.4, 0.01)
            
            if 'running_camera' not in st.session_state:
                st.session_state.running_camera = None

            if st.sidebar.button("解析を開始"):
                st.session_state.running_camera = selected_location
                # 解析開始時にトラッキングIDのセットをリセット（必要に応じて）
                camera_id_for_reset = cameras[selected_location]['id']
                st.session_state[f'seen_ids_{camera_id_for_reset}'] = set()

            if st.sidebar.button("解析を停止"):
                st.session_state.running_camera = None

            col1, col2 = st.columns([3, 1])
            with col1:
                st.header("カメラ映像")
                video_frame_placeholder = st.empty()
            with col2:
                st.header("本日の累計検出数")
                status_placeholder = st.empty()
                
            camera_id = selected_camera_info['id']
            if f'count_{camera_id}' not in st.session_state:
                st.session_state[f'count_{camera_id}'] = 0
            status_placeholder.metric(label=f"📍 {selected_location}", value=f"{st.session_state[f'count_{camera_id}']} 匹")
            
            st.divider()
            st.header(f"📈 {selected_location}の週間検出レポート")
            weekly_history = get_weekly_detection_history(camera_id)
            if any(weekly_history.values()):
                df = pd.DataFrame(list(weekly_history.items()), columns=['日付', '検出数'])
                st.bar_chart(df.set_index('日付'))
            else:
                st.info("過去7日間の検出データがありません。")
                
            if st.session_state.running_camera == selected_location:
                st.sidebar.success("解析を実行中...")
                video_source = selected_camera_info.get('video_source')
                
                ### 修正箇所1: トラッキングIDを記憶するセットを初期化 ###
                if f'seen_ids_{camera_id}' not in st.session_state:
                    st.session_state[f'seen_ids_{camera_id}'] = set()

                try: video_source = int(video_source)
                except (ValueError, TypeError):
                    if not os.path.exists(video_source):
                        st.error(f"動画ファイルが見つかりません: {video_source}")
                        st.session_state.running_camera = None

                if st.session_state.running_camera:
                    cap = cv2.VideoCapture(video_source)
                    while cap.isOpened() and st.session_state.running_camera == selected_location:
                        ret, frame = cap.read()
                        if not ret:
                            st.write("動画の再生が終了しました。")
                            st.session_state.running_camera = None
                            break
                        
                        ### 修正箇所2: predictをtrackに変更し、新規IDのみをカウント ###
                        results = model.track(frame, persist=True, conf=confidence_threshold, classes=[0], verbose=False)
                        annotated_frame = results[0].plot()

                        newly_detected_count = 0
                        if results[0].boxes.id is not None: # トラッキングIDが取得できた場合
                            track_ids = results[0].boxes.id.int().cpu().tolist()

                            for track_id in track_ids:
                                # このIDがまだカウントされていない（初めて見た）場合
                                if track_id not in st.session_state[f'seen_ids_{camera_id}']:
                                    st.session_state[f'seen_ids_{camera_id}'].add(track_id) # 新しいIDとして記憶
                                    newly_detected_count += 1 # 新規検出としてカウント
                        
                        detected_count = newly_detected_count # ログに記録する数を新規検出数に置き換え
                        ### 修正ここまで ###

                        if detected_count > 0 and mqtt_client:
                            mqtt_client.publish(MQTT_TOPIC, "OPEN")
                            
                            total = log_detection(camera_id, detected_count)
                            if total is not None:
                                st.session_state[f'count_{camera_id}'] = total
                                status_placeholder.metric(label=f"📍 {selected_location}", value=f"{total} 匹")
                                threshold = selected_camera_info.get('notification_threshold', 5)
                                if total >= threshold and not st.session_state.get(f'notified_{camera_id}', False):
                                    st.toast(f"🚨 通知: {selected_location}でしきい値超過！ ({total}匹)")
                                    st.session_state[f'notified_{camera_id}'] = True

                        video_frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                        time.sleep(0.1)
                    cap.release()
            else:
                st.sidebar.info("解析は停止しています。")

elif st.session_state["authentication_status"] is False:
    st.error('ユーザー名またはパスワードが間違っています')
elif st.session_state["authentication_status"] is None:
    st.info('メイン画面でログインするか、サイドバーで新規ユーザー登録をしてください。')

# --- 新規ユーザー登録機能 ---
if not st.session_state["authentication_status"]:
    try:
        if authenticator.register_user('新規ユーザー登録', location='sidebar', preauthorization=False):
            st.success('ユーザー登録が成功しました。ログインしてください。')
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        st.error(e)