from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import sqlite3

app = Flask(__name__)

camera = cv2.VideoCapture(0)

# トレーニング済みモデルのロード
model = load_model('model/mobilenetv2_model.h5')


def preprocess_image(frame):
    image_resized = cv2.resize(frame, (128, 128))
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)


def predict_leaf_count(frame):
    preprocessed_frame = preprocess_image(frame)
    prediction = model.predict(preprocessed_frame)
    return int(prediction[0])


def init_db():
    conn = sqlite3.connect('data/growth_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS growth (date TEXT, leaf_count INTEGER)''')
    conn.commit()
    conn.close()


def insert_data(date, leaf_count):
    conn = sqlite3.connect('data/growth_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO growth (date, leaf_count) VALUES (?, ?)", (date, leaf_count))
    conn.commit()
    conn.close()


def fetch_data_by_date(date):
    conn = sqlite3.connect('data/growth_data.db')
    c = conn.cursor()
    c.execute("SELECT leaf_count FROM growth WHERE date=?", (date,))
    data = c.fetchone()
    conn.close()
    return data[0] if data else None


def compare_growth():
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    today_count = fetch_data_by_date(today)
    yesterday_count = fetch_data_by_date(yesterday)
    last_week_count = fetch_data_by_date(last_week)

    growth_yesterday = today_count - yesterday_count if today_count and yesterday_count else None
    growth_last_week = today_count - last_week_count if today_count and last_week_count else None

    return growth_yesterday, growth_last_week


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                leaf_count = predict_leaf_count(frame)
                cv2.putText(frame, f'Leaf Count: {leaf_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture')
def capture():
    success, frame = camera.read()
    if success:
        leaf_count = predict_leaf_count(frame)
        date_str = datetime.now().strftime('%Y-%m-%d')
        insert_data(date_str, leaf_count)
    return "Image Captured and Data Saved!"


@app.route('/compare')
def compare():
    growth_yesterday, growth_last_week = compare_growth()
    return render_template('compare.html', growth_yesterday=growth_yesterday, growth_last_week=growth_last_week)


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5001)
