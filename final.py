import os
import cv2
import json
import glob
import queue
import hashlib
import threading
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import mediapipe as mp

from email_service import send_emotion_report_email

UPLOAD_DIR = "uploads"
REPORT_DIR = "reports"
MODEL_PATH = "models/emotion_model_mobilenet.h5"
IMG_SIZE = 224
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Fear"]
EXPECTED_EMOTIONS = ["happy", "sad", "angry"]

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = None
try:
    model = keras.models.load_model(MODEL_PATH)
    model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)), verbose=0)
except:
    model = None

mp_face = mp.solutions.face_detection

processing_queue = queue.Queue()
processing_tasks = {}
task_lock = threading.Lock()

def preprocess_face(face):
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

def analyze_video(video_path, expected_emotion):
    cap = cv2.VideoCapture(video_path)
    detector = mp_face.FaceDetection(0, 0.5)

    emotion_counts = defaultdict(int)
    emotion_conf = defaultdict(list)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames += 1
        if frames % 3 != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)

        if res.detections:
            d = res.detections[0]
            h, w, _ = frame.shape
            box = d.location_data.relative_bounding_box

            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)

            face = frame[y1:y2, x1:x2]
            if face.size == 0 or model is None:
                continue

            pred = model.predict(preprocess_face(face), verbose=0)[0]
            idx = int(np.argmax(pred))
            emotion = EMOTIONS[idx]

            emotion_counts[emotion] += 1
            emotion_conf[emotion].append(float(pred[idx]))

    cap.release()
    detector.close()

    total = sum(emotion_counts.values())
    dominant = max(emotion_counts, key=emotion_counts.get) if total else "Unknown"
    consistency = (emotion_counts[dominant] / total * 100) if total else 0

    return {
        "video": os.path.basename(video_path),
        "expected": expected_emotion.capitalize(),
        "dominant": dominant,
        "consistency": round(consistency, 2),
        "distribution": {e: round((emotion_counts[e] / total * 100), 2) if total else 0 for e in EMOTIONS},
        "confidence": {e: round(np.mean(emotion_conf[e]), 3) if emotion_conf[e] else 0 for e in EMOTIONS}
    }

def worker():
    while True:
        task = processing_queue.get()
        email = task["email"]
        videos = task["videos"]
        results = []

        for v in videos:
            results.append(analyze_video(v["path"], v["emotion"]))

        accuracy = sum(1 for r in results if r["expected"] == r["dominant"]) / len(results) * 100

        report = {
            "email": email,
            "date": datetime.now().isoformat(),
            "accuracy": round(accuracy, 2),
            "results": results
        }

        report_path = os.path.join(REPORT_DIR, f"{email.replace('@','_')}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        send_emotion_report_email(email, report)

        for v in videos:
            try:
                os.remove(v["path"])
            except:
                pass


        with task_lock:
            processing_tasks[task["id"]] = {"status": "completed"}

        processing_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

@app.route("/upload-emotion-video", methods=["POST"])
def upload():
    email = request.form["email"]
    step = int(request.form["step"])
    video = request.files["video"]

    emotion = EXPECTED_EMOTIONS[step]
    user_dir = os.path.join(UPLOAD_DIR, email.replace("@", "_"))
    os.makedirs(user_dir, exist_ok=True)

    filename = f"{emotion}_{datetime.now().strftime('%Y%m%d%H%M%S')}.webm"
    path = os.path.join(user_dir, filename)
    video.save(path)

    files = []
    for e in EXPECTED_EMOTIONS:
        matches = glob.glob(os.path.join(user_dir, f"{e}_*.webm"))
        if matches:
            files.append({"emotion": e, "path": max(matches, key=os.path.getctime)})

    if len(files) == 3:
        task_id = hashlib.md5(f"{email}{datetime.now()}".encode()).hexdigest()
        processing_queue.put({"id": task_id, "email": email, "videos": files})
        processing_tasks[task_id] = {"status": "processing"}
        return jsonify({"status": "processing", "task_id": task_id})

    # 
    return jsonify({"status": "saved", "count": len(files)})

@app.route("/status/<task_id>")
def status(task_id):
    return jsonify(processing_tasks.get(task_id, {"status": "unknown"}))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
