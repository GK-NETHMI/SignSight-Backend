# =========================================================
# 1. CRITICAL COMPATIBILITY FIXES (Must be at the very top)
# =========================================================
import os
import numpy as np

# Force TensorFlow to use Keras 2 (Legacy) to avoid LSTM 'time_major' errors
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Fix for NumPy 2.0+ compatibility with older TensorFlow/JAX
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'bool_'):
    np.bool_ = np.bool8

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import joblib
import uuid
import traceback
import base64
from io import BytesIO
from PIL import Image

print("=== SignSight Backend Starting ===")

# =========================
# Imports
# =========================
import tensorflow as tf
import keras
from keras.models import load_model
import mediapipe as mp
import keras

# =========================================================
# 2. LSTM MONKEY PATCH (Second layer of defense)
# =========================================================
# This removes 'time_major' from the model config if Keras 3 tries to inject it
orig_lstm_from_config = keras.layers.LSTM.from_config
def new_from_config(cls, config):
    config.pop('time_major', None)
    return orig_lstm_from_config(config)
keras.layers.LSTM.from_config = classmethod(new_from_config)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =========================
# Paths — artifacts live under models-jeran/<variant>/
# Set JERAN_MODEL_VARIANT=bilstm|gru|tcn to switch (default: bilstm).
# Fallback: duplicate training bundle in modeltraining-jeran/ (original names).
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_VARIANT_FILES = {
    "bilstm": ("bilstm_modeltraining2_finetuned_1.h5", "scaler.pkl", "pca.pkl"),
    "gru": ("gru_sign_language_model.h5", "scaler.pkl", "pca.pkl"),
    "tcn": ("tcn_model_best.h5", "scaler_tcn.pkl", "pca_tcn.pkl"),
}
_MODEL_VARIANT = os.environ.get("JERAN_MODEL_VARIANT", "bilstm").strip().lower()
if _MODEL_VARIANT not in _MODEL_VARIANT_FILES:
    _MODEL_VARIANT = "bilstm"

_MODELS_ROOT = os.path.join(BASE_DIR, "models-jeran")
_FALLBACK_ROOT = os.path.join(BASE_DIR, "modeltraining-jeran")
_FALLBACK_H5 = "modeltraining2_finetuned_1.h5"

_h5, _scaler_f, _pca_f = _MODEL_VARIANT_FILES[_MODEL_VARIANT]
_candidate_dir = os.path.join(_MODELS_ROOT, _MODEL_VARIANT)
_candidate_h5 = os.path.join(_candidate_dir, _h5)

if os.path.isfile(_candidate_h5):
    MODEL_DIR = _candidate_dir
    MODEL_PATH = _candidate_h5
    SCALER_PATH = os.path.join(MODEL_DIR, _scaler_f)
    PCA_PATH = os.path.join(MODEL_DIR, _pca_f)
elif os.path.isfile(os.path.join(_FALLBACK_ROOT, _FALLBACK_H5)):
    MODEL_DIR = _FALLBACK_ROOT
    MODEL_PATH = os.path.join(MODEL_DIR, _FALLBACK_H5)
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
else:
    MODEL_DIR = _candidate_dir
    MODEL_PATH = os.path.join(MODEL_DIR, _h5)
    SCALER_PATH = os.path.join(MODEL_DIR, _scaler_f)
    PCA_PATH = os.path.join(MODEL_DIR, _pca_f)

# =========================
# Load Model + Scaler + PCA
# =========================
model = scaler = pca = None
try:
    # Use compile=False to avoid needing custom metrics/optimizers defined during load
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    print("✅ Model, Scaler & PCA Loaded Successfully!")
except Exception as e:
    print(f"❌ Model Loading Failed: {e}")
    traceback.print_exc()

# =========================
# Constants
# =========================
actions_dict = {
    'Beautiful': 'அழகு',
    'Drink': 'குடி',
    'Eat': 'சாப்பிடு',
    'Five': 'ஐந்து',
    'Good': 'நல்லது',
    'Hello': 'வணக்கம்',
    'House': 'வீடு',
    'Love': 'காதல்',
    'Man': 'ஆண்',
    'Mother': 'அம்மா',
    'Run': 'ஓடு',
    'Thank you': 'நன்றி',
    'White': 'வெள்ளை',
    'Yellow': 'மஞ்சள்',
    'You': 'நீ'
}

# Get only the English actions
actions = list(actions_dict.keys())

SEQUENCE_LENGTH = 45
FEATURE_DIM = 1662
REDUCED_DIM = 256

# =========================
# Extract Keypoints
# =========================
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468*3)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

# =========================
# Preprocess
# =========================
def preprocess_sequence(sequence_1662):
    fixed = []
    for frame in sequence_1662:
        if len(frame) != FEATURE_DIM:
            frame = np.zeros(FEATURE_DIM, dtype=np.float32)
        fixed.append(frame)
    
    seq = np.array(fixed, dtype=np.float32)
    X_scaled = scaler.transform(seq)
    X_pca = pca.transform(X_scaled)
    
    return X_pca.reshape(1, SEQUENCE_LENGTH, REDUCED_DIM).astype(np.float32)

# =========================
# Flask App
# =========================
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy' if model is not None else 'error',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.get_json()
        sequence = np.array(data.get('sequence'), dtype=np.float32)
        if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            return jsonify({'error': f'Expected ({SEQUENCE_LENGTH}, {FEATURE_DIM})'}), 400

        model_input = preprocess_sequence(sequence)
        pred = model.predict(model_input, verbose=0)[0]
        
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        english_action = actions[class_idx]
        tamil_action = actions_dict[english_action]

        return jsonify({
            "action": f"{english_action} / {tamil_action}",
            "action_english": english_action,
            "action_tamil": tamil_action,
            "english": english_action,
            "tamil": tamil_action,
            "confidence": confidence
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    video_file = request.files['video']
    temp_path = os.path.join("temp", f"temp_{uuid.uuid4().hex[:8]}.webm")
    os.makedirs("temp", exist_ok=True)
    video_file.save(temp_path)

    try:
        cap = cv2.VideoCapture(temp_path)
        frames = []
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret: break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                frames.append(extract_keypoints(results))
        cap.release()

        # Resample frames to match SEQUENCE_LENGTH
        if len(frames) < SEQUENCE_LENGTH:
            pad = [np.zeros(FEATURE_DIM)] * (SEQUENCE_LENGTH - len(frames))
            frames.extend(pad)
        else:
            start = (len(frames) - SEQUENCE_LENGTH) // 2
            frames = frames[start:start + SEQUENCE_LENGTH]

        model_input = preprocess_sequence(np.array(frames))
        pred = model.predict(model_input, verbose=0)[0]
        class_idx = np.argmax(pred)
        english_action = actions[class_idx]
        tamil_action = actions_dict[english_action]

        if os.path.exists(temp_path): os.remove(temp_path)

        return jsonify({
            "action": f"{english_action} / {tamil_action}",
            "action_english": english_action,
            "action_tamil": tamil_action,
            "english": english_action,
            "tamil": tamil_action,
            "confidence": float(pred[class_idx])
        })
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/webcam_predict', methods=['POST'])
def webcam_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        sequence = data.get('sequence', [])

        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = np.array(image)

        with mp_holistic.Holistic() as holistic:
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            keypoints = extract_keypoints(results)

        sequence.append(keypoints.tolist())
        sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            model_input = preprocess_sequence(np.array(sequence))
            pred = model.predict(model_input, verbose=0)[0]
            class_idx = np.argmax(pred)
            english_action = actions[class_idx]
            tamil_action = actions_dict[english_action]
            
            return jsonify({
                'prediction': f"{english_action} / {tamil_action}",
                'prediction_english': english_action,
                'prediction_tamil': tamil_action,
                'english': english_action,
                'tamil': tamil_action,
                'confidence': float(pred[class_idx]),
                'sequence': sequence
            })
        return jsonify({'message': f'Collecting... {len(sequence)}/{SEQUENCE_LENGTH}', 'sequence': sequence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*70)
    print("🚀 SignSight Backend Running!")
    print(f"Model Loaded : {'✅ Yes' if model else '❌ No'}")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5000)