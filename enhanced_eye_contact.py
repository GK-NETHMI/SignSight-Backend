import os
from collections import deque
import warnings


def _patch_protobuf_symbol_database():
    """Keep older MediaPipe protobuf code working with newer protobuf releases."""
    try:
        from google.protobuf import message_factory, symbol_database
    except Exception:
        return

    if hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
        return

    def get_prototype(self, descriptor):
        if hasattr(message_factory, "GetMessageClass"):
            return message_factory.GetMessageClass(descriptor)
        return message_factory.MessageFactory().GetPrototype(descriptor)

    symbol_database.SymbolDatabase.GetPrototype = get_prototype


_patch_protobuf_symbol_database()

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

MODEL_PATH = "suspicious_model.h5"
IMG_SIZE = 224
MODEL_CLASSES = [
    "EyeContact",
    "LookLeft",
    "LookRight",
    "LookDown",
    "LookUp",
    "NoFace",
    "EyesClosed",
]

MODEL_CONF_THRESHOLD = 0.60
FRAME_SKIP = 2
PREDICTION_BUFFER = 7


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


class Smoother:
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)

    def push(self, probs: np.ndarray) -> None:
        self.buffer.append(probs)

    def get(self):
        if not self.buffer:
            return None
        weights = np.linspace(0.5, 1.0, len(self.buffer))
        weights = weights / weights.sum()
        return np.average(self.buffer, axis=0, weights=weights)

    def clear(self):
        self.buffer.clear()


def enable_gpu_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def load_model_safe(model_path):
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"✓ Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"✗ Model load error: {e}")
        return None


def preprocess_for_model(face_bgr: np.ndarray) -> np.ndarray:
    face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32")
    face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
    return np.expand_dims(face, axis=0)


def _expand_and_clip_bbox(x1, y1, x2, y2, frame_w, frame_h, pad_ratio=0.15):
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    return x1, y1, x2, y2


def analyze_video_enhanced(video_path):
    print("Analyzing with Enhanced Model:", video_path)

    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )

    enable_gpu_growth()
    model = None
    model_available = False

    if os.path.exists(MODEL_PATH):
        try:
            model = load_model_safe(MODEL_PATH)
            if model is not None:
                dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                _ = model.predict(dummy, verbose=0)
                model_available = True
                print(f"✓ Loaded model: {MODEL_PATH}")
            else:
                print("✗ Model load failed - will use detection-only fallback")
        except Exception as e:
            print(f"✗ Model inference test failed: {e}")
            model = None
            model_available = False
    else:
        print(f"✗ Model file not found: {MODEL_PATH}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        try:
            face_detector.close()
        except Exception:
            pass
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps <= 0:
        fps = 30

    results = {
        "total_frames": 0,
        "processed_frames": 0,
        "face_detected_frames": 0,
        "face_not_detected_frames": 0,
        "final_predictions": {cls: 0 for cls in MODEL_CLASSES},
        "model_predictions": {cls: 0 for cls in MODEL_CLASSES},
        "average_confidence": {cls: [] for cls in MODEL_CLASSES},
    }

    model_smoother = Smoother(PREDICTION_BUFFER)
    frame_no = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            results["total_frames"] += 1

            if frame_no % FRAME_SKIP != 0:
                continue

            results["processed_frames"] += 1
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            det_results = face_detector.process(rgb)
            face_detected = bool(det_results.detections)

            face_crop = None
            final_label = "NoFace"

            if face_detected:
                results["face_detected_frames"] += 1
                detection = det_results.detections[0]
                bbox = detection.location_data.relative_bounding_box

                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, x1 + int(bbox.width * w))
                y2 = min(h, y1 + int(bbox.height * h))

                x1, y1, x2, y2 = _expand_and_clip_bbox(x1, y1, x2, y2, w, h, pad_ratio=0.18)

                if x2 > x1 and y2 > y1:
                    face_crop = frame[y1:y2, x1:x2]
            else:
                results["face_not_detected_frames"] += 1
                model_smoother.clear()

            if model_available and face_crop is not None and face_crop.size > 0:
                try:
                    model_input = preprocess_for_model(face_crop)
                    raw = model.predict(model_input, verbose=0)[0]
                    model_smoother.push(raw)
                    probs = model_smoother.get()

                    if probs is not None:
                        idx = int(np.argmax(probs))
                        model_label = MODEL_CLASSES[idx]
                        model_conf = float(probs[idx])

                        results["model_predictions"][model_label] += 1
                        results["average_confidence"][model_label].append(model_conf)

                        final_label = model_label
                    else:
                        final_label = "NoFace" if not face_detected else "EyeContact"

                except Exception as e:
                    print(f"Model prediction error: {e}")
                    final_label = "NoFace" if not face_detected else "EyeContact"
            else:
                if face_detected:
                    final_label = "EyeContact"
                else:
                    final_label = "NoFace"

            if not face_detected:
                final_label = "NoFace"

            results["final_predictions"][final_label] += 1

    finally:
        cap.release()
        try:
            face_detector.close()
        except Exception:
            pass

    processed_total = results["processed_frames"] or 1

    avg_conf_summary = {}
    for cls, confs in results["average_confidence"].items():
        avg_conf_summary[cls] = round(float(np.mean(confs)), 4) if confs else 0.0

    dominant_label = max(results["final_predictions"], key=results["final_predictions"].get)
    face_detected_pct = round(results["face_detected_frames"] / processed_total * 100, 1)
    no_face_pct = round(results["face_not_detected_frames"] / processed_total * 100, 1)

    final_prediction_percentages = {
        cls: round(results["final_predictions"][cls] / processed_total * 100, 1)
        for cls in MODEL_CLASSES
    }

    return {
        "video_duration": round(results["total_frames"] / fps, 1),
        "processed_frames": results["processed_frames"],
        "face_detection": {
            "detected_frames": results["face_detected_frames"],
            "not_detected_frames": results["face_not_detected_frames"],
            "detected_percentage": f"{face_detected_pct}%",
            "not_detected_percentage": f"{no_face_pct}%"
        },
        "dominant_prediction": dominant_label,
        "final_prediction_statistics": results["final_predictions"],
        "final_prediction_percentages": final_prediction_percentages,
        "model_statistics": results["model_predictions"],
        "average_model_confidence": avg_conf_summary,
        "model_enabled": model_available,
        "notes": [
            "Face mesh removed as requested.",
            "Yaw, pitch, eye aspect ratio, and head-pose analytics are not available in this version.",
            "Predictions are generated from face-detection crop + model inference."
        ]
    }


def video_analyze_enhanced(path):
    try:
        return analyze_video_enhanced(path)
    except Exception as e:
        return {"error": str(e)}
