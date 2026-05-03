import os
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


def patch_protobuf_for_mediapipe():
    """Restore protobuf APIs that older MediaPipe releases still call."""
    try:
        from google.protobuf import message_factory, symbol_database
    except Exception:
        return

    if not hasattr(message_factory, "GetMessageClass"):
        return

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        message_factory.MessageFactory.GetPrototype = staticmethod(
            message_factory.GetMessageClass
        )

    db_class = type(symbol_database.Default())
    if not hasattr(db_class, "GetPrototype"):
        db_class.GetPrototype = staticmethod(message_factory.GetMessageClass)


patch_protobuf_for_mediapipe()

import mediapipe as mp

# =========================
# CONFIG
# =========================
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

USE_MODEL = True
MODEL_CONF_THRESHOLD = 0.60
PREDICTION_BUFFER = 7

# Rule thresholds
YAW_LEFT_THRESHOLD = -18.0
YAW_RIGHT_THRESHOLD = 18.0
PITCH_DOWN_THRESHOLD = 16.0
PITCH_UP_THRESHOLD = -14.0
EYE_CLOSED_EAR_THRESHOLD = 0.20

LOOK_AWAY_ALERT_SECONDS = 2.0
NO_FACE_ALERT_SECONDS = 2.5
EYES_CLOSED_ALERT_SECONDS = 1.2

# Display
WINDOW_NAME = "Behavior Monitoring"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

COLORS = {
    "normal": (0, 200, 0),
    "warning": (0, 165, 255),
    "alert": (0, 0, 255),
    "info": (255, 255, 255),
    "LookLeft": (0, 165, 255),
    "LookRight": (0, 165, 255),
    "LookDown": (0, 0, 255),
    "LookUp": (255, 0, 255),
    "NoFace": (0, 0, 180),
    "EyesClosed": (180, 180, 0),
    "EyeContact": (0, 200, 0),
}

# MediaPipe landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# PnP face model points
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype="double")


def enable_gpu_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def preprocess_for_model(face_bgr: np.ndarray) -> np.ndarray:
    face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32")
    face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
    return np.expand_dims(face, axis=0)


class Smoother:
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)
        self.no_face_counter = 0

    def push(self, probs: np.ndarray) -> None:
        self.buffer.append(probs)
        self.no_face_counter = 0

    def get(self):
        if not self.buffer:
            return None
        weights = np.linspace(0.5, 1.0, len(self.buffer))
        weights = weights / weights.sum()
        return np.average(self.buffer, axis=0, weights=weights)

    def clear(self):
        self.no_face_counter += 1
        if self.no_face_counter > 5:
            self.buffer.clear()


class DurationTracker:
    def __init__(self):
        self.started_at = None

    def update(self, active: bool):
        now = time.time()
        if active:
            if self.started_at is None:
                self.started_at = now
        else:
            self.started_at = None

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.time() - self.started_at


class BehaviorMonitor:
    def __init__(self, use_model: bool = True):
        enable_gpu_growth()
        self.use_model = use_model
        self.model = None
        self.model_available = False
        self.model_smoother = Smoother(PREDICTION_BUFFER)

        if use_model and os.path.exists(MODEL_PATH):
            try:
                self.model = keras.models.load_model(MODEL_PATH)
                dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                _ = self.model.predict(dummy, verbose=0)
                self.model_available = True
                print(f"Loaded model: {MODEL_PATH}")
            except Exception as exc:
                print(f"Model load failed, falling back to rules only: {exc}")

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.look_away_tracker = DurationTracker()
        self.no_face_tracker = DurationTracker()
        self.eyes_closed_tracker = DurationTracker()

        self.events = []
        self.last_event_key = None
        self.last_event_time = 0.0

    def close(self):
        try:
            self.face_detector.close()
        except Exception:
            pass
        try:
            self.face_mesh.close()
        except Exception:
            pass

    def _get_head_pose(self, landmarks, w: int, h: int):
        focal = h
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1],
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        image_points = np.array([
            (landmarks[1][0], landmarks[1][1]),
            (landmarks[152][0], landmarks[152][1]),
            (landmarks[263][0], landmarks[263][1]),
            (landmarks[33][0], landmarks[33][1]),
            (landmarks[287][0], landmarks[287][1]),
            (landmarks[57][0], landmarks[57][1]),
        ], dtype="double")

        ok, rotation_vector, _ = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None, None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])
        return yaw, pitch, roll

    def _get_ear(self, landmarks, indices):
        pts = np.array([(landmarks[i][0], landmarks[i][1]) for i in indices], dtype="double")
        a = np.linalg.norm(pts[1] - pts[5])
        b = np.linalg.norm(pts[2] - pts[4])
        c = np.linalg.norm(pts[0] - pts[3])
        if c <= 0:
            return 0.0
        return float((a + b) / (2.0 * c))

    def _log_event(self, label: str, duration: float):
        now = time.time()
        event_key = f"{label}:{int(now // 2)}"
        if self.last_event_key == event_key and now - self.last_event_time < 2:
            return
        self.last_event_key = event_key
        self.last_event_time = now
        self.events.append({
            "time": time.strftime("%H:%M:%S"),
            "label": label,
            "duration_sec": round(duration, 2),
        })

    def analyze(self, frame_bgr: np.ndarray):
        frame = frame_bgr.copy()
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        det_results = self.face_detector.process(rgb)
        mesh_results = self.face_mesh.process(rgb)

        face_detected = bool(det_results.detections)
        bbox = None
        face_crop = None

        if face_detected:
            detection = det_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, x1 + int(bbox.width * w))
            y2 = min(h, y1 + int(bbox.height * h))
            if x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2]

        yaw = pitch = roll = None
        left_ear = right_ear = None
        eyes_closed = False
        looking_direction = "EyeContact"

        if mesh_results.multi_face_landmarks:
            raw_landmarks = mesh_results.multi_face_landmarks[0].landmark
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks]
            yaw, pitch, roll = self._get_head_pose(landmarks, w, h)
            left_ear = self._get_ear(landmarks, LEFT_EYE)
            right_ear = self._get_ear(landmarks, RIGHT_EYE)
            mean_ear = (left_ear + right_ear) / 2.0
            eyes_closed = mean_ear < EYE_CLOSED_EAR_THRESHOLD

            if yaw is not None and pitch is not None:
                if yaw <= YAW_LEFT_THRESHOLD:
                    looking_direction = "LookLeft"
                elif yaw >= YAW_RIGHT_THRESHOLD:
                    looking_direction = "LookRight"
                elif pitch >= PITCH_DOWN_THRESHOLD:
                    looking_direction = "LookDown"
                elif pitch <= PITCH_UP_THRESHOLD:
                    looking_direction = "LookUp"
                else:
                    looking_direction = "EyeContact"

        self.no_face_tracker.update(not face_detected)
        self.eyes_closed_tracker.update(eyes_closed)
        self.look_away_tracker.update(looking_direction != "EyeContact" and face_detected)

        rule_label = "EyeContact"
        if not face_detected:
            rule_label = "NoFace"
        elif eyes_closed:
            rule_label = "EyesClosed"
        else:
            rule_label = looking_direction

        model_label = None
        model_conf = 0.0
        probs = None

        if self.model_available and face_crop is not None and face_crop.size > 0:
            model_input = preprocess_for_model(face_crop)
            raw = self.model.predict(model_input, verbose=0)[0]
            self.model_smoother.push(raw)
            probs = self.model_smoother.get()
            if probs is not None:
                idx = int(np.argmax(probs))
                model_label = MODEL_CLASSES[idx]
                model_conf = float(probs[idx])
        else:
            self.model_smoother.clear()

        final_label = rule_label
        if (
            self.model_available
            and model_label
            and model_conf >= MODEL_CONF_THRESHOLD
            and model_label != "NoFace"
            and face_detected
        ):
            # Use model only as a supporting opinion.
            if final_label == "EyeContact" and model_label != "EyeContact":
                final_label = model_label
            elif final_label == model_label:
                final_label = model_label

        risk_level = "normal"
        alert_reason = None
        alert_duration = 0.0

        if self.no_face_tracker.elapsed >= NO_FACE_ALERT_SECONDS:
            risk_level = "alert"
            alert_reason = "NoFace"
            alert_duration = self.no_face_tracker.elapsed
        elif self.eyes_closed_tracker.elapsed >= EYES_CLOSED_ALERT_SECONDS:
            risk_level = "alert"
            alert_reason = "EyesClosed"
            alert_duration = self.eyes_closed_tracker.elapsed
        elif self.look_away_tracker.elapsed >= LOOK_AWAY_ALERT_SECONDS:
            risk_level = "warning"
            alert_reason = looking_direction
            alert_duration = self.look_away_tracker.elapsed

        if alert_reason:
            self._log_event(alert_reason, alert_duration)

        result = {
            "label": final_label,
            "rule_label": rule_label,
            "model_label": model_label,
            "model_confidence": round(model_conf, 4),
            "face_detected": face_detected,
            "yaw": None if yaw is None else round(yaw, 2),
            "pitch": None if pitch is None else round(pitch, 2),
            "roll": None if roll is None else round(roll, 2),
            "left_ear": None if left_ear is None else round(left_ear, 3),
            "right_ear": None if right_ear is None else round(right_ear, 3),
            "eyes_closed": eyes_closed,
            "risk_level": risk_level,
            "alert_reason": alert_reason,
            "alert_duration_sec": round(alert_duration, 2),
            "probabilities": None if probs is None else {
                cls: round(float(prob), 4) for cls, prob in zip(MODEL_CLASSES, probs)
            },
            "events_count": len(self.events),
        }

        return result, bbox

    def draw_overlay(self, frame_bgr: np.ndarray, result: dict, bbox) -> np.ndarray:
        frame = frame_bgr.copy()
        h, w, _ = frame.shape

        if bbox is not None and result["face_detected"]:
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, x1 + int(bbox.width * w))
            y2 = min(h, y1 + int(bbox.height * h))
            color = COLORS.get(result["label"], COLORS["info"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{result["label"]}'
            if result["model_confidence"] > 0:
                label += f' {result["model_confidence"]*100:.0f}%'
            cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + 220, y1), color, -1)
            cv2.putText(frame, label, (x1 + 8, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (410, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        cv2.putText(frame, f'Label: {result["label"]}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["info"], 2)
        cv2.putText(frame, f'Rule: {result["rule_label"]}', (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["info"], 1)

        model_line = "Model: OFF"
        if self.model_available:
            if result["model_label"]:
                model_line = f'Model: {result["model_label"]} ({result["model_confidence"]*100:.0f}%)'
            else:
                model_line = "Model: READY"
        cv2.putText(frame, model_line, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["info"], 1)

        pose_line = f'Yaw: {result["yaw"]}  Pitch: {result["pitch"]}'
        cv2.putText(frame, pose_line, (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["info"], 1)

        ear_line = f'EAR L: {result["left_ear"]}  R: {result["right_ear"]}'
        cv2.putText(frame, ear_line, (12, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["info"], 1)

        risk_color = COLORS["normal"]
        if result["risk_level"] == "warning":
            risk_color = COLORS["warning"]
        elif result["risk_level"] == "alert":
            risk_color = COLORS["alert"]

        risk_text = f'Risk: {result["risk_level"].upper()}'
        if result["alert_reason"]:
            risk_text += f' - {result["alert_reason"]} ({result["alert_duration_sec"]:.1f}s)'
        cv2.putText(frame, risk_text, (12, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)

        if result["probabilities"]:
            start_x = 12
            start_y = 210
            bar_w = 180
            bar_h = 16
            gap = 24
            for idx, (name, prob) in enumerate(result["probabilities"].items()):
                y = start_y + idx * gap
                cv2.rectangle(frame, (start_x, y), (start_x + bar_w, y + bar_h), (60, 60, 60), -1)
                cv2.rectangle(frame, (start_x, y), (start_x + int(bar_w * prob), y + bar_h), COLORS.get(name, COLORS["info"]), -1)
                cv2.putText(frame, f"{name}: {prob*100:.0f}%", (start_x + bar_w + 10, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(name, COLORS["info"]), 1)

        if result["risk_level"] in {"warning", "alert"}:
            banner = frame.copy()
            cv2.rectangle(banner, (0, h // 2 - 40), (w, h // 2 + 40), (0, 0, 180), -1)
            cv2.addWeighted(banner, 0.25, frame, 0.75, 0, frame)
            msg = f'ATTENTION RISK: {result["alert_reason"] or result["label"]}'
            text_w = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0][0]
            cv2.putText(frame, msg, ((w - text_w) // 2, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        cv2.putText(frame, "Q quit | S save screenshot", (w - 300, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["info"], 1)
        return frame


def main():
    monitor = BehaviorMonitor(use_model=USE_MODEL)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps = 0.0
    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result, bbox = monitor.analyze(frame)
            output = monitor.draw_overlay(frame, result, bbox)

            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / max(time.time() - fps_start, 1e-6)
                fps_start = time.time()

            cv2.putText(output, f"FPS: {fps:.1f}", (output.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                filename = f"behavior_capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, output)
                print(f"Saved: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        monitor.close()

        print("\nSession events:")
        for idx, event in enumerate(monitor.events[-20:], start=1):
            print(f'{idx}. [{event["time"]}] {event["label"]} - {event["duration_sec"]}s')


if __name__ == "__main__":
    main()
