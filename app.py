import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
from collections import deque
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = "emotion_model_mobilenet.h5"
IMG_SIZE = 224
EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Fear']
PREDICTION_BUFFER_SIZE = 7
CONFIDENCE_THRESHOLD = 0.8

EMOTION_COLORS = {
    'Angry': (0, 0, 255),
    'Happy': (0, 255, 0),
    'Neutral': (255, 255, 0),
    'Sad': (255, 0, 0),
    'Fear': (0, 165, 255)
}

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

print("Loading model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype='float32')
    _ = model.predict(dummy_input, verbose=0)
except Exception as e:
    print(f"Error loading model: {e}")
    import os
    for file in os.listdir('.'):
        if file.endswith('.h5'):
            print(f"  - {file}")
    exit(1)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

class PredictionSmoother:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.no_face_counter = 0
    
    def add_prediction(self, probs):
        self.buffer.append(probs)
        self.no_face_counter = 0
    
    def get_smoothed_prediction(self):
        if len(self.buffer) == 0:
            return None
        weights = np.linspace(0.5, 1.0, len(self.buffer))
        weights = weights / weights.sum()
        avg_probs = np.average(self.buffer, axis=0, weights=weights)
        return avg_probs
    
    def clear(self):
        self.no_face_counter += 1
        if self.no_face_counter > 5:
            self.buffer.clear()

smoother = PredictionSmoother(PREDICTION_BUFFER_SIZE)

def preprocess_face(face_img):
    try:
        face = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        return face
    except Exception as e:
        return None

def draw_emotion_bar(frame, probs, x_start=10, y_start=100):
    bar_height = 25
    bar_width = 250
    spacing = 35
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start-5, y_start-10), 
                 (x_start + bar_width + 120, y_start + len(EMOTIONS) * spacing + 10),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    for idx, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        y = y_start + idx * spacing
        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height),
                     (60, 60, 60), -1)
        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height),
                     (100, 100, 100), 2)
        filled_width = int(bar_width * prob)
        if filled_width > 0:
            color = EMOTION_COLORS[emotion]
            cv2.rectangle(frame, (x_start, y), (x_start + filled_width, y + bar_height),
                         color, -1)
        text = f"{emotion}: {prob*100:.1f}%"
        color = EMOTION_COLORS[emotion]
        cv2.putText(frame, text, (x_start + bar_width + 12, y + 19),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, text, (x_start + bar_width + 10, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_face_box_with_emotion(frame, bbox, emotion, confidence, h, w):
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    w_box = int(bbox.width * w)
    h_box = int(bbox.height * h)
    x = max(0, x)
    y = max(0, y)
    x_end = min(w, x + w_box)
    y_end = min(h, y + h_box)
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    thickness = 3
    cv2.rectangle(frame, (x, y), (x_end, y_end), color, thickness)
    corner_length = 20
    cv2.line(frame, (x, y), (x + corner_length, y), color, thickness + 2)
    cv2.line(frame, (x, y), (x, y + corner_length), color, thickness + 2)
    cv2.line(frame, (x_end, y), (x_end - corner_length, y), color, thickness + 2)
    cv2.line(frame, (x_end, y), (x_end, y + corner_length), color, thickness + 2)
    cv2.line(frame, (x, y_end), (x + corner_length, y_end), color, thickness + 2)
    cv2.line(frame, (x, y_end), (x, y_end - corner_length), color, thickness + 2)
    cv2.line(frame, (x_end, y_end), (x_end - corner_length, y_end), color, thickness + 2)
    cv2.line(frame, (x_end, y_end), (x_end, y_end - corner_length), color, thickness + 2)
    label = f"{emotion} ({confidence*100:.0f}%)"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    padding = 8
    label_y = y - 15 if y > 40 else y_end + 15
    cv2.rectangle(frame,
                 (x, label_y - label_size[1] - padding),
                 (x + label_size[0] + padding * 2, label_y + padding),
                 color, -1)
    text_y = label_y - padding // 2
    cv2.putText(frame, label, (x + padding + 1, text_y + 1),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    cv2.putText(frame, label, (x + padding, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def draw_info_panel(frame, fps, face_detected, h, w):
    panel_height = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    status_color = (0, 255, 0) if face_detected else (0, 165, 255)
    status_text = "Face Detected" if face_detected else "No Face"
    cv2.putText(frame, status_text, (20, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'Q' to quit", (w // 2 - 100, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fps_time = time.time()
    fps = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break
        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)
        emotion_detected = None
        confidence = 0
        probs = None
        face_detected = False
        if results.detections:
            face_detected = True
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)
            x = max(0, x)
            y = max(0, y)
            x_end = min(w, x + w_box)
            y_end = min(h, y + h_box)
            face_crop = frame[y:y_end, x:x_end]
            if face_crop.size > 0:
                processed_face = preprocess_face(face_crop)
                if processed_face is not None:
                    prediction = model.predict(processed_face, verbose=0)[0]
                    smoother.add_prediction(prediction)
                    probs = smoother.get_smoothed_prediction()
                    if probs is not None:
                        emotion_idx = np.argmax(probs)
                        emotion_detected = EMOTIONS[emotion_idx]
                        confidence = probs[emotion_idx]
                        if confidence >= CONFIDENCE_THRESHOLD:
                            draw_face_box_with_emotion(frame, bbox, emotion_detected, 
                                                      confidence, h, w)
        else:
            smoother.clear()
        if probs is not None:
            draw_emotion_bar(frame, probs)
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
        draw_info_panel(frame, fps, face_detected, h, w)
        cv2.imshow("Real-time Emotion Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise