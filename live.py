import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import json
import os


SEQUENCE_LENGTH = 30
EMOTIONS = ["happy", "sad", "angry", "afraid", "neutral"]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}



mean = np.load("models/scaler_mean.npy")
scale = np.load("models/scaler_scale.npy")

def scale_features(arr):
    return (arr - mean) / scale


def compute_features(lm):
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    feats = {}

    def dist(a, b):
        return float(np.linalg.norm(pts[a] - pts[b]))

    # Eyes
    feats["eye_left_open"]  = dist(159, 145)
    feats["eye_right_open"] = dist(386, 374)
    feats["eye_left_width"] = dist(33, 133)
    feats["eye_right_width"] = dist(362, 263)

    # Brows
    feats["brow_left_height"]  = dist(70, 33)
    feats["brow_right_height"] = dist(300, 362)

    # Mouth
    feats["mouth_open"]  = dist(13, 14)
    feats["mouth_width"] = dist(61, 291)

    # Nose
    feats["nose_length"] = dist(1, 2)
    feats["nostril_width"] = dist(98, 327)

    # Cheeks
    feats["cheek_left"]  = pts[234][1]
    feats["cheek_right"] = pts[454][1]

    # Jaw
    feats["jaw_drop"] = dist(152, 1)

    # Ratios
    feats["mouth_eye_ratio"] = feats["mouth_open"] / (
        feats["eye_left_open"] + feats["eye_right_open"] + 1e-6
    )
    feats["brow_eye_ratio"] = (
        feats["brow_left_height"] + feats["brow_right_height"]
    ) / (feats["eye_left_open"] + feats["eye_right_open"] + 1e-6)

    return np.array(list(feats.values()), dtype=np.float32)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        w = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(w * lstm_out, dim=1)
        return context


class EmotionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=len(EMOTIONS)):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )
        self.attn = Attention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        return self.fc(context)


def load_model():
    feature_dim = len(compute_features([type("dummy", (), {"x":0, "y":0, "z":0})()]*500))
    model = EmotionNet(feature_dim)
    model.load_state_dict(torch.load("models/emotion_model_npz.pt", map_location="cpu"))
    model.eval()
    return model



def main():

    model = load_model()
    mp_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    buffer = deque(maxlen=SEQUENCE_LENGTH)

    cap = cv2.VideoCapture(0)
    print("\n🔥 LIVE EMOTION DETECTOR READY\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_mesh.process(rgb)

        emotion_text = "Detecting..."

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # Extract features
            feats = compute_features(lm)
            buffer.append(feats)

            # Predict when buffer full
            if len(buffer) == SEQUENCE_LENGTH:
                seq = np.array(buffer)
                seq = scale_features(seq)
                seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    pred = model(seq)
                    idx = pred.argmax(dim=1).item()
                    emotion_text = EMOTIONS[idx]

        # UI
        cv2.putText(frame, f"Emotion: {emotion_text}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("LIVE Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
