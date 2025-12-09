import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime


SEQUENCE_LENGTH = 30
TARGET_PER_EMOTION = 150

EMOTIONS = ["happy", "sad", "angry", "afraid", "neutral"]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}

SAVE_DIR = "emotion_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)



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

    # Ratios (strong indicators for emotion change)
    feats["mouth_eye_ratio"] = feats["mouth_open"] / (
        feats["eye_left_open"] + feats["eye_right_open"] + 1e-6
    )
    feats["brow_eye_ratio"] = (
        feats["brow_left_height"] + feats["brow_right_height"]
    ) / (feats["eye_left_open"] + feats["eye_right_open"] + 1e-6)

    return np.array(list(feats.values()), dtype=np.float32)


mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

IMPORTANT_LANDMARKS = [
    33, 133, 159, 145, 153, 154,     # Left eye
    362, 263, 386, 374, 380, 387,    # Right eye
    70, 63, 105, 300, 293, 334,      # Brows
    13, 14, 61, 291, 78, 308,        # Mouth
    1, 2, 98, 327,                   # Nose
    234, 454,                        # Cheeks
    152, 377, 400                    # Jaw
]


def draw_landmarks_visual(frame, face_landmarks, w, h):

  
    mp_draw.draw_landmarks(
        frame,
        face_landmarks,
        mp.solutions.face_mesh.FACEMESH_TESSELATION,  
        None,
        mp_styles.get_default_face_mesh_tesselation_style()
    )

    mp_draw.draw_landmarks(
        frame,
        face_landmarks,
        mp.solutions.face_mesh.FACEMESH_CONTOURS,    
        None,
        mp_styles.get_default_face_mesh_contours_style()
    )

   
    lm = face_landmarks.landmark
    for idx in IMPORTANT_LANDMARKS:
        x = int(lm[idx].x * w)
        y = int(lm[idx].y * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


def get_next_path(emotion):
    idx = 1
    while True:
        path = f"{SAVE_DIR}/{emotion}_seq_{idx:03}.npz"
        if not os.path.exists(path):
            return path
        idx += 1



def load_counts():
    counts = {e: 0 for e in EMOTIONS}
    for file in os.listdir(SAVE_DIR):
        for emo in EMOTIONS:
            if file.startswith(emo) and file.endswith(".npz"):
                counts[emo] += 1
    return counts



def draw_ui(frame, counts, current_emotion, capturing):
    y = 30
    for emo in EMOTIONS:
        text = f"{emo.upper():7}  {counts[emo]:02}/{TARGET_PER_EMOTION}"

        if emo == current_emotion and capturing:
            color = (0, 255, 255)
        elif counts[emo] >= TARGET_PER_EMOTION:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30

    if capturing:
        cv2.putText(frame,
                    f"CAPTURING {current_emotion.upper()}...",
                    (20, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)



def main():
    mp_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    counts = load_counts()

    capturing = False
    current_emotion = None
    buffer = []

    key_map = {
        ord("1"): "happy",
        ord("2"): "sad",
        ord("3"): "angry",
        ord("4"): "afraid",
        ord("5"): "neutral",
    }

    print("\n🔥 Collector Ready — Press 1–5 to auto-collect.\n")

    while True:
        ret, frame = cap.read()
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_mesh.process(rgb)

        draw_ui(frame, counts, current_emotion, capturing)

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0]
            draw_landmarks_visual(frame, face, w, h)

            if capturing:
                feats = compute_features(face.landmark)
                buffer.append(feats)

                # Progress bar
                cv2.rectangle(frame, (20, 260),
                              (20 + int((len(buffer)/SEQUENCE_LENGTH)*300), 290),
                              (0, 255, 255), -1)

                if len(buffer) >= SEQUENCE_LENGTH:
                    arr = np.stack(buffer)
                    label = EMOTION_TO_ID[current_emotion]
                    save_path = get_next_path(current_emotion)

                    np.savez(save_path, features=arr, label=label)
                    counts[current_emotion] += 1

                    print(f"✔ Saved {save_path}")

                    buffer = []

                    if counts[current_emotion] >= TARGET_PER_EMOTION:
                        print(f"🎉 Completed {current_emotion.upper()}")
                        capturing = False
                        current_emotion = None
                    else:
                        capturing = True

        # Key input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in key_map and not capturing:
            emo = key_map[key]
            if counts[emo] >= TARGET_PER_EMOTION:
                print(f"✔ {emo.upper()} already collected!")
                continue

            print(f"\n▶ STARTING {emo.upper()}...")
            current_emotion = emo
            buffer = []
            capturing = True

        cv2.imshow("Emotion Collector", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
