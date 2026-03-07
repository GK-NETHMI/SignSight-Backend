import os
import cv2
import numpy as np
import mediapipe as mp

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MAX_HEAD_ANGLE = 15
MIN_EYE_OPENNESS = 0.2
FRAME_SKIP = 2

def analyze_video(video_path):
    print("Analyzing:", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    results = {
        'total_frames': 0,
        'eye_contact_frames': 0,
        'look_away_frames': 0,
        'face_not_detected_frames': 0
    }

    LEFT = [362, 385, 387, 263, 373, 380]
    RIGHT = [33, 160, 158, 133, 153, 144]

    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % FRAME_SKIP != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(rgb)

        if not results_face.multi_face_landmarks:
            results['face_not_detected_frames'] += 1
            results['total_frames'] += 1
            continue

        face_landmarks = results_face.multi_face_landmarks[0]
        landmarks = [(int(lm.x * size[0]), int(lm.y * size[1])) for lm in face_landmarks.landmark]

        left_eye = np.array([landmarks[i] for i in LEFT], dtype="double")
        right_eye = np.array([landmarks[i] for i in RIGHT], dtype="double")

        def EAR(eye):
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            return (A + B) / (2.0 * C)

        left_ear = EAR(left_eye)
        right_ear = EAR(right_eye)

        eyes_open = left_ear > MIN_EYE_OPENNESS and right_ear > MIN_EYE_OPENNESS

        if eyes_open:
            results['eye_contact_frames'] += 1
        else:
            results['look_away_frames'] += 1

        results['total_frames'] += 1

    cap.release()

    total = results['total_frames']
    eye_pct = round(results['eye_contact_frames'] / total * 100, 1) if total else 0

    return {
        "video_duration": round(total / fps, 1),
        "eye_contact": {
            "duration": int(results['eye_contact_frames'] / fps),
            "percentage": f"{eye_pct}%"
        },
        "look_away": {
            "duration": int(results['look_away_frames'] / fps),
            "percentage": f"{round(results['look_away_frames'] / total * 100, 1)}%" if total else "0%"
        },
        "face_not_detected": {
            "duration": int(results['face_not_detected_frames'] / fps),
            "percentage": "0%"
        }
    }


def video_analyze(path):
    try:
        return analyze_video(path)
    except Exception as e:
        return {"error": str(e)}
