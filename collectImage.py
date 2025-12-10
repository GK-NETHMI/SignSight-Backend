import cv2
import os
import time
import mediapipe as mp


DATASET_DIR = "dataset"
EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Fear']
TARGET_COUNT = 500
FACE_SIZE = 224  

KEY_MAP = {
    ord('1'): 'Angry',
    ord('2'): 'Happy',
    ord('3'): 'Neutral',
    ord('4'): 'Sad',
    ord('5'): 'Fear'
}


for emo in EMOTIONS:
    os.makedirs(os.path.join(DATASET_DIR, emo), exist_ok=True)


def get_counts():
    return {emo: len(os.listdir(os.path.join(DATASET_DIR, emo))) for emo in EMOTIONS}



mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)


cap = cv2.VideoCapture(0)

print("\n🔥 Mediapipe Face Collector Started")
print("Press keys:")
print("1 = Angry, 2 = Happy, 3 = Neutral, 4 = Sad, 5 = Fear")
print("Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    counts = get_counts()

    # Convert to RGB for Mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    face_crop = None

 
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)

        # clamp
        x = max(0, x)
        y = max(0, y)

        # Crop face ONLY — nothing else
        face_crop = frame[y:y + h_box, x:x + w_box]

        # Resize for consistency
        if face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))

    y_offset = 30
    for emo in EMOTIONS:
        cv2.putText(frame, f"{emo}: {counts[emo]}/{TARGET_COUNT}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow("Mediapipe Face Collector", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

   
    if key in KEY_MAP:
        emo = KEY_MAP[key]

        if counts[emo] >= TARGET_COUNT:
            print(f"⚠ {emo} full.")
            continue

        if face_crop is None or face_crop.size == 0:
            print("⚠ No face detected — skip.")
            continue

        save_path = os.path.join(DATASET_DIR, emo, f"{time.time()}.jpg")
        cv2.imwrite(save_path, face_crop)
        print(f"📸 Saved {emo} ({counts[emo] + 1}/{TARGET_COUNT})")

cap.release()
cv2.destroyAllWindows()
