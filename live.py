import cv2
import numpy as np
import mediapipe as mp
import time
from datetime import datetime

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Define 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Camera internals (assuming webcam with 640x480 resolution)
size = (640, 480)
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

# Distortion coefficients
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

# Thresholds
MAX_HEAD_ANGLE = 15  # degrees
MAX_GAZE_DEVIATION = 0.35  # normalized units
MAX_LOOK_AWAY_TIME = 5.0  # seconds
MIN_EYE_OPENNESS = 0.2  # Eye Aspect Ratio (EAR)

class quizProctor:
    def __init__(self):
        self.look_away_start = None
        self.cheating_events = []
        self.last_eye_contact_time = time.time()
    
    def update(self, eye_contact):
        current_time = time.time()
        if not eye_contact:
            if self.look_away_start is None:
                self.look_away_start = current_time
            elif current_time - self.look_away_start > MAX_LOOK_AWAY_TIME:
                if not self.cheating_events or current_time - self.cheating_events[-1]["timestamp"] > 10:
                    self.cheating_events.append({
                        "timestamp": current_time,
                        "duration": current_time - self.look_away_start
                    })
        else:
            self.look_away_start = None
            self.last_eye_contact_time = current_time

def get_head_pose(landmarks, image_size):
    image_points = np.array([
        (landmarks[1][0], landmarks[1][1]),     # Nose tip
        (landmarks[152][0], landmarks[152][1]), # Chin
        (landmarks[263][0], landmarks[263][1]), # Left eye left corner
        (landmarks[33][0], landmarks[33][1]),   # Right eye right corner
        (landmarks[287][0], landmarks[287][1]), # Left Mouth corner
        (landmarks[57][0], landmarks[57][1])    # Right mouth corner
    ], dtype="double")

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles  # pitch, yaw, roll

def calculate_eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_landmarks(landmarks, indices):
    return np.array([(landmarks[i][0], landmarks[i][1]) for i in indices], dtype="double")

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    
    proctor = quizProctor()
    font = cv2.FONT_HERSHEY_SIMPLEX

    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * size[0]), int(lm.y * size[1])
                landmarks.append((x, y))

            # Head pose estimation
            pitch, yaw, roll = get_head_pose(landmarks, size)
            head_pose_ok = abs(yaw) < MAX_HEAD_ANGLE and abs(pitch) < MAX_HEAD_ANGLE

            # Eye aspect ratio
            left_eye = get_eye_landmarks(landmarks, LEFT_EYE_INDICES)
            right_eye = get_eye_landmarks(landmarks, RIGHT_EYE_INDICES)
            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            eyes_open = left_ear > MIN_EYE_OPENNESS and right_ear > MIN_EYE_OPENNESS

            # Eye contact determination
                # Check if user is looking forward
            looking_forward = abs(yaw) < 10  # yaw angle within ±10 degrees

            # Eye contact only if facing forward and eyes are open
            eye_contact = head_pose_ok and eyes_open and looking_forward

            # Update proctor status
            proctor.update(eye_contact)

            # Draw detection info
            color = (0, 255, 0) if eye_contact else (0, 0, 255)
            cv2.putText(frame, f"EYE CONTACT: {'YES' if eye_contact else 'NO'}", (20, 40), font, 1, color, 2)
            cv2.putText(frame, f"LOOKING FORWARD: {'YES' if looking_forward else 'NO'}", (20, 70), font, 1,
                        (0, 255, 0) if looking_forward else (0, 0, 255), 2)

            # Existing debug info
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 100), font, 0.7, (0, 0, 0), 1)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 130), font, 0.7, (0, 0, 0), 1)
            cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (20, 160), font, 0.7, (0, 0, 0), 1)
            cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (20, 190), font, 0.7, (0, 0, 0), 1)
            

            # Draw warning if cheating detected
            if proctor.cheating_events:
                last_event = proctor.cheating_events[-1]
                if time.time() - last_event["timestamp"] < 5:  # Show warning for 5 seconds
                    cv2.putText(frame, "WARNING: POTENTIAL CHEATING", (size[0]//4, size[1]//2), 
                               font, 1.5, (0, 0, 255), 3)
                    cv2.putText(frame, f"Looked away for {last_event['duration']:.1f}s", 
                               (size[0]//4, size[1]//2 + 50), font, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "FACE NOT DETECTED", (20, 40), font, 1, (0, 0, 255), 2) 

        # Show quiz timer
        quiz_time = time.time() - proctor.last_eye_contact_time
        cv2.putText(frame, f"quiz Time: {datetime.utcfromtimestamp(quiz_time).strftime('%M:%S')}", 
                   (size[0]-300, 40), font, 0.7, (255, 255, 255), 2)

        cv2.imshow('Quiz Proctoring System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print cheating events summary
    print("\nQuiz Proctoring Summary:")
    print(f"Total cheating events: {len(proctor.cheating_events)}")
    for i, event in enumerate(proctor.cheating_events, 1):
        print(f"{i}. At {datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S')} - {event['duration']:.1f}s")
if __name__ == "__main__":
    main()