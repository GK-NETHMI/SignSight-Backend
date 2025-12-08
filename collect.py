import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class AdvancedEmotionDataCollector:
    EMOTIONS = ["happy", "sad", "angry", "afraid", "neutral"]

    TARGET_SAMPLES_PER_EMOTION = 50  
    MIN_SAMPLES_PER_EMOTION = 50      
    SEQUENCE_LENGTH = 30              

    def __init__(self, output_dir: str = "emotion_dataset"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            static_image_mode=False,
        )

      
        self.sequences: List[Dict] = []  
        self.sample_count: Dict[str, int] = {e: 0 for e in self.EMOTIONS}
        self.quality_scores: Dict[str, List[float]] = {e: [] for e in self.EMOTIONS}

        self.current_emotion: str = "neutral"
        self.capturing_sequence: bool = False
        self.sequence_buffer: List[Dict] = []
        self.sequence_start_time: float = 0.0
        self.auto_collect_emotion: Optional[str] = None  

       
        self.feature_variance = {e: defaultdict(list) for e in self.EMOTIONS}
        self.inter_class_distances: Dict[str, float] = {}
        self.status_per_emotion: Dict[str, Dict] = {
            e: {
                "count": 0,
                "target": self.TARGET_SAMPLES_PER_EMOTION,
                "min": self.MIN_SAMPLES_PER_EMOTION,
                "avg_quality": 0.0,
                "need_more": True,
                "status_text": "NO DATA",
            }
            for e in self.EMOTIONS
        }
        self.dataset_ready: bool = False

        
        self.load_existing_data()
        self.assess_data_quality()

        print("✓ Advanced Emotion Data Collector initialized")
        print("✓ Using ALL 468 FaceMesh landmarks")
        print(f"✓ Sequence length: {self.SEQUENCE_LENGTH} frames")
        print(f"✓ Existing total sequences: {sum(self.sample_count.values())}")

   

    def data_file_path(self) -> str:
        return os.path.join(self.output_dir, "emotion_sequences.json")

    def load_existing_data(self):
        path = self.data_file_path()
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.sequences = data.get("sequences", [])

            for seq in self.sequences:
                label = seq.get("label")
                quality = seq.get("quality", 0.0)
                if label in self.sample_count:
                    self.sample_count[label] += 1
                    self.quality_scores[label].append(float(quality))

            print(f"✓ Loaded {len(self.sequences)} existing sequences")

        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")

    def save_all_data(self):
        meta = {
            "created_at": datetime.now().isoformat(),
            "emotions": self.EMOTIONS,
            "sequence_length": self.SEQUENCE_LENGTH,
            "target_per_emotion": self.TARGET_SAMPLES_PER_EMOTION,
            "min_per_emotion": self.MIN_SAMPLES_PER_EMOTION,
            "sample_count": self.sample_count,
        }
        data = {"meta": meta, "sequences": self.sequences}

        try:
            with open(self.data_file_path(), "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")


    def extract_all_facial_landmarks(self, face_landmarks, img_w: int, img_h: int) -> np.ndarray:
        coords = []
        for lm in face_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    def estimate_head_pose(self, face_landmarks, img_w: int, img_h: int) -> Tuple[float, float, float]:
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),          # nose tip
                (0.0, -330.0, -65.0),     # chin
                (-225.0, 170.0, -135.0),  # left eye corner
                (225.0, 170.0, -135.0),   # right eye corner
                (-150.0, -150.0, -125.0), # left mouth corner
                (150.0, -150.0, -125.0),  # right mouth corner
            ],
            dtype=np.float64,
        )

        lm = face_landmarks.landmark
        image_points = np.array(
            [
                (lm[1].x * img_w, lm[1].y * img_h),    # nose tip
                (lm[152].x * img_w, lm[152].y * img_h),# chin
                (lm[33].x * img_w, lm[33].y * img_h),  # left eye corner
                (lm[263].x * img_w, lm[263].y * img_h),# right eye corner
                (lm[61].x * img_w, lm[61].y * img_h),  # left mouth corner
                (lm[291].x * img_w, lm[291].y * img_h),# right mouth corner
            ],
            dtype=np.float64,
        )

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros((4, 1))
        success, rotation_vec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

        if sy > 1e-6:
            pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = 0.0

        return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))

    def compute_comprehensive_features(self, face_landmarks, img_w: int, img_h: int) -> Dict:
        lm = face_landmarks.landmark

        def get_point(idx):
            return np.array([lm[idx].x, lm[idx].y, lm[idx].z], dtype=np.float32)

        def distance(p1, p2):
            return float(np.linalg.norm(p1 - p2))

        def angle_between(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.dot(v1, v2) / denom
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            return float(np.arccos(cos_angle))

        features: Dict[str, float] = {}

        # ===== Eyes =====
        left_eye_center = (get_point(33) + get_point(133)) / 2
        right_eye_center = (get_point(362) + get_point(263)) / 2

        left_ear = distance(get_point(160), get_point(144)) / (distance(get_point(33), get_point(133)) + 1e-6)
        right_ear = distance(get_point(385), get_point(380)) / (distance(get_point(362), get_point(263)) + 1e-6)

        features["left_eye_aspect_ratio"] = left_ear
        features["right_eye_aspect_ratio"] = right_ear
        features["avg_eye_aspect_ratio"] = (left_ear + right_ear) / 2
        features["eye_aspect_ratio_diff"] = abs(left_ear - right_ear)

        features["left_eye_width"] = distance(get_point(33), get_point(133))
        features["right_eye_width"] = distance(get_point(362), get_point(263))
        features["left_eye_height"] = distance(get_point(160), get_point(144))
        features["right_eye_height"] = distance(get_point(385), get_point(380))

        features["inter_eye_distance"] = distance(left_eye_center, right_eye_center)

        left_upper = [get_point(i) for i in [33, 160, 158, 133]]
        right_upper = [get_point(i) for i in [362, 385, 387, 263]]
        features["left_eye_curvature"] = float(np.std([p[1] for p in left_upper]))
        features["right_eye_curvature"] = float(np.std([p[1] for p in right_upper]))

        features["left_eye_squeeze"] = distance(left_eye_center, get_point(33))
        features["right_eye_squeeze"] = distance(right_eye_center, get_point(362))

        features["left_lower_lid"] = distance(get_point(145), left_eye_center)
        features["right_lower_lid"] = distance(get_point(374), right_eye_center)

        features["left_eye_corner_height_diff"] = abs(get_point(33)[1] - get_point(133)[1])
        features["right_eye_corner_height_diff"] = abs(get_point(362)[1] - get_point(263)[1])

        # ===== Brows =====
        left_brow_center = get_point(63)
        right_brow_center = get_point(293)
        forehead_center = get_point(10)

        features["left_brow_height"] = distance(left_brow_center, left_eye_center)
        features["right_brow_height"] = distance(right_brow_center, right_eye_center)
        features["avg_brow_height"] = (features["left_brow_height"] + features["right_brow_height"]) / 2
        features["brow_height_diff"] = abs(features["left_brow_height"] - features["right_brow_height"])

        features["left_brow_width"] = distance(get_point(70), get_point(107))
        features["right_brow_width"] = distance(get_point(300), get_point(336))
        features["brow_furrow"] = distance(get_point(70), get_point(300))

        left_brow_points = [get_point(i) for i in [70, 63, 105, 66, 107]]
        right_brow_points = [get_point(i) for i in [300, 293, 334, 296, 336]]
        features["left_brow_arch"] = float(np.std([p[1] for p in left_brow_points]))
        features["right_brow_arch"] = float(np.std([p[1] for p in right_brow_points]))

        features["left_brow_angle"] = angle_between(get_point(70), get_point(63), get_point(107))
        features["right_brow_angle"] = angle_between(get_point(300), get_point(293), get_point(336))

        features["left_brow_tilt"] = get_point(70)[1] - get_point(107)[1]
        features["right_brow_tilt"] = get_point(300)[1] - get_point(336)[1]

        features["left_brow_forehead_dist"] = distance(left_brow_center, forehead_center)
        features["right_brow_forehead_dist"] = distance(right_brow_center, forehead_center)

        features["left_inner_brow_height"] = get_point(70)[1]
        features["right_inner_brow_height"] = get_point(300)[1]
        features["inner_brow_height_avg"] = (
            features["left_inner_brow_height"] + features["right_inner_brow_height"]
        ) / 2

        features["left_outer_brow_height"] = get_point(107)[1]
        features["right_outer_brow_height"] = get_point(336)[1]

        features["left_brow_compression"] = get_point(70)[1] - get_point(33)[1]
        features["right_brow_compression"] = get_point(300)[1] - get_point(362)[1]

        # ===== Mouth =====
        mouth_left = get_point(61)
        mouth_right = get_point(291)
        mouth_top = get_point(13)
        mouth_bottom = get_point(14)
        upper_lip_top = get_point(0)
        lower_lip_bottom = get_point(17)
        nose_tip = get_point(1)

        features["mouth_width"] = distance(mouth_left, mouth_right)
        features["mouth_height"] = distance(mouth_top, mouth_bottom)
        features["mouth_aspect_ratio"] = features["mouth_height"] / (features["mouth_width"] + 1e-6)

        features["upper_lip_thickness"] = distance(get_point(13), get_point(12))
        features["lower_lip_thickness"] = distance(get_point(14), get_point(15))

        features["inner_mouth_height"] = distance(get_point(13), get_point(14))
        features["inner_mouth_width"] = distance(get_point(78), get_point(308))

        features["left_lip_corner_height"] = mouth_left[1] - nose_tip[1]
        features["right_lip_corner_height"] = mouth_right[1] - nose_tip[1]
        features["avg_lip_corner_height"] = (
            features["left_lip_corner_height"] + features["right_lip_corner_height"]
        ) / 2

        features["lip_stretch"] = features["mouth_width"] / (features["inter_eye_distance"] + 1e-6)
        features["lip_compression"] = distance(upper_lip_top, lower_lip_bottom)

        mouth_center = (mouth_left + mouth_right) / 2
        face_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
        features["mouth_center_vertical_pos"] = mouth_center[1] - face_center_y

        upper_lip_points = [get_point(i) for i in [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]]
        lower_lip_points = [get_point(i) for i in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]]
        features["upper_lip_curvature"] = float(np.std([p[1] for p in upper_lip_points]))
        features["lower_lip_curvature"] = float(np.std([p[1] for p in lower_lip_points]))

        features["upper_lip_protrusion"] = upper_lip_top[2]
        features["lower_lip_protrusion"] = lower_lip_bottom[2]

        features["left_corner_to_center"] = distance(mouth_left, mouth_center)
        features["right_corner_to_center"] = distance(mouth_right, mouth_center)

        features["philtrum_length"] = distance(nose_tip, upper_lip_top)
        features["mouth_horizontal_asymmetry"] = abs(
            features["left_lip_corner_height"] - features["right_lip_corner_height"]
        )

        features["teeth_visibility_score"] = features["inner_mouth_height"] * (
            1 - features["lip_compression"] / (features["mouth_height"] + 1e-6)
        )

        features["left_lip_angle"] = angle_between(nose_tip, mouth_left, get_point(146))
        features["right_lip_angle"] = angle_between(nose_tip, mouth_right, get_point(375))

        mouth_perimeter_approx = features["mouth_width"] + features["mouth_height"]
        mouth_area_approx = features["mouth_width"] * features["mouth_height"]
        features["mouth_roundness"] = mouth_area_approx / (mouth_perimeter_approx**2 + 1e-6)

        features["lip_balance"] = features["upper_lip_thickness"] / (features["lower_lip_thickness"] + 1e-6)

        # ===== Nose =====
        nose_bridge_top = get_point(6)
        nose_left = get_point(98)
        nose_right = get_point(327)
        nose_bottom = get_point(2)

        features["nose_width"] = distance(nose_left, nose_right)
        features["nose_length"] = distance(nose_bridge_top, nose_tip)
        features["nose_height"] = distance(nose_tip, nose_bottom)

        nose_bridge_points = [get_point(i) for i in [6, 168, 197, 195, 5, 4, 1]]
        features["nose_bridge_curvature"] = float(np.std([p[1] for p in nose_bridge_points]))

        features["nostril_width"] = distance(nose_left, nose_right)
        features["nostril_flare_ratio"] = features["nostril_width"] / (features["inter_eye_distance"] + 1e-6)

        features["nose_wrinkle_x_var"] = float(np.var([p[0] for p in nose_bridge_points]))
        features["nose_wrinkle_y_var"] = float(np.var([p[1] for p in nose_bridge_points]))

        features["nose_tip_height"] = nose_tip[1]
        features["nose_tip_protrusion"] = nose_tip[2]
        features["nose_angle"] = angle_between(nose_bridge_top, nose_tip, nose_bottom)

        features["nose_to_mouth_dist"] = distance(nose_tip, mouth_top)

        nose_center_x = nose_tip[0]
        features["nose_asymmetry"] = abs(
            distance(nose_tip, nose_left) - distance(nose_tip, nose_right)
        ) + abs(nose_center_x - (nose_left[0] + nose_right[0]) / 2)

        # ===== Cheeks =====
        left_cheek = get_point(234)
        right_cheek = get_point(454)

        features["left_cheek_height"] = left_cheek[1]
        features["right_cheek_height"] = right_cheek[1]
        features["avg_cheek_height"] = (features["left_cheek_height"] + features["right_cheek_height"]) / 2

        features["left_cheek_raise"] = nose_tip[1] - left_cheek[1]
        features["right_cheek_raise"] = nose_tip[1] - right_cheek[1]
        features["avg_cheek_raise"] = (features["left_cheek_raise"] + features["right_cheek_raise"]) / 2

        features["face_width"] = distance(left_cheek, right_cheek)

        features["left_cheek_to_eye"] = distance(left_cheek, left_eye_center)
        features["right_cheek_to_eye"] = distance(right_cheek, right_eye_center)

        features["cheek_asymmetry"] = abs(features["left_cheek_raise"] - features["right_cheek_raise"])

        # ===== Jaw & chin =====
        chin = get_point(152)
        jaw_left = get_point(234)
        jaw_right = get_point(454)

        features["jaw_width"] = distance(jaw_left, jaw_right)
        features["jaw_drop"] = distance(nose_tip, chin)

        features["chin_height"] = chin[1]
        features["chin_protrusion"] = chin[2]

        features["left_jaw_angle"] = angle_between(left_eye_center, jaw_left, chin)
        features["right_jaw_angle"] = angle_between(right_eye_center, jaw_right, chin)

        features["face_length"] = distance(forehead_center, chin)
        features["face_aspect_ratio"] = features["face_length"] / (features["face_width"] + 1e-6)

        jawline_points = [
            get_point(i)
            for i in [
                234,
                227,
                137,
                177,
                215,
                138,
                32,
                208,
                169,
                152,
                394,
                395,
                369,
                396,
                175,
                152,
                148,
                176,
                149,
                150,
                454,
            ]
        ]
        features["jawline_tension"] = float(np.std([distance(p, chin) for p in jawline_points]))

        features["lower_face_height"] = distance(nose_tip, chin)
        features["chin_to_lip_dist"] = distance(chin, lower_lip_bottom)
        features["jaw_symmetry"] = abs(distance(jaw_left, chin) - distance(jaw_right, chin))

        # ===== Forehead =====
        forehead_points = [get_point(i) for i in [10, 151, 9, 8, 168, 6, 197, 195]]

        features["forehead_height"] = distance(forehead_center, nose_bridge_top)
        features["forehead_wrinkle_horizontal"] = float(np.var([p[1] for p in forehead_points]))
        features["forehead_wrinkle_vertical"] = float(np.var([p[0] for p in forehead_points]))
        features["forehead_width"] = distance(get_point(21), get_point(251))
        features["forehead_curvature"] = float(np.std([p[2] for p in forehead_points]))
        features["forehead_to_brow_dist"] = distance(forehead_center, left_brow_center)
        features["upper_face_height"] = distance(forehead_center, nose_tip)
        features["upper_third_ratio"] = features["forehead_height"] / (features["face_length"] + 1e-6)

        # ===== Head pose =====
        yaw, pitch, roll = self.estimate_head_pose(face_landmarks, img_w, img_h)
        features["head_yaw"] = yaw
        features["head_pitch"] = pitch
        features["head_roll"] = roll

        # ===== Symmetry =====
        features["overall_symmetry_score"] = (
            features["eye_aspect_ratio_diff"]
            + features["brow_height_diff"]
            + features["mouth_horizontal_asymmetry"]
            + features["cheek_asymmetry"]
            + features["jaw_symmetry"]
        ) / 5

        features["vertical_face_symmetry"] = abs(left_eye_center[0] - right_eye_center[0])
        features["horizontal_face_alignment"] = abs(left_eye_center[1] - right_eye_center[1])

        features["feature_count"] = float(len(features))

        return features

    

    def compute_sequence_quality(self, sequence: List[Dict]) -> float:
        if len(sequence) < self.SEQUENCE_LENGTH * 0.8:
            return 0.0

        quality_factors: List[float] = []

       
        completeness = len(sequence) / self.SEQUENCE_LENGTH
        quality_factors.append(completeness)

       
        variances = []
        feature_names = list(sequence[0]["derived_features"].keys())
        for fname in feature_names[:20]:
            vals = [frame["derived_features"].get(fname, 0.0) for frame in sequence]
            if vals:
                variances.append(float(np.std(vals)))

        if variances:
            stability = 1.0 / (1.0 + float(np.mean(variances)))
        else:
            stability = 0.5
        quality_factors.append(stability * 0.5)

       
        deviations = []
        for frame in sequence:
            f = frame["derived_features"]
            d = 0.0
            d += abs(f.get("avg_eye_aspect_ratio", 0.3) - 0.3)
            d += abs(f.get("mouth_aspect_ratio", 0.3) - 0.3)
            d += abs(f.get("avg_brow_height", 0.05) - 0.05)
            deviations.append(d)

        if deviations:
            intensity = min(1.0, float(np.mean(deviations)) * 5.0)
        else:
            intensity = 0.3
        quality_factors.append(intensity)

       
        quality_factors.append(1.0)

        return float(np.mean(quality_factors))

    def assess_data_quality(self):
        total = sum(self.sample_count.values())
        print("\n" + "=" * 70)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 70)
        print(f"Total sequences: {total}")

        self.dataset_ready = True

        for emotion in self.EMOTIONS:
            count = self.sample_count[emotion]
            target = self.TARGET_SAMPLES_PER_EMOTION
            min_req = self.MIN_SAMPLES_PER_EMOTION
            avg_q = (
                float(np.mean(self.quality_scores[emotion]))
                if self.quality_scores[emotion]
                else 0.0
            )

            if count == 0:
                status_text = "NO DATA"
                need_more = True
            elif count >= target and avg_q >= 0.6:
                status_text = "✓ EXCELLENT"
                need_more = False
            elif count >= min_req and avg_q >= 0.5:
                status_text = "OK"
                need_more = True  
            elif count < min_req:
                status_text = "LOW"
                need_more = True
            else:
                status_text = "WEAK"
                need_more = True

            self.status_per_emotion[emotion] = {
                "count": count,
                "target": target,
                "min": min_req,
                "avg_quality": avg_q,
                "need_more": need_more,
                "status_text": status_text,
            }

            if need_more:
                self.dataset_ready = False

            print(
                f"- {emotion.upper():7s} | "
                f"count={count:3d}/{target} | "
                f"avg_q={avg_q:0.3f} | "
                f"{status_text}"
            )

        if self.dataset_ready:
            print("\n✓ DATASET LOOKS READY (all emotions above thresholds)")
        else:
            print("\n⚠ Dataset still needs more sequences for some emotions.")

    def get_emotion_status(self, emotion: str) -> Dict:
        return self.status_per_emotion.get(
            emotion,
            {
                "count": 0,
                "target": self.TARGET_SAMPLES_PER_EMOTION,
                "min": self.MIN_SAMPLES_PER_EMOTION,
                "avg_quality": 0.0,
                "need_more": True,
                "status_text": "NO DATA",
            },
        )

   
    def start_sequence(self, emotion: str):
        self.current_emotion = emotion
        self.sequence_buffer = []
        self.sequence_start_time = time.time()
        self.capturing_sequence = True
        print(f"\n▶ Start sequence for {emotion.upper()}")

    def finish_sequence(self) -> Dict:
        if not self.sequence_buffer:
            self.capturing_sequence = False
            return {"saved": False}

        quality = self.compute_sequence_quality(self.sequence_buffer)

        sequence_dict = {
            "label": self.current_emotion,
            "timestamp": datetime.now().isoformat(),
            "quality": float(quality),
            "length": len(self.sequence_buffer),
            "frames": self.sequence_buffer,
        }

        self.sequences.append(sequence_dict)
        self.sample_count[self.current_emotion] += 1
        self.quality_scores[self.current_emotion].append(float(quality))

        print(
            f"■ Saved sequence #{self.sample_count[self.current_emotion]} "
            f"for {self.current_emotion.upper()} | quality={quality:0.3f} "
            f"| length={len(self.sequence_buffer)}"
        )

        self.save_all_data()
        self.capturing_sequence = False
        self.sequence_buffer = []

        self.assess_data_quality()
        status = self.get_emotion_status(self.current_emotion)

        return {
            "saved": True,
            "quality": quality,
            "status": status,
            "count": self.sample_count[self.current_emotion],
        }





def draw_overlay(
    frame,
    collector: AdvancedEmotionDataCollector,
    fps: float,
):
    h, w, _ = frame.shape
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

   
    x = 10
    y = 25
    for idx, emotion in enumerate(collector.EMOTIONS, start=1):
        status = collector.get_emotion_status(emotion)
        count = status["count"]
        target = status["target"]
        text = f"[{idx}] {emotion.upper()} {count}/{target}"

        if emotion == collector.current_emotion and collector.capturing_sequence:
            color = (0, 255, 255)  
        elif not status["need_more"]:
            color = (0, 255, 0) 
        else:
            color = (255, 255, 255)  

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 20

    
    y += 5
    cv2.putText(
        frame,
        f"Current: {collector.current_emotion.upper()}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255) if collector.capturing_sequence else (255, 255, 255),
        2 if collector.capturing_sequence else 1,
        cv2.LINE_AA,
    )
    y += 25

    if collector.capturing_sequence:
        progress = len(collector.sequence_buffer) / float(collector.SEQUENCE_LENGTH)
        progress = max(0.0, min(1.0, progress))
        bar_x1, bar_x2 = 10, w - 10
        bar_y1, bar_y2 = y, y + 15
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), 1)
        filled_x2 = int(bar_x1 + progress * (bar_x2 - bar_x1))
        cv2.rectangle(frame, (bar_x1, bar_y1), (filled_x2, bar_y2), (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"Sequence progress: {len(collector.sequence_buffer)}/{collector.SEQUENCE_LENGTH}",
            (10, bar_y2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y = bar_y2 + 35
    else:
        cv2.putText(
            frame,
            "Press 1-5 to start capturing for that emotion.",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20

    
    status_text = (
        "DATASET READY" if collector.dataset_ready else "Dataset still needs more sequences"
    )
    status_color = (0, 255, 0) if collector.dataset_ready else (0, 165, 255)
    cv2.putText(
        frame,
        status_text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        1,
        cv2.LINE_AA,
    )
    y += 20

  
    cv2.putText(
        frame,
        "Controls: [1-5]=emotion  [C]=cancel current  [Q]=quit",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:0.1f}",
        (w - 100, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    collector = AdvancedEmotionDataCollector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    mp_drawing = collector.mp_drawing
    mp_styles = collector.mp_drawing_styles
    mp_face_mesh = collector.mp_face_mesh

    key_to_emotion = {
        ord("1"): "happy",
        ord("2"): "sad",
        ord("3"): "angry",
        ord("4"): "afraid",
        ord("5"): "neutral",
    }

    prev_time = time.time()
    fps = 0.0

    window_name = "Emotion Data Collection - Stage 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        img_h, img_w, _ = frame.shape

       
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        fps = 1.0 / dt if dt > 0 else fps

       
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = collector.face_mesh.process(rgb)

        has_face = results.multi_face_landmarks is not None
        face_landmarks = None
        if has_face:
            face_landmarks = results.multi_face_landmarks[0]
           
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
            )

       
        if collector.capturing_sequence and has_face and face_landmarks is not None:
            all_landmarks = collector.extract_all_facial_landmarks(face_landmarks, img_w, img_h)
            derived = collector.compute_comprehensive_features(face_landmarks, img_w, img_h)

            collector.sequence_buffer.append(
                {
                    "landmarks": all_landmarks.tolist(),
                    "derived_features": {k: float(v) for k, v in derived.items()},
                }
            )

           
            if len(collector.sequence_buffer) >= collector.SEQUENCE_LENGTH:
                result = collector.finish_sequence()
                if result.get("saved") and collector.auto_collect_emotion == collector.current_emotion:
                    status = result["status"]
                    if status.get("need_more", True):
                       
                        collector.start_sequence(collector.current_emotion)
                    else:
                     
                        collector.auto_collect_emotion = None

       
        draw_overlay(frame, collector, fps)

        if not has_face:
            cv2.putText(
                frame,
                "No face detected - align your face to start capturing.",
                (10, img_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

      
        if key in (ord("q"), ord("Q")):
            print("Exiting...")
            break

        if key in (ord("c"), ord("C")):
            if collector.capturing_sequence:
                collector.capturing_sequence = False
                collector.sequence_buffer = []
                collector.auto_collect_emotion = None
                print("■ Sequence cancelled.")
            continue

        
        if key in key_to_emotion and not collector.capturing_sequence:
            emotion = key_to_emotion[key]
            status = collector.get_emotion_status(emotion)

          
            collector.auto_collect_emotion = emotion
            collector.start_sequence(emotion)

            
            print(
                f"\n=== START COLLECTING '{emotion.upper()}' ===\n"
                f"Current count: {status['count']}/{status['target']}\n"
                "The system will automatically record multiple sequences for this emotion\n"
                "until it's satisfied (count + quality).\n"
                "It will NOT auto-start other emotions."
            )

    cap.release()
    cv2.destroyAllWindows()
    collector.save_all_data()
    print("Done. Data saved to:", collector.data_file_path())


if __name__ == "__main__":
    main()
