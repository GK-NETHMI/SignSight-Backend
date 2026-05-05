import cv2
import numpy as np

from live import BehaviorMonitor, MODEL_CLASSES


ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
FRAME_SKIP = 1
MIRROR_FRAMES_TO_MATCH_LIVE = True


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_video_enhanced(video_path):
    print("Analyzing video with live BehaviorMonitor:", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps <= 0:
        fps = 30

    monitor = BehaviorMonitor(use_model=True)
    results = {
        "total_frames": 0,
        "processed_frames": 0,
        "face_detected_frames": 0,
        "face_not_detected_frames": 0,
        "final_predictions": {cls: 0 for cls in MODEL_CLASSES},
        "rule_predictions": {cls: 0 for cls in MODEL_CLASSES},
        "model_predictions": {cls: 0 for cls in MODEL_CLASSES},
        "average_confidence": {cls: [] for cls in MODEL_CLASSES},
        "risk_levels": {"normal": 0, "warning": 0, "alert": 0},
    }

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
            if MIRROR_FRAMES_TO_MATCH_LIVE:
                frame = cv2.flip(frame, 1)

            result, _ = monitor.analyze(frame)

            final_label = result.get("label") or "NoFace"
            rule_label = result.get("rule_label") or "NoFace"
            model_label = result.get("model_label")
            model_conf = float(result.get("model_confidence") or 0.0)
            risk_level = result.get("risk_level") or "normal"

            if result.get("face_detected"):
                results["face_detected_frames"] += 1
            else:
                results["face_not_detected_frames"] += 1

            if final_label in results["final_predictions"]:
                results["final_predictions"][final_label] += 1
            if rule_label in results["rule_predictions"]:
                results["rule_predictions"][rule_label] += 1
            if model_label in results["model_predictions"]:
                results["model_predictions"][model_label] += 1
                results["average_confidence"][model_label].append(model_conf)
            if risk_level in results["risk_levels"]:
                results["risk_levels"][risk_level] += 1

    finally:
        cap.release()
        monitor.close()

    processed_total = results["processed_frames"] or 1

    avg_conf_summary = {
        cls: round(float(np.mean(confs)), 4) if confs else 0.0
        for cls, confs in results["average_confidence"].items()
    }
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
            "not_detected_percentage": f"{no_face_pct}%",
        },
        "dominant_prediction": dominant_label,
        "final_prediction_statistics": results["final_predictions"],
        "final_prediction_percentages": final_prediction_percentages,
        "rule_statistics": results["rule_predictions"],
        "model_statistics": results["model_predictions"],
        "average_model_confidence": avg_conf_summary,
        "risk_level_statistics": results["risk_levels"],
        "model_enabled": monitor.model_available,
        "notes": [
            "Video analysis uses the same BehaviorMonitor decision logic as live.py.",
            "Final predictions combine face rules, face mesh signals, and model confidence threshold.",
            "Percentages are calculated from processed video frames.",
        ],
    }


def video_analyze_enhanced(path):
    try:
        return analyze_video_enhanced(path)
    except Exception as e:
        return {"error": str(e)}
