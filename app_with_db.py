from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from werkzeug.utils import secure_filename
import requests
from dbConfig import save_attempt, init_db
from enhanced_eye_contact import allowed_file, video_analyze_enhanced

app = Flask(__name__)
CORS(app)

# Initialize DB
init_db()

# Folders
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB max

# EXTERNAL VIDEO API
EXTERNAL_VIDEO_API_URL = "http://localhost:5001/process-video"

# Categories
VARIATIONS = ["family", "alphabet", "numbers", "objects", "actions", "emotions"]

LEVEL_CATEGORIES = {
    "basic": ["category_1", "category_2"],
    "intermediate": ["category_2", "category_3"],
    "advanced": ["category_3", "category_4"]
}

# -------------------------
#       SHAP MODEL
# -------------------------
class SHAPModel:
    def __init__(self):
        feature_count = 4 + len(VARIATIONS)

        X = np.random.rand(1000, feature_count) * 100
        y = X.mean(axis=1) + np.random.randn(1000) * 5
        y = np.clip(y, 0, 100)

        self.model = RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            random_state=42
        )
        self.model.fit(X, y)

        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, features):
        arr = np.array(features).reshape(1, -1)

        shap_values = self.explainer.shap_values(arr, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return {
            "predicted_score": float(self.model.predict(arr)[0]),
            "base_value": float(self.explainer.expected_value.ravel()[0]),
            "shap_values": shap_values.flatten().tolist()
        }


shap_model = SHAPModel()


def _normalize_answer(value):
    if value is None:
        return ""
    normalized = str(value).strip().lower().strip("\"'")
    return " ".join(normalized.replace("_", " ").replace("-", " ").split())


def _first_value(data, keys):
    if not hasattr(data, "get"):
        return None

    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value

    return None


def _get_advanced_correct_answer(quiz_data, form_data=None):
    answer_keys = (
        "correct_answer",
        "correctAnswer",
        "expected_answer",
        "expectedAnswer",
        "cat4_correct_answer",
        "cat4CorrectAnswer",
        "advanced_correct_answer",
        "advancedCorrectAnswer",
        "target_answer",
        "targetAnswer",
        "answer",
        "text",
    )

    if form_data:
        correct_answer = _first_value(form_data, answer_keys)
        if correct_answer is not None:
            return correct_answer

    correct_answer = _first_value(quiz_data, answer_keys)
    if correct_answer is not None:
        return correct_answer

    category_4 = quiz_data.get("category_4", [])

    if isinstance(category_4, list) and category_4:
        for question in category_4:
            correct_answer = _first_value(question, answer_keys)
            if correct_answer is not None:
                return correct_answer

    if isinstance(category_4, dict):
        correct_answer = _first_value(category_4, answer_keys)
        if correct_answer is not None:
            return correct_answer

    return None


def _append_advanced_answer_status(sign_result, correct_answer):
    if not isinstance(sign_result, dict) or sign_result.get("error"):
        return sign_result

    predicted_answer = sign_result.get("answer") or sign_result.get("text")
    if predicted_answer is None:
        return sign_result

    if correct_answer is None:
        sign_result["raw_answer"] = predicted_answer
        sign_result["correct_answer"] = None
        sign_result["is_correct"] = False
        sign_result["answer_status"] = "Incorrect"
        sign_result["answer"] = f"{predicted_answer} - Incorrect"
        sign_result["status_note"] = "correct_answer not received by backend"
        return sign_result

    is_correct = (
        _normalize_answer(predicted_answer) == _normalize_answer(correct_answer)
    )
    status = "Correct" if is_correct else "Incorrect"

    sign_result["raw_answer"] = predicted_answer
    sign_result["correct_answer"] = correct_answer
    sign_result["is_correct"] = is_correct
    sign_result["answer_status"] = status
    sign_result["answer"] = f"{predicted_answer} - {status}"

    return sign_result


# -------------------------
#    QUIZ EVALUATION
# -------------------------
class QuizEvaluator:
    def __init__(self, quiz_data, level):
        self.quiz_data = quiz_data
        self.level = level
        self.relevant_categories = LEVEL_CATEGORIES[level]
        self.results = {
            "area_performance": defaultdict(lambda: {"correct": 0, "total": 0, "percentage": 0}),
            "category_scores": {},
            "correct_answers": 0,
            "total_questions": 0,
            "overall_score": 0
        }

    def evaluate_category(self, questions):
        correct = 0
        for q in questions:
            area = q.get("area", "").lower()
            is_correct = str(q.get("correct_answer")).upper() == str(q.get("user_answer")).upper()
            self.results["area_performance"][area]["total"] += 1

            if is_correct:
                correct += 1
                self.results["area_performance"][area]["correct"] += 1

        total = len(questions)
        return {
            "score": correct,
            "total": total,
            "percentage": round((correct / total) * 100, 2) if total else 0
        }

    def evaluate(self):
        for cat in self.relevant_categories:
            questions = self.quiz_data.get(cat, [])
            cat_result = self.evaluate_category(questions)
            self.results["category_scores"][cat] = cat_result
            self.results["correct_answers"] += cat_result["score"]
            self.results["total_questions"] += cat_result["total"]

        for area, stats in self.results["area_performance"].items():
            if stats["total"] > 0:
                stats["percentage"] = round((stats["correct"] / stats["total"]) * 100, 2)

        if self.results["total_questions"] > 0:
            self.results["overall_score"] = round(
                (self.results["correct_answers"] / self.results["total_questions"]) * 100, 2
            )

        return self.results


# -------------------------
#   ADVANCED ANALYZER
# -------------------------
class AdvancedAnalyzer:
    def __init__(self, evaluator_results, shap_model, level):
        self.results = evaluator_results
        self.model = shap_model
        self.level = level.lower()
        self.features = []
        self.shap_result = None

    def extract_features(self):
        features = []
        for i in range(1, 5):
            category = f"category_{i}"
            features.append(self.results["category_scores"].get(category, {}).get("percentage", 0))

        for area in VARIATIONS:
            features.append(self.results["area_performance"].get(area, {}).get("percentage", 0))

        self.features = features
        return features

    def analyze(self):
        self.extract_features()
        self.shap_result = self.model.explain(self.features)
        return self.shap_result

    def generate_insights(self):
        if not self.shap_result:
            self.analyze()

        area_scores = []
        for area in VARIATIONS:
            stats = self.results["area_performance"].get(area, None)
            if stats and stats["total"] > 0:
                area_scores.append({
                    "area": area,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "percentage": stats["percentage"],
                    "score_display": f"{stats['correct']} out of {stats['total']}"
                })

        area_scores_sorted = sorted(area_scores, key=lambda x: x["percentage"])
        weak = [a for a in area_scores_sorted if a["percentage"] < 70]
        strong = [a for a in area_scores_sorted if a["percentage"] >= 70]

        insights = {
            "level": self.level,
            "overall_performance": {
                "score": self.results["overall_score"],
                "assessment": self._assess(self.results["overall_score"]),
                "total_correct": self.results["correct_answers"],
                "total_questions": self.results["total_questions"]
            },
            "area_performance": area_scores,
            "areas_needing_improvement": weak[:3],
            "strong_areas": strong,
            "recommendations": self._recommend(weak, self.results["overall_score"])
        }

        return insights

    def _assess(self, score):
        if score >= 85: return "Excellent"
        if score >= 70: return "Good"
        if score >= 50: return "Needs Improvement"
        return "Critical"

    def _recommend(self, weak, score):
        recs = []
        if weak: recs.append(f"Primary focus area: {weak[0]['area']}")
        if score < 50:
            recs.append("Structured guided learning is strongly recommended")
        elif score < 70:
            recs.append("Balance practice between weak and strong areas")
        else:
            recs.append("Increase difficulty while maintaining consistency")
        return recs


# -------------------------
#    MAIN API
# -------------------------
@app.route("/api/quiz/submit", methods=["POST"])
def submit_quiz():
    print("✔ /api/quiz/submit hit")

    quizzes_raw = request.form.get("quizzes")
    level = request.form.get("level")
    user_id = request.form.get("user_id")

    if not quizzes_raw or not level or not user_id:
        return jsonify({"error": "Missing form fields"}), 400

    try:
        quiz_data = json.loads(quizzes_raw)
    except:
        return jsonify({"error": "Invalid quizzes JSON"}), 400

    # Evaluate text quiz
    evaluator = QuizEvaluator(quiz_data, level)
    eval_results = evaluator.evaluate()

    analyzer = AdvancedAnalyzer(eval_results, shap_model, level)
    insights = analyzer.generate_insights()

    response = {
        "user_id": user_id,
        "level": level,
        "results": insights
    }

    # --------------------------
    #    VIDEO HANDLING
    # --------------------------
    if level == "advanced" and "cat4" in request.files:

        video = request.files["cat4"]
        advanced_correct_answer = _get_advanced_correct_answer(quiz_data, request.form)
        print("Incoming video:", video.filename)
        print("Advanced correct answer:", advanced_correct_answer)

        if video.filename == "":
            return jsonify({"error": "Empty video filename"}), 400

        if not allowed_file(video.filename):
            return jsonify({"error": "Unsupported video format"}), 400

        # Save locally
        filename = secure_filename(f"{user_id}_{datetime.utcnow().timestamp()}.webm")
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        video.save(path)
        print("Saved video:", path)

        video_analysis = {}

        # -------- External sign recognition --------
        try:
            with open(path, "rb") as f:
                res = requests.post(
                    EXTERNAL_VIDEO_API_URL,
                    files={"cat4": (filename, f, "video/webm")},
                    timeout=60
                )
                res.raise_for_status()
                video_analysis["sign_recognition"] = _append_advanced_answer_status(
                    res.json(),
                    advanced_correct_answer
                )
        except Exception as e:
            video_analysis["sign_recognition"] = {"error": str(e)}

        # -------- Local eye contact with model predictions --------
        try:
            video_analysis["eye_contact"] = video_analyze_enhanced(path)
        except Exception as e:
            video_analysis["eye_contact"] = {"error": str(e)}

        response["video_analysis"] = video_analysis
        
        # Delete local temp video file
        try:
            if os.path.exists(path):
                os.remove(path)
                print("Deleted video:", path)
        except Exception as e:
            print("Error deleting video:", str(e))

    # Save attempt to DB
    payload = {
        "level": level,
        "results": insights,
        "user_id": user_id,
        "video_analysis": response.get("video_analysis"),
        "ml": analyzer.shap_result
    }

    try:
        save_res = save_attempt(payload)
        print("Saved attempt:", save_res["attempt_id"])
    except Exception as e:
        print("Error saving attempt to Mongo:", str(e))

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
