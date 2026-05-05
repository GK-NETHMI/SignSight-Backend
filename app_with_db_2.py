from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from collections import defaultdict
from werkzeug.utils import secure_filename
import requests
from dbConfig import save_attempt
from eyeContact import allowed_file, video_analyze
from dbConfig import init_db, save_attempt

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

CATEGORY_INDEX = {
    "category_1": 0,
    "category_2": 1,
    "category_3": 2,
    "category_4": 3
}

class RuleBasedModel:
    def explain(self, features, level):
        category_scores = features[:4]
        area_scores = [score for score in features[4:] if score > 0]

        relevant_indexes = [
            CATEGORY_INDEX[category]
            for category in LEVEL_CATEGORIES[level]
            if category in CATEGORY_INDEX
        ]
        relevant_scores = [category_scores[index] for index in relevant_indexes]

        category_average = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0
        area_average = sum(area_scores) / len(area_scores) if area_scores else category_average
        predicted_score = round((category_average * 0.7) + (area_average * 0.3), 2)

        return {
            "model_type": "rule_based",
            "predicted_score": predicted_score,
            "base_value": round(category_average, 2),
            "shap_values": [],
            "components": {
                "category_average": round(category_average, 2),
                "area_average": round(area_average, 2)
            }
        }


rule_based_model = RuleBasedModel()


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
    def __init__(self, evaluator_results, level):
        self.results = evaluator_results
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
        self.shap_result = rule_based_model.explain(self.features, self.level)
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

    analyzer = AdvancedAnalyzer(eval_results, level)
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
        print("Incoming video:", video.filename)

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
                video_analysis["sign_recognition"] = res.json()
        except Exception as e:
            video_analysis["sign_recognition"] = {"error": str(e)}

        # -------- Local eye contact --------
        try:
            video_analysis["eye_contact"] = video_analyze(path)
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
    app.run(debug=True, host="0.0.0.0", port=5000)
