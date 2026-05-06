import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson.json_util import dumps, RELAXED_JSON_OPTIONS
import certifi, json
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

# ----------------------------------------------------------------
# CONFIG  — identical to your dbConfig.py
# ----------------------------------------------------------------
MONGO_URI = (
    "mongodb+srv://signsight8_db_user:IhafUkyQov1hzFdG@signsight.6fgqsty.mongodb.net/"
    "?retryWrites=true&w=majority"
)
DB_NAME = "signsight"

LEVELS = ["basic", "intermediate", "advanced"]
AREAS  = ["family", "alphabet", "numbers", "objects", "actions", "emotions"]

# ----------------------------------------------------------------
# APP + DB  (single client)
# ----------------------------------------------------------------
app  = Flask(__name__)
CORS(app, origins="*")

client = MongoClient(
    MONGO_URI,
    server_api=ServerApi("1"),
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=8000
)
db            = client[DB_NAME]
users_col     = db["users"]
attempts_col  = db["attempts"]
snapshots_col = db["progress_snapshots"]
mentors_col   = db["mentors"]


def to_json(obj):
    """Serialize ObjectId / datetime for JSON."""
    return json.loads(dumps(obj, json_options=RELAXED_JSON_OPTIONS))


# ================================================================
# 1.  LIST USERS
# ================================================================
@app.route("/api/<mentorEmail>/dashboard/users", methods=["GET"])
def get_users(mentorEmail):
    mentor = mentors_col.find_one({"email": mentorEmail})
    mentor_id = str(mentor["_id"]) if mentor else None

    if not mentor_id:
        return jsonify({"error": "Mentor not found"}), 404


    print("🚀  Fetching users for mentor:", mentor_id)
    users = list(
        users_col.find(
            {"mentorId": mentor_id},
            { "userId": 1, "createdAt": 1, "lastSeenAt": 1, "stats": 1, "_id": 0 }
        ).sort("createdAt", -1)
    )
    return jsonify(to_json(users)), 200


# ================================================================
# 2.  FULL USER SUMMARY  (everything the dashboard needs in one call)
# ================================================================
@app.route("/api/dashboard/users/<user_id>/summary", methods=["GET"])
def user_summary(user_id):
    user = users_col.find_one({"userId": user_id}, {"_id": 0})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # all attempts, newest first
    attempts = list(attempts_col.find({"userId": user_id}).sort("submittedAt", -1))

    # bucket by level
    by_level = {lvl: [] for lvl in LEVELS}
    for att in attempts:
        by_level.setdefault(att.get("level", "basic"), []).append(att)

    # ---- per-level aggregation ----
    level_stats = {}
    for lvl in LEVELS:
        atts   = by_level[lvl]
        count  = len(atts)
        scores = [a["quiz"]["overallScore"] for a in atts if "quiz" in a]

        avg   = round(sum(scores) / len(scores), 2) if scores else 0
        best  = max(scores) if scores else 0
        worst = min(scores) if scores else 0

        # area averages
        area_acc = {a: {"sum": 0, "cnt": 0} for a in AREAS}
        for a in atts:
            for area, data in a.get("areas", {}).items():
                if area in area_acc:
                    area_acc[area]["sum"] += data.get("percentage", 0)
                    area_acc[area]["cnt"] += 1

        area_avgs = {
            area: round(t["sum"] / t["cnt"], 2) if t["cnt"] else 0
            for area, t in area_acc.items()
        }

        # assessment counts
        assessments = {}
        for a in atts:
            key = a.get("quiz", {}).get("assessment", "Unknown")
            assessments[key] = assessments.get(key, 0) + 1

        level_stats[lvl] = {
            "attemptCount" : count,
            "avgScore"     : avg,
            "bestScore"    : best,
            "worstScore"   : worst,
            "areaAverages" : area_avgs,
            "assessments"  : assessments,
            "videoAttempts": sum(1 for a in atts if a.get("videoAnalysis"))
        }

    # ---- progress series per level (ordered by attemptNumber) ----
    progress = {}
    for lvl in LEVELS:
        progress[lvl] = [
            {
                "attempt": a.get("attemptNumber", 0),
                "score"  : a.get("quiz", {}).get("overallScore", 0),
                "date"   : a.get("submittedAt")
            }
            for a in sorted(by_level[lvl], key=lambda x: x.get("attemptNumber", 0))
        ]

    # ---- latest attempt ----
    latest = None
    if attempts:
        a = attempts[0]
        latest = {
            "level"         : a.get("level"),
            "score"         : a.get("quiz", {}).get("overallScore", 0),
            "assessment"    : a.get("quiz", {}).get("assessment"),
            "areas"         : a.get("areas", {}),
            "insights"      : a.get("insights", {}),
            "submittedAt"   : a.get("submittedAt"),
            "videoAnalysis" : a.get("videoAnalysis")
        }

    # ---- global area averages (across every attempt) ----
    g_acc = {a: {"sum": 0, "cnt": 0} for a in AREAS}
    for att in attempts:
        for area, data in att.get("areas", {}).items():
            if area in g_acc:
                g_acc[area]["sum"] += data.get("percentage", 0)
                g_acc[area]["cnt"] += 1

    global_avgs = {
        area: round(t["sum"] / t["cnt"], 2) if t["cnt"] else 0
        for area, t in g_acc.items()
    }

    sorted_areas = sorted(global_avgs.items(), key=lambda x: x[1])
    weak   = [{"area": a, "avg": s} for a, s in sorted_areas if s < 70]
    strong = [{"area": a, "avg": s} for a, s in sorted_areas if s >= 70]

    # ---- assemble ----
    summary = {
        "user"               : user,
        "levelStats"         : level_stats,
        "progress"           : progress,
        "latest"             : latest,
        "globalAreaAverages" : global_avgs,
        "weakAreas"          : weak,
        "strongAreas"        : strong,
        "totalAttempts"      : len(attempts)
    }
    return jsonify(to_json(summary)), 200


# ================================================================
# 3.  PAGINATED ATTEMPTS
# ================================================================
@app.route("/api/dashboard/users/<user_id>/attempts", methods=["GET"])
def user_attempts(user_id):
    level = request.args.get("level")
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 8))

    query = {"userId": user_id}
    if level and level in LEVELS:
        query["level"] = level

    total    = attempts_col.count_documents(query)
    attempts = list(
        attempts_col.find(query, {"_id": 0})
        .sort("submittedAt", -1)
        .skip((page - 1) * limit)
        .limit(limit)
    )

    return jsonify(to_json({
        "attempts": attempts,
        "total"   : total,
        "page"    : page,
        "limit"   : limit
    })), 200


# ================================================================
# 4.  GLOBAL OVERVIEW  (cross-user stats — optional admin card)
# ================================================================
@app.route("/api/dashboard/overview", methods=["GET"])
def global_overview():
    total_users    = users_col.count_documents({})
    total_attempts = attempts_col.count_documents({})

    level_avgs = {}
    for lvl in LEVELS:
        pipeline = [
            {"$match": {"level": lvl}},
            {"$group": {"_id": None, "avgScore": {"$avg": "$quiz.overallScore"}, "count": {"$sum": 1}}}
        ]
        res = list(attempts_col.aggregate(pipeline))
        level_avgs[lvl] = {
            "avgScore": round(res[0]["avgScore"], 2) if res else 0,
            "count"   : res[0]["count"] if res else 0
        }

    return jsonify(to_json({
        "totalUsers"    : total_users,
        "totalAttempts" : total_attempts,
        "levelAverages" : level_avgs
    })), 200

@app.route("/api/mentors", methods=["POST"])
def create_mentor():
    data = request.json

    name = data.get("name")
    email = data.get("email")
    firebase_uid = data.get("firebaseUid")

    if not name or not email or not firebase_uid:
        return jsonify({
            "error": "Missing required fields"
        }), 400

    # prevent duplicate mentors
    existing = mentors_col.find_one({
        "$or": [
            {"email": email},
            {"firebaseUid": firebase_uid}
        ]
    })

    if existing:
        return jsonify({
            "error": "Mentor already exists"
        }), 409

    mentor = {
        "name": name,
        "email": email,
        "firebaseUid": firebase_uid,
        "maxStudents": 5,
        "createdAt":  datetime.datetime.utcnow()
    }

    mentors_col.insert_one(mentor)

    return jsonify({
        "message": "Mentor created successfully",
        "mentor": {
            "name": name,
            "email": email
        }
    }), 201

@app.route("/api/admin/mentors", methods=["GET"])
def get_mentors():
    mentors = []

    for m in mentors_col.find():
        mentors.append({
            "id": str(m["_id"]),
            "name": m.get("name"),
            "email": m.get("email"),
            "users": m.get("users", []),
            "usersCount": len(m.get("users", [])),
            "maxUsers": 5
        })

    return jsonify(mentors), 200



@app.route("/api/admin/students", methods=["GET"])
def get_students():
    users = list(
        users_col.find(
            {},
            {
                "userId": 1,
                "createdAt": 1,
                "lastSeenAt": 1,
                "mentorId": 1
            }
        ).sort("createdAt", -1)
    )

    for u in users:
        u["_id"] = str(u["_id"])
        if "mentorId" in u:
            u["mentorId"] = str(u["mentorId"])

    return jsonify(users), 200


@app.route("/api/admin/save-mentor-users", methods=["POST"])
def save_mentor_users():
    data = request.json

    mentor_id = data.get("mentorId")
    user_ids = data.get("userIds", [])

    if not mentor_id or not isinstance(user_ids, list):
        return jsonify({"error": "Invalid payload"}), 400

    if len(user_ids) > 5:
        return jsonify({
            "error": "A mentor can have maximum 5 users"
        }), 400

    mentor = mentors_col.find_one({"_id": ObjectId(mentor_id)})
    if not mentor:
        return jsonify({"error": "Mentor not found"}), 404

    # 🔹 convert userIds → ObjectId
    user_object_ids = [ObjectId(uid) for uid in user_ids]

    # 🔄 1. Remove mentorId from users previously assigned to this mentor
    users_col.update_many(
        {"mentorId": mentor_id},
        {"$unset": {"mentorId": ""}}
    )

    # 🔄 2. Assign mentorId to selected users
    users_col.update_many(
        {"_id": {"$in": user_object_ids}},
        {"$set": {"mentorId": mentor_id}}
    )

    # 🔄 3. Update mentor.users array
    mentors_col.update_one(
        {"_id": ObjectId(mentor_id)},
        {"$set": {"users": user_ids}}
    )

    return jsonify({
        "message": "Mentor users saved successfully",
        "mentorId": mentor_id,
        "userCount": len(user_ids)
    }), 200


# ================================================================
# STUDENT
# ================================================================

@app.route("/api/students", methods=["POST"])
def create_student():
    data = request.json
    username     = data.get("username")
    name         = data.get("name")
    email        = data.get("email")
    age          = data.get("age")
    gender       = data.get("gender")
    password     = data.get("password")
    firebase_uid = data.get("firebaseUid")

    if not all([username, name, email, password, firebase_uid]):
        return jsonify({"message": "Missing required fields (username, name, email, password, firebaseUid)"}), 400

    existing = users_col.find_one({
        "$or": [{"username": username}, {"email": email}]
    })
    if existing:
        return jsonify({
            "message": "Username already taken" if existing.get("username") == username else "Email already registered"
        }), 409

    # Hash password before storing
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    result = users_col.insert_one({
        "userId":      username,
        "username":    username,
        "name":        name,
        "email":       email,
        "age":         age,
        "gender":      gender,
        "firebaseUid": firebase_uid,
        "password":    hashed_password,
        "createdAt":   datetime.datetime.utcnow()
    })

    return jsonify({
        "_id":      str(result.inserted_id),
        "username": username,
        "name":     name,
        "email":    email,
    }), 201


@app.route("/api/students/by-username/<username>", methods=["GET"])
def get_student_by_username(username):
    student = users_col.find_one({"username": username})
    if not student:
        return jsonify({"message": "Student not found"}), 404

    return jsonify({
        "_id":      str(student["_id"]),
        "username": student.get("username"),
        "name":     student.get("name"),
        "email":    student.get("email"),
    }), 200


@app.route("/api/login", methods=["POST"])
def login():
    """Student login endpoint - validates username and password."""
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    student = users_col.find_one({"username": username})
    if not student:
        return jsonify({"error": "Invalid username or password"}), 401

    # Check if password field exists
    stored_password = student.get("password")
    if not stored_password:
        return jsonify({"error": "Invalid username or password"}), 401

    # Verify password
    if not check_password_hash(stored_password, password):
        return jsonify({"error": "Invalid username or password"}), 401

    # Return user data on successful login
    return jsonify({
        "_id":       str(student["_id"]),
        "username":  student.get("username"),
        "name":      student.get("name"),
        "email":     student.get("email"),
        "age":       student.get("age"),
        "gender":    student.get("gender"),
        "message":   "Login successful"
    }), 200


# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    print("🚀  SignSight Mentor Dashboard API — port 5080")
    app.run(debug=True, host="0.0.0.0", port=5080)

