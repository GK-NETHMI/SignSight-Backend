import datetime
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson.json_util import dumps, RELAXED_JSON_OPTIONS
import certifi, json, logging
from bson import ObjectId
# ----------------------------------------------------------------
# CONFIG  — identical to your dbConfig.py
# ----------------------------------------------------------------
# Prefer environment variable for credentials / cluster address. Fall back to the
# previous hard-coded value only if MONGO_URI is not provided in the env.
MONGO_URI = os.getenv(
    "MONGO_URI",
    (
        "mongodb+srv://signsight8_db_user:IhafUkyQov1hzFdG@signsight.6fgqsty.mongodb.net/"
        "?retryWrites=true&w=majority"
    ),
)

# When set to 'false' the service will fail fast if it cannot reach MongoDB.
# Set USE_DB_FALLBACK=true to keep the in-memory fallback behaviour used during
# local development when a real MongoDB is not reachable.
USE_DB_FALLBACK = os.getenv("USE_DB_FALLBACK", "true").lower() in ("1", "true", "yes")
DB_NAME = "signsight"

LEVELS = ["basic", "intermediate", "advanced"]
AREAS  = ["family", "alphabet", "numbers", "objects", "actions", "emotions"]

# ----------------------------------------------------------------
# APP + DB  (single client)
# ----------------------------------------------------------------
app  = Flask(__name__)
CORS(app, origins="*")

logging.basicConfig(level=logging.INFO)
DB_CONNECTED = False
USING_DB_FALLBACK = False

# Attempt to connect to MongoDB; if it fails (e.g. missing DNS or credentials)
# fall back to lightweight in-memory fake collections so the API can still run
# for local development / integration testing.
try:
    client = MongoClient(
        MONGO_URI,
        server_api=ServerApi("1"),
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=8000,
    )
    # Verify connectivity early with a ping (will raise on failure)
    client.admin.command('ping')
    db = client[DB_NAME]
    users_col = db["users"]
    attempts_col = db["attempts"]
    snapshots_col = db["progress_snapshots"]
    mentors_col = db["mentors"]
    logging.info("Connected to MongoDB cluster: %s", MONGO_URI)
    DB_CONNECTED = True
    USING_DB_FALLBACK = False
except Exception as exc:
    logging.error("MongoDB connection failed: %s", exc)
    if not USE_DB_FALLBACK:
        # Fail fast so deployment/CI surfaces configuration problems immediately
        logging.error("USE_DB_FALLBACK is false — exiting due to DB connection failure")
        raise
    logging.warning("Falling back to in-memory collections because USE_DB_FALLBACK=%s", USE_DB_FALLBACK)
    DB_CONNECTED = False
    USING_DB_FALLBACK = True

    # Minimal in-memory collection / cursor implementations to keep endpoints working
    class FakeCursor(list):
        def sort(self, *a, **k):
            return self
        def skip(self, n):
            return self
        def limit(self, n):
            return self

    class FakeCollection:
        def __init__(self):
            self._data = []

        def find(self, *args, **kwargs):
            return FakeCursor(self._data.copy())

        def find_one(self, *args, **kwargs):
            return None

        def count_documents(self, query):
            return 0

        def aggregate(self, pipeline):
            return []

        def insert_one(self, doc):
            oid = ObjectId()
            doc_copy = doc.copy()
            doc_copy["_id"] = oid
            self._data.append(doc_copy)
            class R: pass
            r = R()
            r.inserted_id = oid
            return r

        def update_many(self, filter, update):
            class R: pass
            r = R()
            r.modified_count = 0
            return r

        def update_one(self, filter, update):
            class R: pass
            r = R()
            r.modified_count = 0
            return r

    # create fake collections
    db = None
    users_col = FakeCollection()
    attempts_col = FakeCollection()
    snapshots_col = FakeCollection()
    mentors_col = FakeCollection()


def to_json(obj):
    """Serialize ObjectId / datetime for JSON."""
    return json.loads(dumps(obj, json_options=RELAXED_JSON_OPTIONS))


@app.route('/api/admin/status', methods=['GET'])
def admin_status():
    """Return service status including DB connectivity (safe, masked)."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(MONGO_URI)
        host = parsed.hostname or ''
    except Exception:
        host = ''

    return jsonify({
        'dbConnected': bool(DB_CONNECTED),
        'usingFallback': bool(USING_DB_FALLBACK),
        'mongoHost': host,
        'useDbFallbackEnv': USE_DB_FALLBACK
    }), 200


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
    firebase_uid = data.get("firebaseUid")

    if not all([username, name, email, firebase_uid]):
        return jsonify({"message": "Missing required fields"}), 400

    existing = users_col.find_one({
        "$or": [{"username": username}, {"email": email}]
    })
    if existing:
        return jsonify({
            "message": "Username already taken" if existing.get("username") == username else "Email already registered"
        }), 409

    result = users_col.insert_one({
        "userId":      username,
        "username":    username,
        "name":        name,
        "email":       email,
        "age":         age,
        "gender":      gender,
        "firebaseUid": firebase_uid,
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

# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    mentor_port = int(os.getenv("MENTOR_PORT", os.getenv("PORT", "5081")))
    mentor_debug = os.getenv("FLASK_ENV", "development") == "development"
    print(f"🚀  SignSight Mentor Dashboard API — port {mentor_port}")
    app.run(debug=mentor_debug, host="0.0.0.0", port=mentor_port)

    
