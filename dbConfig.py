# dbConfig.py
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ServerSelectionTimeoutError
from bson.objectid import ObjectId
import certifi

# =========================
# CONFIG
# =========================
MONGO_URI = (
    "mongodb+srv://signsight8_db_user:IhafUkyQov1hzFdG@signsight.6fgqsty.mongodb.net/"
    "?retryWrites=true&w=majority"
)

DB_NAME = "signsight"

# =========================
# GLOBALS (initialized once)
# =========================
client = None
db = None

users_col = None
attempts_col = None
snapshots_col = None


# =========================
# DB INITIALIZATION
# =========================
def init_db():
    """
    Call ONCE when app starts.
    """
    global client, db, users_col, attempts_col, snapshots_col

    try:
        client = MongoClient(
            MONGO_URI,
            server_api=ServerApi("1"),
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000
        )

        # Force connection test
        client.admin.command("ping")
        print("✅ MongoDB connected successfully")

        db = client[DB_NAME]

        users_col = db["users"]
        attempts_col = db["attempts"]
        snapshots_col = db["progress_snapshots"]

        ensure_indexes()

    except ServerSelectionTimeoutError as e:
        print("❌ MongoDB connection failed")
        raise e


# =========================
# INDEXES
# =========================
def ensure_indexes():
    users_col.create_index("userId", unique=True)

    attempts_col.create_index([("userId", 1), ("submittedAt", -1)])
    attempts_col.create_index([("userId", 1), ("level", 1)])
    attempts_col.create_index([("level", 1), ("submittedAt", -1)])

    snapshots_col.create_index([("userId", 1), ("area", 1), ("recordedAt", -1)])

    print("✅ MongoDB indexes ensured")


# =========================
# USER HANDLING
# =========================
def ensure_user(user_id: str, profile: dict = None):
    """
    Create user document if not exists.
    """
    if profile is None:
        profile = {}

    now = datetime.utcnow()

    users_col.update_one(
        {"userId": user_id},
        {
            "$setOnInsert": {
                "userId": user_id,
                "createdAt": now,
                "profile": profile,
                "stats": {
                    "totalAttempts": 0,
                    "levels": {}
                }
            },
            "$set": {
                "lastSeenAt": now
            }
        },
        upsert=True
    )


# =========================
# ATTEMPT SAVE
# =========================
def save_attempt(payload: dict):
    """
    Saves one quiz attempt.
    """
    user_id = payload["user_id"]
    level = payload["level"]
    results = payload["results"]

    ensure_user(user_id)

    # Attempt number per level
    attempt_no = attempts_col.count_documents({
        "userId": user_id,
        "level": level
    }) + 1

    # Normalize areas
    areas = {}
    for a in results.get("area_performance", []):
        areas[a["area"]] = {
            "correct": a["correct"],
            "total": a["total"],
            "percentage": a["percentage"],
            "score_display": a["score_display"]
        }

    overall = results.get("overall_performance", {})

    attempt_doc = {
        "userId": user_id,
        "level": level,
        "attemptNumber": attempt_no,
        "submittedAt": datetime.utcnow(),

        "quiz": {
            "totalQuestions": overall.get("total_questions", 0),
            "correctAnswers": overall.get("total_correct", 0),
            "overallScore": overall.get("score", 0),
            "assessment": overall.get("assessment")
        },

        "areas": areas,

        "insights": {
            "weakAreas": [w["area"] for w in results.get("areas_needing_improvement", [])],
            "strongAreas": [s["area"] for s in results.get("strong_areas", [])],
            "recommendations": results.get("recommendations", [])
        },

        "videoAnalysis": payload.get("video_analysis"),
        "ml": payload.get("ml"),

        "createdAt": datetime.utcnow()
    }

    res = attempts_col.insert_one(attempt_doc)

    # Update user stats
    users_col.update_one(
        {"userId": user_id},
        {
            "$inc": {
                "stats.totalAttempts": 1,
                f"stats.levels.{level}": 1
            },
            "$set": {
                "stats.lastAttemptAt": datetime.utcnow()
            }
        }
    )

    # Optional snapshots (for charts)
    snapshots = []
    for area, data in areas.items():
        snapshots.append({
            "userId": user_id,
            "level": level,
            "area": area,
            "attempt": attempt_no,
            "score": data["percentage"],
            "recordedAt": datetime.utcnow()
        })

    if snapshots:
        snapshots_col.insert_many(snapshots)

    return {"attempt_id": str(res.inserted_id)}