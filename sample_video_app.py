from flask import Flask, request, jsonify
import random
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'mock_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

WORDS = [
    "good","mother","father"
]

@app.route("/process-video", methods=["POST"])
def process_video():
    print("✔ /process-video hit")

    if "cat4" not in request.files:
        return jsonify({"success": False, "error": "cat4 not provided"}), 400

    video = request.files["cat4"]
    if video.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    filename = secure_filename(video.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video.save(path)

    detected = random.choice(WORDS)

    res = {
        "success": True,
        "answer": detected,
        "text": detected,
        "confidence": round(random.uniform(0.6, 0.99), 2),
        "metadata": {
            "filename": filename,
            "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2)
        }
    }

    return jsonify(res), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
