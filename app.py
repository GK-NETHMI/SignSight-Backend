from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}
ALLOWED_TEXT_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

# Create upload directories if they don't exist
UPLOAD_FOLDER_AUDIO = 'uploads/audio'
UPLOAD_FOLDER_TEXT = 'uploads/text'
os.makedirs(UPLOAD_FOLDER_AUDIO, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_TEXT, exist_ok=True)

def allowed_file(filename, allowed_extensions):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Dummy response function
from dummy_responses import get_dummy_sign_response

# Endpoint 1: Audio to Sign
@app.route('/audio-to-sign', methods=['POST'])
def audio_to_sign():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({"error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"}), 400

    # Secure the filename to prevent path traversal attacks
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER_AUDIO, filename)
    file.save(filepath)

    # Return dummy response
    response = get_dummy_sign_response()
    return jsonify({"sign_output": response, "filename": filename}), 200

# Endpoint 2: Text to Sign
@app.route('/text-to-sign', methods=['POST'])
def text_to_sign():
    if 'file' not in request.files:
        return jsonify({"error": "No text file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_TEXT_EXTENSIONS):
        return jsonify({"error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_TEXT_EXTENSIONS)}"}), 400

    # Secure the filename to prevent path traversal attacks
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER_TEXT, filename)
    file.save(filepath)

    # Return dummy response
    response = get_dummy_sign_response()
    return jsonify({"sign_output": response, "filename": filename}), 200

if __name__ == '__main__':
    app.run(debug=True)

