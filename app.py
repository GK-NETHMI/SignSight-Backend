from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Create upload directories if they don't exist
UPLOAD_FOLDER_AUDIO = 'uploads/audio'
UPLOAD_FOLDER_TEXT = 'uploads/text'
os.makedirs(UPLOAD_FOLDER_AUDIO, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_TEXT, exist_ok=True)

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

    # Save file
    filepath = os.path.join(UPLOAD_FOLDER_AUDIO, file.filename)
    file.save(filepath)

    # Return dummy response
    response = get_dummy_sign_response()
    return jsonify({"sign_output": response}), 200

# Endpoint 2: Text to Sign
@app.route('/text-to-sign', methods=['POST'])
def text_to_sign():
    if 'file' not in request.files:
        return jsonify({"error": "No text file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save file
    filepath = os.path.join(UPLOAD_FOLDER_TEXT, file.filename)
    file.save(filepath)

    # Return dummy response
    response = get_dummy_sign_response()
    return jsonify({"sign_output": response}), 200

if __name__ == '__main__':
    app.run(debug=True)
