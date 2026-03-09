from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging - reduce noise from health checks
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)  # Only show warnings and errors, not INFO

# Configure CORS for TypeScript frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://localhost:4200"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Import routes
from routes import audio_to_sign_routes

# Register blueprints
app.register_blueprint(audio_to_sign_routes.bp, url_prefix='/api/audio-to-sign')

# Health check
# Commented out - was for testing purposes, causing repeated GET requests
# @app.route('/')
# def health_check():
#     return jsonify({
#         'status': 'running',
#         'message': 'SignSight Backend is live',
#         'endpoints': {
#             'upload_audio': '/api/audio-to-sign/upload-audio',
#             'upload_video': '/api/audio-to-sign/upload-video',
#             'text_to_signs': '/api/audio-to-sign/text-to-signs',
#             'get_sign_image': '/api/audio-to-sign/get-sign-image/<sign_name>'
#         }
#     }), 200

# Simple health check without verbose logging
@app.route('/')
def health_check():
    return jsonify({'status': 'ok'}), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'success': False}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'success': False}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB', 'success': False}), 413

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5080))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    import sys
    # Force unbuffered output so all print statements appear immediately
    sys.stdout.reconfigure(line_buffering=True)
    print(f"\nStarting Flask server on http://localhost:{port}")
    print(f"Python: {sys.executable}")
    print(f"Debug mode: {debug}\n")
    sys.stdout.flush()
    app.run(host="0.0.0.0", port=port, debug=debug)
