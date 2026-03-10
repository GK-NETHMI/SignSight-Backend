from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging - show INFO for debugging, but filter out health checks
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        # Hide GET / requests (health checks) but show everything else
        return not ('GET / HTTP' in record.getMessage() and 'GET /' in record.getMessage())

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)
werkzeug_logger.addFilter(HealthCheckFilter())

app.logger.setLevel(logging.INFO)

# Configure CORS for TypeScript frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://localhost:4200"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type", "Content-Length"],
        "supports_credentials": False,
        "send_wildcard": False,
        "max_age": 3600
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

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin', 'http://localhost:3000')
    if origin in ["http://localhost:3000", "http://localhost:5173", "http://localhost:4200"]:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'false')
    return response

# Health check
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
