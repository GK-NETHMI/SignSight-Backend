"""
SignSight Unified Backend Application
Combines Mentor Dashboard API and Audio-to-Sign API on a single port using Flask blueprints.

Run with:
    python main.py
    
Or with gunicorn:
    gunicorn -w 4 -b 0.0.0.0:5080 main:app
"""

import os
import sys
import logging
import warnings
from dotenv import load_dotenv

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

def create_app() -> 'Flask':  # type: ignore
    """Create and configure the Flask application."""
    from flask import Flask
    from flask_cors import CORS

    app: 'Flask' = Flask(__name__)  # type: ignore

    # Configure basic settings
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MODEL_FOLDER'] = 'models'

    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

    # Configure CORS for all origins (can be restricted later)
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000", "http://localhost:5173", "http://localhost:4200", "*"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Accept"],
            "expose_headers": ["Content-Type", "Content-Length"],
            "supports_credentials": False,
            "max_age": 3600
        }
    })

    # Register health check
    @app.route('/')
    def health_check():
        """Health check endpoint."""
        from flask import jsonify
        return jsonify({'status': 'ok', 'service': 'SignSight Backend'}), 200

    # Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        from flask import jsonify
        return jsonify({'error': 'Endpoint not found', 'success': False}), 404

    @app.errorhandler(500)
    def internal_error(error):
        from flask import jsonify
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error', 'success': False}), 500

    @app.errorhandler(413)
    def request_entity_too_large(error):
        from flask import jsonify
        return jsonify({'error': 'File too large. Maximum size is 16MB', 'success': False}), 413

    # Register blueprint: Audio-to-Sign API
    try:
        from routes.audio_to_sign_routes import bp as audio_bp
        app.register_blueprint(audio_bp, url_prefix='/api/audio-to-sign')
        logger.info("✓ Audio-to-Sign API registered at /api/audio-to-sign")
    except Exception as e:
        logger.error(f"✗ Failed to register audio-to-sign routes: {e}")
        sys.exit(1)

    # Register blueprint: Mentor Dashboard API
    try:
        import mentor_backend
        # Copy all routes from mentor_backend.app to main app
        for rule in mentor_backend.app.url_map.iter_rules():
            if 'static' not in str(rule):
                # Get the view function from mentor_backend app
                endpoint = rule.endpoint
                view_func = mentor_backend.app.view_functions.get(endpoint)
                if view_func:
                    # Register the route on the main app
                    app.add_url_rule(
                        rule.rule,
                        endpoint=f'mentor_{endpoint}',
                        view_func=view_func,
                        methods=rule.methods - {'HEAD', 'OPTIONS'}
                    )
        logger.info("✓ Mentor Dashboard API registered")
    except Exception as e:
        logger.error(f"⚠️  Mentor dashboard routes not available: {e}")
        import traceback
        traceback.print_exc()

    # Register blueprint: Jeran ML Model Inference API (Optional - Branch 3)
    try:
        import jeranapp
        # Copy all routes from jeranapp to main app
        for rule in jeranapp.app.url_map.iter_rules():
            if 'static' not in str(rule) and str(rule) != '/':
                endpoint = rule.endpoint
                view_func = jeranapp.app.view_functions.get(endpoint)
                if view_func:
                    route_path = f"/api/ml{rule.rule}" if rule.rule != '/' else '/api/ml'
                    app.add_url_rule(
                        route_path,
                        endpoint=f'jeran_{endpoint}',
                        view_func=view_func,
                        methods=rule.methods - {'HEAD', 'OPTIONS'}
                    )
        logger.info("✓ Jeran ML Model Inference API registered at /api/ml/*")
    except AttributeError as e:
        logger.debug(f"⚠️  Jeran ML API skipped (optional - MediaPipe version issue): {str(e)[:100]}")
    except Exception as e:
        logger.debug(f"⚠️  Jeran ML API not available (optional): {type(e).__name__}: {str(e)[:100]}")

    # Register blueprint: Emotion Video Analysis API (Optional - Branch 4)
    try:
        import final
        # Copy all routes from final.app to main app
        for rule in final.app.url_map.iter_rules():
            if 'static' not in str(rule):
                endpoint = rule.endpoint
                view_func = final.app.view_functions.get(endpoint)
                if view_func:
                    route_path = f"/api/emotion{rule.rule}" if rule.rule != '/' else '/api/emotion'
                    app.add_url_rule(
                        route_path,
                        endpoint=f'final_{endpoint}',
                        view_func=view_func,
                        methods=rule.methods - {'HEAD', 'OPTIONS'}
                    )
        logger.info("✓ Emotion Video Analysis API registered at /api/emotion/*")
    except AttributeError as e:
        logger.debug(f"⚠️  Emotion API skipped (optional - MediaPipe version issue): {str(e)[:100]}")
    except Exception as e:
        logger.debug(f"⚠️  Emotion analysis API not available (optional): {type(e).__name__}: {str(e)[:100]}")

    # Register admin status endpoint
    @app.route('/api/admin/status', methods=['GET'])
    def admin_status():
        """Return service status."""
        from flask import jsonify
        return jsonify({
            'service': 'SignSight Backend',
            'status': 'running',
            'environment': os.getenv('FLASK_ENV', 'development')
        }), 200

    return app


# Create the application instance
app = create_app()


def run_development() -> None:
    """Run the application in development mode."""
    port: int = int(os.getenv('PORT', '5080'))
    debug: bool = os.getenv('FLASK_ENV', 'development') == 'development'

    print("\n" + "="*80)
    print("🎉 SignSight Backend - All 4 Branches Integrated".center(80))
    print("="*80)
    print(f"\n🚀 Starting on http://localhost:{port}")
    print(f"📝 Mode: {os.getenv('FLASK_ENV', 'development').upper()}")
    print(f"🔍 Debug: {'ON' if debug else 'OFF'}")
    print("\n" + "-"*80)
    print("APIs Available:")
    print("  ✓ Branch 1: Audio-to-Sign API (/api/audio-to-sign/*)")
    print("  ✓ Branch 2: Mentor Dashboard (/api/*)")
    print("  ✓ Branch 3: ML Inference (optional - /api/ml/*)")
    print("  ✓ Branch 4: Emotion Analysis (optional - /api/emotion/*)")
    print("-"*80)
    print("Endpoints:")
    print("  Health: GET   http://localhost:{}/".format(port))
    print("  Admin:  GET   http://localhost:{}/api/admin/status".format(port))
    print("  Audio:  POST  http://localhost:{}/api/audio-to-sign/text-to-signs".format(port))
    print("  Mentor: GET   http://localhost:{}/api/admin/mentors".format(port))
    print("-"*80 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=debug)


if __name__ == '__main__':
    run_development()

