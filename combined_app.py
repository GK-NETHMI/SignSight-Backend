"""Combined WSGI entrypoint that serves both the mentor dashboard and the
audio-to-sign API on a single port using a DispatcherMiddleware.

Run with:
    python3 combined_app.py

This mounts the original `mentor_backend.app` as the default application and
the `app` from `app.py` (audio-to-sign) under `/api/audio-to-sign` so all
existing endpoints keep their paths.
"""
import os
from werkzeug.serving import run_simple

# Import the mentor Flask app and register the audio blueprint onto it so a
# single Flask/Werkzeug process can serve both APIs under one port.
import mentor_backend as mentor_module
from routes.audio_to_sign_routes import bp as audio_bp


def make_combined_app():
    """Return the mentor app with audio blueprint registered."""
    app = mentor_module.app
    # Avoid duplicate registration if this file is reloaded during development
    if 'audio_to_sign' not in [bp.name for bp in app.blueprints.values()]:
        try:
            app.register_blueprint(audio_bp, url_prefix='/api/audio-to-sign')
        except Exception:
            # Blueprint may already be registered when importing app.py directly
            pass
    return app


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5080'))
    use_debug = os.getenv('FLASK_ENV', 'development') == 'development'

    application = make_combined_app()
    print(f"Starting combined SignSight app on http://0.0.0.0:{port}")
    run_simple('0.0.0.0', port, application, use_reloader=use_debug, use_debugger=use_debug)

