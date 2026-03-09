"""
Audio and Video to Tamil Sign Language conversion routes
Handles audio/video file uploads and converts them to TSL sign demonstrations
"""
from flask import Blueprint, request, jsonify, send_file, make_response
import os
import sys
from werkzeug.utils import secure_filename
from services.audio_to_sign_service import AudioToSignService

bp = Blueprint('audio_to_sign', __name__)
audio_service = AudioToSignService()

ALLOWED_AUDIO_EXTENSIONS = {'wav'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def build_signs_response(result):
    """
    Merge image_url directly into each sign item.
    image_url is ALWAYS a non-null string so the frontend can safely call .startsWith() on it.
    """
    sign_images_map = {
        s.get('sign_name', s.get('word', '')): s
        for s in result.get('sign_images', [])
    }
    signs_with_images = []
    for sign in result.get('signs', []):
        sign_name = sign.get('sign', 'unknown')
        img = sign_images_map.get(sign_name, sign_images_map.get(sign.get('word', ''), {}))
        # Always provide a string image_url — never null
        image_url = img.get('image_url', f'/api/audio-to-sign/get-sign-image/{sign_name}') \
            if sign.get('found') else f'/api/audio-to-sign/get-sign-image/{sign_name}'
        signs_with_images.append({
            'sign': sign_name,
            'word': sign.get('word', sign_name),
            'image_url': image_url,
            'confidence': sign.get('confidence', 0),
            'found': sign.get('found', False),
        })
    return signs_with_images

@bp.route('/upload-audio', methods=['POST'])
def upload_audio():
    try:
        print("\n" + "="*70)
        print("[UPLOAD-AUDIO] NEW REQUEST")
        print("="*70)
        print(f"[upload-audio] Received files: {list(request.files.keys())}")
        print(f"[upload-audio] Received form: {list(request.form.keys())}")
        sys.stdout.flush()

        if 'audio' not in request.files:
            print("[upload-audio] ERROR: 'audio' field missing from request")
            sys.stdout.flush()
            return jsonify({'error': "No audio file provided. Send file with field name 'audio'", 'success': False}), 400

        file = request.files['audio']
        print(f"[upload-audio] Filename: '{file.filename}'")
        sys.stdout.flush()

        if file.filename == '':
            print("[upload-audio] ERROR: Empty filename")
            sys.stdout.flush()
            return jsonify({'error': 'No file selected', 'success': False}), 400

        if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
            print(f"[upload-audio] ERROR: Invalid file type '{file.filename}'")
            sys.stdout.flush()
            return jsonify({'error': f'Invalid file type. Only .wav files are accepted', 'success': False}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        print(f"[upload-audio] Saved to: {filepath}")
        print(f"[upload-audio] Processing audio file...")
        sys.stdout.flush()

        result = audio_service.process_audio_to_signs(filepath)

        print(f"[upload-audio] Processing complete!")
        print(f"[upload-audio] Result: {result['text']}")
        print("="*70 + "\n")
        sys.stdout.flush()

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            'text': result['text'],
            'signs': build_signs_response(result),
            'duration': result.get('duration', 0),
            'success': True
        }), 200

    except Exception as e:
        print(f"[upload-audio] EXCEPTION: {e}")
        sys.stdout.flush()
        return jsonify({'error': str(e), 'success': False}), 500

@bp.route('/upload-video', methods=['POST'])
def upload_video():
    try:
        print(f"[upload-video] Received files: {list(request.files.keys())}")
        print(f"[upload-video] Received form: {list(request.form.keys())}")

        if 'video' not in request.files:
            print("[upload-video] ERROR: 'video' field missing from request")
            return jsonify({'error': "No video file provided. Send file with field name 'video'", 'success': False}), 400

        file = request.files['video']
        print(f"[upload-video] Filename: '{file.filename}'")

        if file.filename == '':
            print("[upload-video] ERROR: Empty filename")
            return jsonify({'error': 'No file selected', 'success': False}), 400

        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            print(f"[upload-video] ERROR: Invalid file type '{file.filename}'")
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}', 'success': False}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        print(f"[upload-video] Saved to: {filepath}")

        result = audio_service.process_video_to_signs(filepath)
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            'text': result['text'],
            'signs': build_signs_response(result),
            'duration': result.get('duration', 0),
            'video_info': result.get('video_info', {}),
            'success': True
        }), 200

    except Exception as e:
        print(f"[upload-video] EXCEPTION: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@bp.route('/text-to-signs', methods=['POST'])
def text_to_signs():
    """
    Convert Tamil text directly to TSL signs
    Expected input: { "text": "Tamil text here" }
    """
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'success': False
            }), 400

        text = data['text']

        if not text.strip():
            return jsonify({
                'error': 'Text cannot be empty',
                'success': False
            }), 400

        result = audio_service.text_to_signs(text)

        return jsonify({
            'text': text,
            'signs': build_signs_response(result),
            'word_count': result.get('word_count', 0),
            'success': True
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@bp.route('/get-sign-image/<sign_name>', methods=['GET'])
def get_sign_image(sign_name):
    try:
        image_path = audio_service.get_sign_image_path(sign_name)

        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Sign image not found', 'success': False}), 404

        # Detect correct mimetype from actual file extension
        ext = image_path.rsplit('.', 1)[-1].lower()
        mimetype_map = {
            'gif':  'image/gif',
            'png':  'image/png',
            'jpg':  'image/jpeg',
            'jpeg': 'image/jpeg',
        }
        mimetype = mimetype_map.get(ext, 'image/gif')

        # Send file inline (not as download) with no-cache so GIF animation always replays
        response = make_response(send_file(
            image_path,
            mimetype=mimetype,
            as_attachment=False,
            conditional=False
        ))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


