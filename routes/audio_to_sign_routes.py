from flask import Blueprint, request, jsonify, send_file, make_response
import os
import sys
from werkzeug.utils import secure_filename
from services.audio_to_sign_service import AudioToSignService

try:
    import cloudinary
    import cloudinary.uploader
except Exception:
    cloudinary = None

bp = Blueprint('audio_to_sign', __name__)
# Delay heavy service initialization until first request to avoid long startup time
audio_service = None

def get_audio_service():
    global audio_service
    if audio_service is None:
        audio_service = AudioToSignService()
    return audio_service

ALLOWED_AUDIO_EXTENSIONS = {'wav'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def build_signs_response(result):

    sign_images_map = {
        s.get('sign_name', s.get('word', '')): s
        for s in result.get('sign_images', [])
    }
    signs_with_images = []
    for sign in result.get('signs', []):
        sign_name = sign.get('sign', 'unknown')
        img = sign_images_map.get(sign_name, sign_images_map.get(sign.get('word', ''), {}))
        # Get Cloudinary URL from sign_images or construct placeholder
        image_url = img.get('image_url', '') if sign.get('found') else ''
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

        svc = get_audio_service()
        result = svc.process_audio_to_signs(filepath)

        print(f"[upload-audio] Processing complete!")
        print(f"[upload-audio] Result: {result['text']}")

        # Build response
        signs_response = build_signs_response(result)

        # Ensure we have at least one sign
        if not signs_response or len(signs_response) == 0:
            print("[upload-audio] WARNING: No signs in response!")
            sys.stdout.flush()

        response_data = {
            'success': True,
            'text': result['text'],
            'signs': signs_response,
            'duration': result.get('duration', 0),
        }

        print(f"[upload-audio] Response signs count: {len(response_data['signs'])}")
        print(f"[upload-audio] First sign: {response_data['signs'][0] if response_data['signs'] else 'NONE'}")

        # Import json to pretty print the response
        import json
        print(f"[upload-audio] JSON Response being sent:")
        print(json.dumps(response_data, indent=2))
        print("="*70 + "\n")
        sys.stdout.flush()

        if os.path.exists(filepath):
            os.remove(filepath)

        response = jsonify(response_data)
        print(f"[upload-audio] Response status: 200")
        print(f"[upload-audio] Response headers: {dict(response.headers)}")
        sys.stdout.flush()

        return response, 200

    except Exception as e:
        print(f"[upload-audio] EXCEPTION: {e}")
        sys.stdout.flush()
        return jsonify({'error': str(e), 'success': False}), 500

@bp.route('/upload-video', methods=['POST'])
def upload_video():
    try:
        print("\n" + "="*70)
        print("[UPLOAD-VIDEO] NEW REQUEST")
        print("="*70)
        print(f"[upload-video] Received files: {list(request.files.keys())}")
        print(f"[upload-video] Received form: {list(request.form.keys())}")
        sys.stdout.flush()

        if 'video' not in request.files:
            print("[upload-video] ERROR: 'video' field missing from request")
            sys.stdout.flush()
            return jsonify({'error': "No video file provided. Send file with field name 'video'", 'success': False}), 400

        file = request.files['video']
        print(f"[upload-video] Filename: '{file.filename}'")
        sys.stdout.flush()

        if file.filename == '':
            print("[upload-video] ERROR: Empty filename")
            sys.stdout.flush()
            return jsonify({'error': 'No file selected', 'success': False}), 400

        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            print(f"[upload-video] ERROR: Invalid file type '{file.filename}'")
            sys.stdout.flush()
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}', 'success': False}), 400

        # Save uploaded file to uploads/
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        print(f"[upload-video] Saved to: {filepath}")
        sys.stdout.flush()

        # Process the video to signs using existing pipeline
        print(f"[upload-video] Processing video file...")
        sys.stdout.flush()
        svc = get_audio_service()
        result = svc.process_video_to_signs(filepath)

        # Build sign responses (existing helper)
        signs_response = build_signs_response(result)

        # Read requested size from form (small|medium|large)
        size_label = request.form.get('size', 'medium')
        if size_label not in ('small', 'medium', 'large'):
            size_label = 'medium'

        # Map size to width (pixels)
        SIZE_MAP = {'small': 320, 'medium': 640, 'large': 1280}
        target_width = SIZE_MAP.get(size_label, 640)

        # Upload the original video to Cloudinary as a video resource
        video_url_to_return = None
        try:
            # Skip upload if cloudinary not available
            if cloudinary is None:
                raise Exception("cloudinary package not available")
            # Upload as video resource
            upload_res = cloudinary.uploader.upload(
                filepath,
                resource_type='video',
                folder=audio_service.cloudinary_folder
            )
            public_id = upload_res.get('public_id')
            secure_url = upload_res.get('secure_url')
            print(f"[upload-video] Cloudinary upload OK: public_id={public_id} url={secure_url}")
            sys.stdout.flush()

            # Build a transformed Cloudinary URL scaled to the requested width
            # Use build_url on CloudinaryVideo to ensure resource_type='video'
            video_url_to_return = cloudinary.CloudinaryVideo(public_id).build_url(
                transformation={'width': target_width, 'crop': 'scale'},
                resource_type='video',
                secure=True
            )

        except Exception as cloud_err:
            # If Cloudinary upload fails, log and fallback (return None or local path if you can serve it)
            print(f"[upload-video] Cloudinary upload failed: {cloud_err}")
            sys.stdout.flush()
            video_url_to_return = None

        response_data = {
            'text': result.get('text', ''),
            'signs': signs_response if signs_response else [],
            'duration': result.get('duration', 0),
            'video_info': result.get('video_info', {}),
            'video_url': video_url_to_return,   # may be None if upload failed
            'success': True
        }

        print(f"[upload-video] Processing complete!")
        print(f"[upload-video] Result: {result.get('text')}")
        print(f"[upload-video] Signs count: {len(response_data['signs'])}")
        print("="*70 + "\n")
        sys.stdout.flush()

        # Clean up local upload file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass

        return jsonify(response_data), 200

    except Exception as e:
        print(f"[upload-video] EXCEPTION: {e}")
        sys.stdout.flush()
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

        svc = get_audio_service()
        result = svc.text_to_signs(text)

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
        from flask import redirect

        print(f"[get-sign-image] Request for: {sign_name}")
        sys.stdout.flush()

        svc = get_audio_service()
        cloudinary_url = svc.get_sign_image_url(sign_name)

        if not cloudinary_url:
            print(f"[get-sign-image] ERROR: Image not found in Cloudinary for '{sign_name}'")
            sys.stdout.flush()
            return jsonify({'error': 'Sign image not found', 'success': False}), 404

        print(f"[get-sign-image] Redirecting to: {cloudinary_url}")
        sys.stdout.flush()

        # Redirect to Cloudinary URL
        return redirect(cloudinary_url, code=302)

    except Exception as e:
        print(f"[get-sign-image] ERROR: {e}")
        sys.stdout.flush()
        return jsonify({'error': str(e), 'success': False}), 500


