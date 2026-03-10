"""
Audio/Video to Sign Language Service
Pipeline: WAV → MFCC (with deltas) → normalize → pad → LSTM model → sign label
Model input:  (1, 128, 40)  float32
Model output: (1, 21)        softmax over 21 Tamil sign classes
"""
import os
import sys
import json
import subprocess
import numpy as np
import cv2
from typing import Any, Dict, List, Optional
import cloudinary
import cloudinary.api
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Constants matching training config ────────────────────────────────────────
MAX_LEN          = 128   # padded sequence length
N_MFCC           = 40    # number of MFCC coefficients
ADD_DELTAS       = True  # training used delta + delta-delta features  → 40×3 = 120
CONFIDENCE_THRESHOLD = 0.0


class AudioToSignService:
    def __init__(self):
        self._project_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sign_images_path = os.path.join(self._project_root, 'static', 'sign_images')
        self._models_dir   = os.path.join(self._project_root, 'models')

        # Preferred model filenames in order
        self._model_candidates = [
            'audio_to_sign_best.h5',
            'audio_to_sign_model.h5',
            'audio_to_sign_model.keras'
        ]
        self._norm_file     = os.path.join(self._models_dir, 'audio_to_sign_norm_stats.json')

        # Initialize Cloudinary
        self._init_cloudinary()

        os.makedirs(self.sign_images_path, exist_ok=True)
        os.makedirs(os.path.join(self._project_root, 'uploads'), exist_ok=True)

        self._norm          = self._load_norm_stats()
        self.model: Optional[Any] = self._load_ml_model()
        self.classes: List[str]   = self._norm.get('classes', []) if self._norm else []
        print(f"[Service] Classes ({len(self.classes)}): {self.classes}")
        sys.stdout.flush()

    def _init_cloudinary(self):
        """Initialize Cloudinary configuration from environment variables."""
        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET'),
            secure=True
        )
        self.cloudinary_folder = os.getenv('CLOUDINARY_FOLDER', 'Sign_Sight_Assets')
        print(f"[Service] ✓ Cloudinary configured: {os.getenv('CLOUDINARY_CLOUD_NAME')}/{self.cloudinary_folder}")
        sys.stdout.flush()

    # ── Loading ────────────────────────────────────────────────────────────────

    def _discover_model_files(self) -> List[str]:
        """Return absolute paths of all candidate model files, ordered by preference."""
        found: List[str] = []
        if not os.path.isdir(self._models_dir):
            return []

        # First, add explicit preferred names if they exist
        for name in self._model_candidates:
            path = os.path.join(self._models_dir, name)
            if os.path.exists(path):
                found.append(path)

        # Then add any other .h5/.keras files not already in the list
        for fname in os.listdir(self._models_dir):
            if not (fname.endswith('.h5') or fname.endswith('.keras')):
                continue
            path = os.path.join(self._models_dir, fname)
            if path not in found:
                found.append(path)

        print(f"[Service] Model candidates: {found}")
        return found

    def _load_norm_stats(self) -> Optional[Dict]:
        if not os.path.exists(self._norm_file):
            print(f"[Service] Norm stats not found: {self._norm_file}")
            sys.stdout.flush()
            return None
        with open(self._norm_file, 'r') as f:
            stats = json.load(f)
        print(f"[Service] ✓ Norm stats loaded — {len(stats.get('classes',[]))} classes")
        sys.stdout.flush()
        return stats

    def _load_ml_model(self) -> Optional[Any]:
        """Try to load a model from any suitable file in the models/ folder.

        Tries each discovered model file via tf.keras, then standalone keras.
        Returns the first successfully loaded model.
        """
        model_files = self._discover_model_files()
        if not model_files:
            print(f"[Service] No model files found in {self._models_dir}")
            sys.stdout.flush()
            return None

        last_error: Optional[Exception] = None

        for path in model_files:
            print(f"[Service] Trying model file: {path}")
            sys.stdout.flush()
            # Try tf.keras
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(path)
                print(f"[Service] ✓ Model loaded via tf.keras from {path} | input={model.input_shape} output={model.output_shape}")
                sys.stdout.flush()
                return model
            except Exception as e1:
                print(f"[Service] tf.keras failed for {path}: {e1}")
                sys.stdout.flush()
                last_error = e1
            # Try standalone keras
            try:
                import keras  # type: ignore
                model = keras.models.load_model(path)
                print(f"[Service] ✓ Model loaded via standalone keras from {path}")
                sys.stdout.flush()
                return model
            except Exception as e2:
                print(f"[Service] standalone keras failed for {path}: {e2}")
                sys.stdout.flush()
                last_error = e2

        print(f"[Service] ✗ Could not load any model from {self._models_dir}. Last error: {last_error}")
        sys.stdout.flush()
        return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_audio_to_signs(self, audio_path: str) -> Dict:
        """WAV file → predicted sign label + Cloudinary image URL."""
        try:
            wav_path = self._ensure_wav(audio_path)
            sign_label, confidence = self._predict_from_wav(wav_path)
            if wav_path != audio_path and os.path.exists(wav_path):
                os.remove(wav_path)

            duration = self._get_audio_duration(audio_path)

            if sign_label:
                # Get Cloudinary URL
                cloudinary_url = self.get_sign_image_url(sign_label)

                if cloudinary_url:
                    signs = [{'word': sign_label, 'sign': sign_label,
                              'confidence': round(confidence, 4), 'found': True}]
                    sign_images = [{'word': sign_label,
                                    'image_url': cloudinary_url,
                                    'sign_name': sign_label,
                                    'confidence': round(confidence, 4)}]
                else:
                    # GIF not found in Cloudinary
                    print(f"[Service] ⚠️  Sign '{sign_label}' predicted but GIF not found in Cloudinary")
                    sys.stdout.flush()
                    signs = [{'word': sign_label, 'sign': sign_label,
                              'confidence': round(confidence, 4), 'found': False}]
                    sign_images = []
            else:
                signs = [{'word': 'unknown', 'sign': 'unknown', 'found': False}]
                sign_images = []

            return {'text': sign_label or 'unknown', 'signs': signs,
                    'sign_images': sign_images, 'duration': duration}
        except Exception as e:
            raise Exception(f"Error processing audio: {e}")

    def process_video_to_signs(self, video_path: str) -> Dict:
        """Video file → extract audio → predicted sign label."""
        try:
            audio_path = self._extract_audio_from_video(video_path)
            video_info = self._get_video_info(video_path)
            result = self.process_audio_to_signs(audio_path)
            result['video_info'] = video_info
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return result
        except Exception as e:
            raise Exception(f"Error processing video: {e}")

    def text_to_signs(self, text: str) -> Dict:
        """Text → map words to available sign classes."""

        words = text.lower().strip().split()

        signs: List[Dict] = []
        sign_images: List[Dict] = []

        for word in words:
            clean = word.strip('.,!?;:')

            if clean in self.classes:
                cloudinary_url = self.get_sign_image_url(clean)

                if cloudinary_url:
                    signs.append({
                        'word': clean,
                        'sign': clean,
                        'confidence': 1.0,
                        'found': True
                    })

                    sign_images.append({
                        'word': clean,
                        'image_url': cloudinary_url,
                        'sign_name': clean,
                        'confidence': 1.0
                    })
                else:
                    # Word in classes but GIF not in Cloudinary
                    signs.append({
                        'word': clean,
                        'sign': clean,
                        'confidence': 1.0,
                        'found': False
                    })

            else:
                signs.append({
                    'word': clean,
                    'sign': 'unknown',
                    'found': False,
                    'letters': list(clean)
                })

                sign_images.append({
                    'word': clean,
                    'fingerspell': True,
                    'letters': list(clean)
                })

        return {
            'signs': signs,
            'sign_images': sign_images,
            'word_count': len(words)
        }

    def get_sign_image_url(self, sign_name: str) -> Optional[str]:
        """
        Get Cloudinary URL for sign GIF and validate it exists.
        Returns None if the GIF doesn't exist in Cloudinary.
        """
        try:
            # Construct the public_id for Cloudinary
            # Note: Cloudinary public_id should NOT include extension, but your files were uploaded with .gif.gif
            # So we try both patterns
            public_ids_to_try = [
                f"{self.cloudinary_folder}/{sign_name}.gif",  # Try with .gif.gif (your current upload)
                f"{self.cloudinary_folder}/{sign_name}"       # Try without extension (correct way)
            ]

            for public_id in public_ids_to_try:
                try:
                    cloudinary.api.resource(public_id, resource_type="image")
                    # If no exception, resource exists - return the URL
                    cloudinary_url = cloudinary.CloudinaryImage(public_id).build_url(
                        secure=True,
                        resource_type="image"
                    )
                    print(f"[Service] ✓ Cloudinary GIF found: {public_id}")
                    sys.stdout.flush()
                    return cloudinary_url
                except cloudinary.api.NotFound:
                    continue

            # Neither pattern found
            print(f"[Service] ✗ Cloudinary GIF not found for: {sign_name} (tried {len(public_ids_to_try)} patterns)")
            sys.stdout.flush()
            return None

        except Exception as e:
            print(f"[Service] Cloudinary error for {sign_name}: {e}")
            sys.stdout.flush()
            return None

    def get_sign_image_path(self, sign_name: str) -> Optional[str]:
        """
        Deprecated: Kept for backward compatibility.
        Use get_sign_image_url() for Cloudinary URLs.
        """
        return self.get_sign_image_url(sign_name)

    # ── Core prediction pipeline ───────────────────────────────────────────────

    def _predict_from_wav(self, wav_path: str):
        """Extract MFCCs, normalize, pad, run model → (label, confidence)."""
        if self.model is None or self._norm is None:
            return None, 0.0
        try:
            import librosa

            # 1. Load audio
            y, sr = librosa.load(wav_path, sr=16000, mono=True)

            # 2. Extract MFCCs
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=N_MFCC,
                n_fft=1024,
                hop_length=512
            )

            if ADD_DELTAS:
                delta1 = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                features = np.concatenate([mfcc, delta1, delta2], axis=0)  # (120, T)
            else:
                features = mfcc  # (40, T)

            features = features.T  # (T, 120)
            # DEBUG INFO
            print("Features shape:", features.shape)
            print("mu shape:", np.array(self._norm['mu']).shape)
            print("sigma shape:", np.array(self._norm['sigma']).shape)
            sys.stdout.flush()

            # 3. Normalize using saved mu / sigma
            mu    = np.array(self._norm['mu'],    dtype=np.float32)
            sigma = np.array(self._norm['sigma'], dtype=np.float32)
            sigma = np.where(sigma == 0, 1.0, sigma)
            features = (features - mu) / sigma

            # 4. Pad / truncate to MAX_LEN
            T = features.shape[0]
            if T >= MAX_LEN:
                features = features[:MAX_LEN]
            else:
                pad = np.zeros((MAX_LEN - T, features.shape[1]), dtype=np.float32)
                features = np.vstack([features, pad])

            # 5. Run model
            x = features[np.newaxis, ...]            # (1, 128, 120)
            probs = self.model.predict(x, verbose=0) # (1, 21)
            idx   = int(np.argmax(probs[0]))
            confidence = float(probs[0][idx])

            print(f"[Service] Predicted class index={idx} confidence={confidence:.4f}")

            # if confidence < CONFIDENCE_THRESHOLD:
            #     print(f"[Service] Confidence too low ({confidence:.4f} < {CONFIDENCE_THRESHOLD})")
            #     return None, confidence
            if confidence < CONFIDENCE_THRESHOLD:
                print(f"[Service] Low confidence but returning prediction anyway")

            label = self.classes[idx] if idx < len(self.classes) else f"SIGN_{idx}"
            print(f"[Service] ✓ Prediction: {label} ({confidence:.4f})")
            sys.stdout.flush()
            return label, confidence

        except Exception as e:
            print(f"[Service] Prediction error: {e}")
            sys.stdout.flush()
            return None, 0.0

    # ── Audio / Video helpers ──────────────────────────────────────────────────

    def _ensure_wav(self, audio_path: str) -> str:
        """Convert any audio to clean 16kHz mono WAV using ffmpeg."""
        if audio_path.lower().endswith('.wav'):
            # Still re-encode to ensure correct sample rate / channels
            out = audio_path.rsplit('.', 1)[0] + '_16k.wav'
        else:
            out = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        try:
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', audio_path,
                 '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', out],
                capture_output=True, text=True
            )
            print("FFmpeg return code:", r.returncode)
            print("FFmpeg stderr:", r.stderr[-200:])
            print("Converted file:", out)
            sys.stdout.flush()

            if r.returncode == 0:
                return out
        except FileNotFoundError:
            pass
        return audio_path

    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg."""
        audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'

        print(f"[Service] Extracting audio from video: {video_path}")
        sys.stdout.flush()

        try:
            # First check if video has audio stream using ffprobe
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=codec_type', '-of', 'default=nw=1', video_path],
                capture_output=True, text=True
            )

            if 'codec_type=audio' not in probe.stdout:
                print(f"[Service] ⚠️  Video has no audio stream")
                sys.stdout.flush()
                raise Exception("Video file does not contain an audio stream. Please upload a video with audio or use an audio file directly.")

            # Extract audio
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', video_path,
                 '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path],
                capture_output=True, text=True
            )

            print(f"[Service] FFmpeg extraction return code: {r.returncode}")
            sys.stdout.flush()

            if r.returncode != 0:
                # Check if file was created anyway
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    print(f"[Service] Audio file created despite FFmpeg warning")
                    sys.stdout.flush()
                    return audio_path
                print(f"[Service] FFmpeg stderr: {r.stderr[-300:]}")
                sys.stdout.flush()
                raise Exception(f"Failed to extract audio from video: {r.stderr[-200:]}")

            print(f"[Service] ✓ Audio extracted successfully: {audio_path}")
            sys.stdout.flush()
            return audio_path

        except FileNotFoundError as e:
            print(f"[Service] FFmpeg/ffprobe not found, trying pydub fallback")
            sys.stdout.flush()
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(video_path)
                audio.export(audio_path, format='wav', parameters=['-ar', '16000', '-ac', '1'])
                return audio_path
            except Exception as pydub_error:
                raise Exception(f"Could not extract audio. FFmpeg not found and pydub failed: {pydub_error}")
        except Exception as e:
            print(f"[Service] Video extraction error: {e}")
            sys.stdout.flush()
            raise

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            from pydub import AudioSegment
            return len(AudioSegment.from_file(audio_path)) / 1000.0
        except Exception:
            return 0.0

    def _get_video_info(self, video_path: str) -> Dict:
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return {'fps': fps, 'frame_count': fc, 'width': w, 'height': h,
                    'duration': fc / fps if fps > 0 else 0}
        except Exception:
            return {}

