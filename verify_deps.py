#!/usr/bin/env python3
import sys

print("=" * 60)
print("SIGNSIGHT BACKEND - DEPENDENCY VERIFICATION")
print("=" * 60)

packages = {
    "flask": "Flask",
    "flask_cors": "Flask-CORS",
    "tensorflow": "TensorFlow",
    "numpy": "NumPy",
    "cv2": "OpenCV",
    "mediapipe": "MediaPipe",
    "librosa": "Librosa",
    "pymongo": "PyMongo",
    "cloudinary": "Cloudinary",
}

errors = []
for import_name, display_name in packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {display_name:<20} v{version}")
    except Exception as e:
        errors.append(f"✗ {display_name}: {str(e)}")
        print(f"✗ {display_name}: Failed to import")

print("=" * 60)
if errors:
    print(f"⚠️  {len(errors)} package(s) failed to load")
    for error in errors:
        print(f"  {error}")
else:
    print(" ALL DEPENDENCIES VERIFIED SUCCESSFULLY!")
    print(f" Python {sys.version.split()[0]} configured correctly")
print("=" * 60)

