#!/usr/bin/env python3
"""
SignSight Backend - Comprehensive System Verification
Tests all components and ensures everything is properly configured.
"""

import sys
import os
import json
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        logger.info("✓ Python version is compatible")
        return True
    else:
        logger.error("✗ Python 3.8+ is required")
        return False

def check_dependencies() -> Tuple[bool, Dict[str, bool]]:
    """Check if all required dependencies are installed."""
    logger.info("\n" + "="*60)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("="*60)

    dependencies = {
        'flask': 'Flask',
        'flask_cors': 'flask-cors',
        'dotenv': 'python-dotenv',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'cv2': 'opencv-python',
        'pymongo': 'pymongo',
        'bson': 'pymongo (bson)',
        'librosa': 'librosa',
        'cloudinary': 'cloudinary',
    }

    results = {}
    all_ok = True

    for import_name, display_name in dependencies.items():
        try:
            __import__(import_name)
            logger.info(f"  ✓ {display_name}")
            results[display_name] = True
        except ImportError as e:
            logger.warning(f"  ✗ {display_name}: {str(e)}")
            results[display_name] = False
            all_ok = False

    return all_ok, results

def check_file_structure() -> bool:
    """Check if all required files and directories exist."""
    logger.info("\n" + "="*60)
    logger.info("CHECKING FILE STRUCTURE")
    logger.info("="*60)

    required_files = [
        'main.py',
        'config.py',
        'app.py',
        'mentor_backend.py',
        'requirements.txt',
        '.env',
        'routes/audio_to_sign_routes.py',
        'services/audio_to_sign_service.py',
    ]

    required_dirs = [
        'uploads',
        'models',
        'static/sign_images',
        'routes',
        'services',
    ]

    all_ok = True

    for file in required_files:
        if os.path.exists(file):
            logger.info(f"  ✓ {file}")
        else:
            logger.warning(f"  ✗ {file} (MISSING)")
            all_ok = False

    for directory in required_dirs:
        if os.path.isdir(directory):
            logger.info(f"  ✓ {directory}/")
        else:
            logger.warning(f"  ✗ {directory}/ (MISSING)")
            # Create it if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            logger.info(f"    → Created {directory}/")

    return all_ok

def check_environment() -> bool:
    """Check environment configuration."""
    logger.info("\n" + "="*60)
    logger.info("CHECKING ENVIRONMENT CONFIGURATION")
    logger.info("="*60)

    from dotenv import load_dotenv
    load_dotenv()

    required_env_vars = [
        ('PORT', 'Flask server port'),
        ('FLASK_ENV', 'Flask environment'),
        ('CLOUDINARY_CLOUD_NAME', 'Cloudinary cloud name'),
    ]

    all_ok = True

    for var, description in required_env_vars:
        value = os.getenv(var)
        if value:
            masked_value = value[:10] + '...' if len(str(value)) > 10 else value
            logger.info(f"  ✓ {var}={masked_value}")
        else:
            logger.warning(f"  ✗ {var} (NOT SET) - {description}")
            all_ok = False

    return all_ok

def check_app_imports() -> bool:
    """Check if the main app can be imported."""
    logger.info("\n" + "="*60)
    logger.info("CHECKING APP IMPORTS")
    logger.info("="*60)

    try:
        logger.info("  Importing main...")
        from main import app, create_app
        logger.info("  ✓ main.py imports successfully")

        logger.info("  Importing config...")
        from config import get_config
        logger.info("  ✓ config.py imports successfully")

        logger.info("  Checking Flask app...")
        if app:
            logger.info("  ✓ Flask app instance created")

            logger.info("  Checking registered routes...")
            routes = []
            for rule in app.url_map.iter_rules():
                routes.append(str(rule))

            logger.info(f"    Found {len(routes)} routes")
            for route in sorted(routes):
                if 'static' not in route:
                    logger.info(f"    - {route}")

            return True
        else:
            logger.error("  ✗ Flask app not created")
            return False

    except Exception as e:
        logger.error(f"  ✗ Failed to import: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_database_config() -> bool:
    """Check database configuration."""
    logger.info("\n" + "="*60)
    logger.info("CHECKING DATABASE CONFIGURATION")
    logger.info("="*60)

    try:
        mongo_uri = os.getenv('MONGO_URI')
        if mongo_uri:
            logger.info(f"  ✓ MONGO_URI is configured (length: {len(mongo_uri)} chars)")
        else:
            logger.warning("  ⚠️  MONGO_URI not configured in .env")

        use_fallback = os.getenv('USE_DB_FALLBACK', 'true').lower() in ('1', 'true', 'yes')
        logger.info(f"  ✓ USE_DB_FALLBACK={use_fallback}")

        if use_fallback:
            logger.info("  → Database will use in-memory fallback if connection fails")

        return True
    except Exception as e:
        logger.error(f"  ✗ Database config error: {str(e)}")
        return False

def check_models() -> bool:
    """Check if ML models are present."""
    logger.info("\n" + "="*60)
    logger.info("CHECKING ML MODELS")
    logger.info("="*60)

    model_dir = 'models'
    norm_file = os.path.join(model_dir, 'audio_to_sign_norm_stats.json')

    if os.path.exists(model_dir):
        logger.info(f"  ✓ Models directory exists")

        files = os.listdir(model_dir)
        logger.info(f"    Found {len(files)} files:")
        for f in files:
            fpath = os.path.join(model_dir, f)
            size = os.path.getsize(fpath)
            logger.info(f"    - {f} ({size/1024/1024:.1f} MB)")

        if os.path.exists(norm_file):
            logger.info(f"  ✓ Norm stats file found")
            try:
                with open(norm_file, 'r') as f:
                    norm_data = json.load(f)
                    classes = norm_data.get('classes', [])
                    logger.info(f"    Classes: {len(classes)} - {', '.join(classes[:5])}...")
                    return True
            except Exception as e:
                logger.error(f"  ✗ Failed to read norm stats: {str(e)}")
                return False
        else:
            logger.warning(f"  ⚠️  Norm stats file not found: {norm_file}")
            return False
    else:
        logger.warning(f"  ⚠️  Models directory not found")
        return False

def run_all_checks() -> bool:
    """Run all verification checks."""
    logger.info("\n")
    logger.info("╔" + "="*60 + "╗")
    logger.info("║" + " "*60 + "║")
    logger.info("║" + "  SignSight Backend - System Verification".center(60) + "║")
    logger.info("║" + " "*60 + "║")
    logger.info("╚" + "="*60 + "╝")

    results = []

    results.append(("Python Version", check_python_version()))
    deps_ok, deps = check_dependencies()
    results.append(("Dependencies", deps_ok))
    results.append(("File Structure", check_file_structure()))
    results.append(("Environment Config", check_environment()))
    results.append(("App Imports", check_app_imports()))
    results.append(("Database Config", check_database_config()))
    results.append(("ML Models", check_models()))

    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)

    passed = 0
    failed = 0

    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {check_name}")
        if result:
            passed += 1
        else:
            failed += 1

    logger.info("="*60)
    logger.info(f"\nTotal: {passed} passed, {failed} failed\n")

    if failed == 0:
        logger.info("🎉 All checks passed! The system is ready to run.")
        logger.info("\nTo start the server, run:")
        logger.info("  cd /Users/farsithfawzer/Desktop/Farsith\\ AudioToSign/SignSight-Backend")
        logger.info("  python3 main.py")
        logger.info("\nOr with gunicorn:")
        logger.info("  gunicorn -w 4 -b 0.0.0.0:5080 main:app")
        return True
    else:
        logger.warning("\n⚠️  Some checks failed. Please review the output above and fix any issues.")
        return False

if __name__ == '__main__':
    success = run_all_checks()
    sys.exit(0 if success else 1)

