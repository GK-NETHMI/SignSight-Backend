"""
SignSight Backend Configuration Module
Centralized configuration for the entire application.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== Flask Configuration ====================
FLASK_ENV: str = os.getenv('FLASK_ENV', 'development')
DEBUG: bool = FLASK_ENV == 'development'
PORT: int = int(os.getenv('PORT', '5080'))

# ==================== Database Configuration ====================
MONGO_URI: str = os.getenv(
    'MONGO_URI',
    'mongodb+srv://signsight8_db_user:IhafUkyQov1hzFdG@signsight.6fgqsty.mongodb.net/?retryWrites=true&w=majority'
)
DB_NAME: str = 'signsight'
USE_DB_FALLBACK: bool = os.getenv('USE_DB_FALLBACK', 'true').lower() in ('1', 'true', 'yes')
DB_CONNECTION_TIMEOUT_MS: int = 8000

# ==================== File Upload Configuration ====================
MAX_FILE_SIZE: int = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER: str = 'uploads'
MODEL_FOLDER: str = 'models'

ALLOWED_AUDIO_EXTENSIONS: set = {'wav', 'mp3', 'ogg', 'flac'}
ALLOWED_VIDEO_EXTENSIONS: set = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# ==================== Cloudinary Configuration ====================
CLOUDINARY_CLOUD_NAME: str = os.getenv('CLOUDINARY_CLOUD_NAME', '')
CLOUDINARY_API_KEY: str = os.getenv('CLOUDINARY_API_KEY', '')
CLOUDINARY_API_SECRET: str = os.getenv('CLOUDINARY_API_SECRET', '')
CLOUDINARY_FOLDER: str = os.getenv('CLOUDINARY_FOLDER', 'Sign_Sight_Assets')

# ==================== CORS Configuration ====================
CORS_ORIGINS: List[str] = [
    'http://localhost:3000',
    'http://localhost:5173',
    'http://localhost:4200',
    '*'
]

CORS_METHODS: List[str] = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_HEADERS: List[str] = ['Content-Type', 'Authorization', 'Accept']
CORS_EXPOSE_HEADERS: List[str] = ['Content-Type', 'Content-Length']

# ==================== ML Model Configuration ====================
MAX_MFCC_LENGTH: int = 128
N_MFCC: int = 40
ADD_DELTAS: bool = True
CONFIDENCE_THRESHOLD: float = 0.4

SIGN_CLASSES: List[str] = [
    'amma', 'anbalippu', 'apple', 'arambam', 'aruvi', 'illam', 'kaalai',
    'kadal', 'kattadam', 'keylvi', 'master', 'mownam', 'mudivu', 'nandri',
    'neram', 'nimmadhi', 'phone', 'samayal', 'thambi', 'udhavi', 'urakkam'
]

# ==================== Database Level Configuration ====================
LEVELS: List[str] = ['basic', 'intermediate', 'advanced']
AREAS: List[str] = ['family', 'alphabet', 'numbers', 'objects', 'actions', 'emotions']

# ==================== Logging Configuration ====================
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL: str = 'INFO'

def get_config() -> Dict:
    """Get all configuration as a dictionary."""
    return {
        'flask': {
            'env': FLASK_ENV,
            'debug': DEBUG,
            'port': PORT
        },
        'database': {
            'uri': MONGO_URI,
            'name': DB_NAME,
            'use_fallback': USE_DB_FALLBACK
        },
        'upload': {
            'max_size': MAX_FILE_SIZE,
            'folder': UPLOAD_FOLDER,
            'model_folder': MODEL_FOLDER
        },
        'cloudinary': {
            'cloud_name': CLOUDINARY_CLOUD_NAME,
            'folder': CLOUDINARY_FOLDER
        }
    }

