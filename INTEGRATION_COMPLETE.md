# SignSight Backend - Complete System Integration Guide

## 📋 Project Overview

SignSight Backend is now a fully integrated system combining three major components:

1. **Audio-to-Sign API** - Converts audio/video to Tamil Sign Language predictions
2. **Mentor Dashboard API** - Manages mentors, students, and learning attempts
3. **Unified Entry Point** - Both APIs served on a single port via `main.py`

---

## ✅ System Status

**All integration complete!** ✓

- ✓ All 3 branches merged
- ✓ All dependencies installed and verified
- ✓ Type checking completed
- ✓ Configuration centralized
- ✓ Both APIs registered and working together
- ✓ Database (MongoDB) configured with fallback support
- ✓ ML models loaded and ready
- ✓ File uploads and processing configured

---

## 🚀 Quick Start

### 1. Start the Server

```bash
cd "/Users/farsithfawzer/Desktop/Farsith AudioToSign/SignSight-Backend"
python3 main.py
```

Server will start at: `http://localhost:5080`

### 2. Alternative: Using Gunicorn (Production)

```bash
gunicorn -w 4 -b 0.0.0.0:5080 main:app --timeout 120
```

### 3. Or Use the Startup Script

```bash
chmod +x start.sh
./start.sh
```

---

## 📚 API Endpoints

### Health Check
- **GET** `/` - Service status

### Admin
- **GET** `/api/admin/status` - System status and configuration

### Audio-to-Sign API
- **POST** `/api/audio-to-sign/upload-audio` - Upload audio file → sign prediction
- **POST** `/api/audio-to-sign/upload-video` - Upload video → extract audio → sign prediction
- **POST** `/api/audio-to-sign/text-to-signs` - Direct text → sign lookup
- **GET** `/api/audio-to-sign/get-sign-image/<sign_name>` - Get sign GIF/image

### Mentor Dashboard API
- **GET** `/api/<mentorEmail>/dashboard/users` - List students for a mentor
- **GET** `/api/dashboard/users/<user_id>/summary` - Student performance summary
- **GET** `/api/dashboard/users/<user_id>/attempts` - Student attempt history
- **GET** `/api/dashboard/overview` - Global statistics
- **POST** `/api/mentors` - Create new mentor
- **GET** `/api/admin/mentors` - List all mentors
- **GET** `/api/admin/students` - List all students
- **POST** `/api/admin/save-mentor-users` - Assign students to mentor
- **POST** `/api/students` - Create new student
- **GET** `/api/students/by-username/<username>` - Get student by username

---

## 📁 Project Structure

```
SignSight-Backend/
├── main.py                      # ⭐ Unified entry point
├── config.py                    # ⭐ Centralized configuration
├── app.py                        # Original audio-to-sign Flask app
├── mentor_backend.py             # Mentor dashboard Flask app
├── combined_app.py               # (Legacy - can be removed)
├── requirements.txt              # Dependencies
├── .env                          # Environment variables
├── verify_system.py              # System verification script
├── start.sh                      # Startup script
│
├── routes/
│   ├── __init__.py
│   └── audio_to_sign_routes.py  # Audio-to-sign route handlers
│
├── services/
│   └── audio_to_sign_service.py # ML model and processing
│
├── models/
│   ├── audio_to_sign_best.h5           # Sign prediction model
│   ├── audio_to_sign_norm_stats.json   # Normalization statistics
│   ├── emotion_model_mobilenet.h5      # Emotion detection model
│   └── emotion_model_mobilenetOld.h5   # Alt emotion model
│
├── static/
│   └── sign_images/             # Sign GIF/image files
│
├── uploads/                     # Temporary file uploads
├── reports/                     # Generated reports
└── mock_uploads/                # Test files
```

---

## 🔧 Configuration

All configuration is in `.env`:

```env
# Flask
PORT=5080
FLASK_ENV=development

# MongoDB
MONGO_URI=mongodb+srv://...
USE_DB_FALLBACK=true

# Cloudinary
CLOUDINARY_CLOUD_NAME=dgnpxg3jd
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
CLOUDINARY_FOLDER=Sign_Sight_Assets
```

Or programmatically in `config.py` with typed configuration.

---

## 📦 Dependencies

All core dependencies are installed:

| Package | Purpose |
|---------|---------|
| Flask 3.0 | Web framework |
| flask-cors | CORS support |
| pymongo 4.6+ | MongoDB driver |
| tensorflow 2.15+ | ML models |
| numpy | Numerical computing |
| librosa | Audio processing |
| cloudinary | Image/video hosting |
| gunicorn | WSGI server |

See `requirements.txt` for complete list.

---

## 🧪 Verification

Run the system verification script:

```bash
python3 verify_system.py
```

This checks:
- ✓ Python version compatibility
- ✓ All dependencies installed
- ✓ File structure and directories
- ✓ Environment configuration
- ✓ App imports and routes
- ✓ Database configuration
- ✓ ML models present and valid

---

## 🎯 How It Works

### Request Flow

1. **Client Request** → Hits `main.py` unified Flask app
2. **Route Matching** → Flask router matches to registered blueprint
3. **Audio-to-Sign Path** → `/api/audio-to-sign/*` → audio_to_sign service
4. **Mentor Path** → `/api/*` (mentors, students, etc.) → mentor_backend routes
5. **Response** → JSON response with CORS headers

### Audio Processing Pipeline

```
Audio File → FFmpeg Conversion → MFCC Features → Normalization → 
LSTM Model → Prediction → Cloudinary Lookup → Response
```

### Database Fallback

- Primary: MongoDB Atlas (cloud)
- Fallback: In-memory collections (if MongoDB unavailable)
- Ensures API works during development even without DB connection

---

## 🔍 Key Features

### Type Safety
- Python type hints throughout codebase
- Type checking support ready for mypy/pyright

### Error Handling
- Comprehensive error handlers (404, 500, 413)
- Graceful database fallback
- Optional dependency handling (Cloudinary, opencv)

### Logging
- Structured logging with timestamps
- Separate service initialization logging
- Debug headers for troubleshooting

### CORS Support
- Pre-configured for localhost (3000, 5173, 4200)
- Easy to extend for production domains

### File Management
- Max 16MB file uploads
- Automatic directory creation
- Temporary file cleanup
- Cloudinary integration for persistent storage

---

## 📝 Development Notes

### Adding New Routes

1. Create handler function in appropriate service/routes file
2. Register with Flask app or blueprint
3. Add type hints
4. Update API documentation

### Running Tests

```bash
# Type checking
python3 -m mypy main.py mentor_backend.py app.py --ignore-missing-imports

# System verification
python3 verify_system.py

# Manual API test (after server starts)
curl http://localhost:5080/
```

### Environment Variables

- `PORT` - Server port (default: 5080)
- `FLASK_ENV` - development/production
- `MONGO_URI` - MongoDB connection string
- `USE_DB_FALLBACK` - Enable in-memory DB fallback
- `CLOUDINARY_*` - Cloudinary API credentials

---

## 🐛 Troubleshooting

### Issue: Port already in use
```bash
# Find process using port 5080
lsof -i :5080

# Kill process
kill -9 <PID>

# Or use different port
PORT=5081 python3 main.py
```

### Issue: MongoDB connection fails
- Check `MONGO_URI` in `.env`
- Verify network connectivity
- Set `USE_DB_FALLBACK=true` for development
- Check IP whitelist in MongoDB Atlas

### Issue: Model loading fails
- Ensure model files exist in `models/` directory
- Check TensorFlow installation: `python3 -c "import tensorflow; print(tensorflow.__version__)"`
- Verify normalization stats JSON is valid

### Issue: Cloudinary upload fails
- Check API credentials in `.env`
- Verify folder permissions on Cloudinary
- Service will fallback gracefully if upload fails

---

## 📊 Performance Optimization

For production deployment:

```bash
# Use Gunicorn with multiple workers
gunicorn -w 4 -b 0.0.0.0:5080 main:app --timeout 120

# Load models once on startup (already implemented)
# Use Redis for session management (optional)
# Enable response gzip compression (add to Flask config)
```

---

## 🚢 Deployment Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Use `gunicorn` or similar WSGI server
- [ ] Configure reverse proxy (nginx)
- [ ] Set up SSL/TLS certificates
- [ ] Enable rate limiting
- [ ] Set proper CORS origins (not `*`)
- [ ] Use environment-based secrets (not in .env)
- [ ] Enable database connection pooling
- [ ] Set up monitoring/logging
- [ ] Configure backup strategy for models

---

## 📞 Support

### Key Files for Reference
- `main.py` - Unified app entry point and route registration
- `config.py` - All configuration constants
- `services/audio_to_sign_service.py` - ML model integration
- `routes/audio_to_sign_routes.py` - Audio API endpoints
- `mentor_backend.py` - Mentor dashboard endpoints

### Debugging
- Enable Flask debug mode: `FLASK_ENV=development`
- Check logs in console output
- Use `verify_system.py` to diagnose issues
- Test endpoints with curl/Postman

---

## ✨ What Was Done

### Integration Tasks Completed

1. ✅ **Branch Merge**
   - Combined audio-to-sign API
   - Combined mentor dashboard API  
   - Combined database configuration

2. ✅ **Dependency Management**
   - Created comprehensive `requirements.txt`
   - Added pymongo for MongoDB
   - Added gunicorn for production
   - Fixed Python 3.13 compatibility

3. ✅ **Code Quality**
   - Fixed type hints and imports
   - Fixed datetime usage (timezone-aware)
   - Fixed code indentation issues
   - Added comprehensive logging

4. ✅ **Configuration**
   - Created `config.py` for centralized settings
   - Updated `.env` with all variables
   - Made configuration environment-based

5. ✅ **Unified Entry Point**
   - Created `main.py` as single entry point
   - Registered both APIs as blueprints
   - Implemented error handlers
   - Added health check endpoints

6. ✅ **Verification & Testing**
   - Created `verify_system.py` for system check
   - Verified all imports work
   - Verified all routes registered
   - Verified dependencies installed
   - Verified models present

7. ✅ **Documentation**
   - This comprehensive guide
   - Inline code comments
   - API endpoint documentation
   - Troubleshooting guide

---

## 🎉 System Ready!

The SignSight Backend is now fully integrated and ready to run as a complete system.

**To get started:**
```bash
cd "/Users/farsithfawzer/Desktop/Farsith AudioToSign/SignSight-Backend"
python3 main.py
```

**Then visit:** http://localhost:5080

Enjoy! 🚀

