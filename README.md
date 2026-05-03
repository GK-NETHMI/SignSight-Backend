# SignSight Backend

## Run the server

```bash
cd "/Users/farsithfawzer/Desktop/Farsith AudioToSign/SignSight-Backend"
conda run -n base python app.py
```

> **Important:** Always use `conda run -n base python` — this is the only Python environment that has TensorFlow 2.20 with working `tf.keras`, which is required to load the model.

## API — Base URL: `http://localhost:5080`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/audio-to-sign/upload-audio` | Upload `.wav` file → sign prediction |
| POST | `/api/audio-to-sign/upload-video` | Upload video → sign prediction |
| POST | `/api/audio-to-sign/text-to-signs` | JSON `{"text":"..."}` → sign lookup |
| GET  | `/api/audio-to-sign/get-sign-image/<sign_name>` | Returns sign GIF/image |

## Model

- File: `models/audio_to_sign_best.h5`
- Norm stats: `models/audio_to_sign_norm_stats.json`
- Input: MFCC (40 coefficients + deltas + delta-deltas = 120 features), padded to 128 timesteps
- Output: 21 Tamil sign classes: amma, anbalippu, apple, arambam, aruvi, illam, kaalai, kadal, kattadam, keylvi, master, mownam, mudivu, nandri, neram, nimmadhi, phone, samayal, thambi, udhavi, urakkam

## Sign images

Place GIF/PNG files in `static/sign_images/` named after each class:
```
static/sign_images/nandri.gif
static/sign_images/amma.gif
...
```
 - Audio/Video to Tamil Sign Language

**Audio & Video to TSL Conversion Module**

This repository contains the backend API for the Audio/Video to Tamil Sign Language conversion component of the Sign Sight project.

## 🚩 About This Module

This module converts Tamil audio or video files into TSL (Tamil Sign Language) sign demonstrations:
- Processes audio/video files to extract Tamil speech
- Converts speech to text using speech recognition
- Uses a trained ML model to predict corresponding sign gestures
- Serves sign language images/GIFs for visualization

## 🎯 Core Features

1. **Audio Processing**: Upload audio files (mp3, wav, etc.) and convert Tamil speech to text
2. **Video Processing**: Extract audio from videos and process similarly
3. **ML Model Integration**: Predict sign language from Tamil text using your trained model
4. **Sign Image Retrieval**: Serve sign language images/GIFs based on model predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Your trained ML model for Tamil sign language

### Installation

```bash
# Navigate to the repository
cd SignSight-Backend

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads models static/sign_images

# Copy environment file
cp .env.example .env
```

### Add Your ML Model

1. Place your trained model in the `models/` directory
2. Update `services/audio_to_sign_service.py` to load your model (see API_DOCUMENTATION.md)
3. Add sign language images to `static/sign_images/`

### Run the Server

```bash
python app.py
```

Server will start at `http://localhost:5000`

## 📚 API Endpoints

### 1. Upload Audio
```
POST /api/audio-to-sign/upload-audio
```
Upload audio file and get sign language predictions

### 2. Upload Video
```
POST /api/audio-to-sign/upload-video
```
Upload video file and get sign language predictions

### 3. Text to Signs
```
POST /api/audio-to-sign/text-to-signs
```
Convert Tamil text directly to sign language

### 4. Get Sign Image
```
GET /api/audio-to-sign/get-sign-image/{sign_name}
```
Retrieve sign language image/gif

## 🔧 Configuration

The backend supports CORS for these frontend origins:
- http://localhost:3000 (React)
- http://localhost:5173 (Vite)
- http://localhost:4200 (Angular)

## 📖 Full Documentation

See [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) for:
- Complete API reference
- ML model integration guide
- Frontend connection examples (TypeScript)
- Error handling
- Testing examples

## 🧪 Testing

```bash
python test_api.py
```

## 📝 What You Need to Replace

1. **ML Model**: Place your trained model in `models/` and update loading logic
2. **Prediction Logic**: Update `_predict_sign_from_model()` method with your model's prediction code
3. **Sign Images**: Add Tamil sign language images/GIFs to `static/sign_images/`

See API_DOCUMENTATION.md for detailed instructions.
