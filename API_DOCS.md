# Flask Backend API Documentation

## Quick Start

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

The server will start at `http://localhost:5000`

### 4. Test the API
```bash
python test_api.py
```

## API Endpoints

### Health Check
**GET** `/health`

Check if the server is running.

**Response:**
```json
{
  "status": "healthy",
  "message": "Flask backend is running"
}
```

---

### Audio to Sign Language
**POST** `/audio-to-sign`

Convert audio file to Tamil Sign Language.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `file` field containing the audio file

**Supported Audio Formats:**
- MP3 (.mp3)
- WAV (.wav)
- OGG (.ogg)
- M4A (.m4a)
- FLAC (.flac)

**File Size Limit:** 16MB

**Success Response (200):**
```json
{
  "sign_output": "SIGN_SEQUENCE_001",
  "filename": "audio_file.mp3"
}
```

**Error Responses:**
- `400` - No file provided, empty filename, or invalid file type
- `413` - File too large (>16MB)

---

### Text to Sign Language
**POST** `/text-to-sign`

Convert text file to Tamil Sign Language.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `file` field containing the text file

**Supported Text Formats:**
- TXT (.txt)
- PDF (.pdf)
- DOC (.doc)
- DOCX (.docx)

**File Size Limit:** 16MB

**Success Response (200):**
```json
{
  "sign_output": "SIGN_SEQUENCE_001",
  "filename": "text_file.txt"
}
```

**Error Responses:**
- `400` - No file provided, empty filename, or invalid file type
- `413` - File too large (>16MB)

---

## Security Features

✅ **File Type Validation** - Only allowed file extensions are accepted
✅ **Secure Filename Handling** - Prevents path traversal attacks using `secure_filename()`
✅ **File Size Limits** - Maximum 16MB upload size
✅ **CORS Enabled** - Supports cross-origin requests for frontend integration
✅ **Secret Key Configuration** - Uses environment variable or default for development

## Configuration

Create a `.env` file (see `.env.example`):
```bash
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
FLASK_DEBUG=True
MAX_CONTENT_LENGTH=16777216
```

## Project Structure
```
flask-backend/
├── app.py                 # Main Flask application
├── dummy_responses.py     # Dummy response functions
├── requirements.txt       # Python dependencies
├── test_api.py           # API test suite
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── uploads/              # Upload directories
│   ├── audio/           # Audio file uploads
│   └── text/            # Text file uploads
├── static/              # Static files
└── templates/           # HTML templates
```

## Development

### Running in Development Mode
```bash
python app.py
```

### Running in Production
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Testing
```bash
# Make sure the server is running first
python test_api.py
```

## Error Handling

The API includes comprehensive error handlers:

- **404** - Endpoint not found
- **413** - File too large
- **500** - Internal server error

All errors return JSON responses with an `error` field.

## Next Steps

1. **Integration**: Connect your ML models for actual sign language conversion
2. **Authentication**: Add user authentication if needed
3. **Database**: Implement database for storing conversion history
4. **Logging**: Add detailed logging for debugging and monitoring
5. **Testing**: Expand test coverage with unit and integration tests
6. **Deployment**: Deploy to production server (AWS, Heroku, etc.)

## Support

For issues and questions, please refer to the main project README.md

