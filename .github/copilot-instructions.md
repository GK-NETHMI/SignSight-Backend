# AudioToSign Flask Backend - AI Coding Agent Instructions

## Project Overview
This is a minimal Flask web application for the AudioToSign project. The backend serves as a web interface foundation for audio-to-sign-language conversion features.

## Architecture

### Core Components
- **`app.py`**: Single-file Flask application entry point
- **`templates/`**: Jinja2 HTML templates (currently `index.html`)
- **`static/`**: Client-side assets (CSS, future JS)
- **`venv/`**: Python 3.9 virtual environment (excluded from version control)

### Route Structure
- `/page` - Main application page rendering `index.html`
- No API endpoints currently defined

## Development Workflow

### Environment Setup
```bash
# Activate virtual environment (required before any Python operations)
source venv/bin/activate

# Run development server
python app.py
```

### Python Environment
- **Python Version**: 3.9
- **Virtual Environment**: Always activate `venv/` before running Python commands
- **Dependencies**: Flask (managed via venv, no `requirements.txt` present)

## Conventions & Patterns

### Flask Application Structure
- Single Flask app instance declared in `app.py` as `app = Flask(__name__)`
- Route handlers use decorator pattern: `@app.route("/path")`
- Templates rendered via `render_template()` function from Flask

### File Organization
- **Templates**: HTML files go in `templates/` directory
- **Static Assets**: CSS/JS files go in `static/` directory
- **No Blueprint Architecture**: All routes currently in single `app.py` file

## Project-Specific Notes

### Current State
- **Minimal Implementation**: Project is in early stages with basic Flask scaffolding
- **No Database**: No database layer or models implemented yet
- **No API**: No REST/JSON endpoints; currently only renders HTML
- **Empty CSS**: `static/style.css` exists but contains no styles

### Likely Future Expansions
Given the "AudioToSign" project name, expect:
- Audio file upload/processing endpoints
- Integration with sign language translation services
- Video/animation generation for sign language output
- WebSocket or streaming for real-time processing

## Key Files
- **`app.py`**: Main application logic and route definitions
- **`templates/index.html`**: Primary user interface template

## Common Tasks

### Adding New Routes
Add route decorators and handlers in `app.py`:
```python
@app.route("/new-path")
def new_handler():
    return render_template("template.html")
```

### Adding Static Assets
Link CSS in templates using Flask's url_for:
```html
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
```

### Running the Application
Always ensure venv is activated, then `python app.py` (runs on default Flask dev server)
