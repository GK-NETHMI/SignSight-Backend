from flask import Flask, request, jsonify
import random
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'mock_uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

SIGN_LANGUAGE_VOCABULARY = {
    'family': ['mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'aunt', 'uncle', 'cousin'],
    'alphabet': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
    'numbers': ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'twenty', 'thirty', 'hundred'],
    'objects': ['table', 'chair', 'book', 'pen', 'door', 'window', 'computer', 'phone', 'bag', 'cup', 'bottle', 'car', 'house', 'tree', 'flower'],
    'actions': ['eat', 'drink', 'sleep', 'run', 'walk', 'sit', 'stand', 'write', 'read', 'play', 'work', 'study', 'dance', 'sing', 'jump']
}

ALL_WORDS = []
for words in SIGN_LANGUAGE_VOCABULARY.values():
    ALL_WORDS.extend(words)

@app.route('/process-video', methods=['POST'])
def process_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        file_size = os.path.getsize(filepath)
        
        processing_time = min(file_size / (1024 * 1024) * 0.5, 5)
        
        response_type = random.choice(['correct', 'incorrect', 'partial'])
        
        if response_type == 'correct':
            detected_word = random.choice(ALL_WORDS)
            confidence = random.uniform(0.85, 0.99)
        elif response_type == 'incorrect':
            detected_word = random.choice(ALL_WORDS)
            confidence = random.uniform(0.60, 0.84)
        else:
            detected_word = random.choice(ALL_WORDS)
            confidence = random.uniform(0.50, 0.75)
        
        try:
            os.remove(filepath)
        except:
            pass
        
        response_data = {
            'success': True,
            'text': detected_word,
            'answer': detected_word,
            'confidence': round(confidence, 2),
            'processing_time': round(processing_time, 2),
            'metadata': {
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'filename': filename,
                'detected_signs': [detected_word],
                'alternative_interpretations': [
                    {'word': random.choice(ALL_WORDS), 'confidence': round(random.uniform(0.3, 0.6), 2)}
                    for _ in range(2)
                ]
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'text': 'error', 'answer': 'error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
