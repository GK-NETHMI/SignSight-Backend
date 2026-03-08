## 💡 Emotion & Context-Aware Tamil Sign Language System – Child Module

### 👶 Purpose
This component helps deaf or hearing-impaired children (ages 3–10) by analyzing their **facial emotional reactions** while watching friendly cartoon videos. It provides real-time emotional insights to guardians for better care and learning support.

---

### 🔍 Key Features
- Detects emotions like **Happy**, **Sad**, **Angry**, **Fear**, and **Neutral** from webcam video.
- Uses **MediaPipe** for face detection and **MobileNetV2** for emotion classification.
- Sends a simple feedback report with emotion scores and parenting tips to the guardian via email.

---

### ⚙️ Technologies Used
| Area            | Tech |
|------------------|------|
| Model Architecture | MobileNetV2 (TensorFlow/Keras) |
| Video Input       | OpenCV + MediaPipe |
| Frontend Stack    | React + Vite + Tailwind CSS |
| Emotion Model     | `emotion_model_mobilenet.h5` |
| Backend Server    | Flask (Python) |
| Visual Feedback   | Emotion bar chart, bounding box, live FPS |
| Email Reporting   | SMTP / Flask-Mail |

---

### 🧪 Emotion Flow
1. Child watches 3 cartoon videos (Happy, Sad, Angry scenes).
2. Webcam records facial reactions.
3. Model predicts emotion scores in real time.
4. Scores are summarized and sent to the guardian with suggestions.

---

### 📦 Model Output
- Top emotion label (e.g., Happy)
- Percentage confidence for each emotion
- Text feedback for the guardian

---

### 🛡️ Novelty
- First to combine **emotion detection + Tamil Sign Language system** for young children.
- Provides **real-time emotional guidance** to parents.
- Supports **non-verbal emotional monitoring** in educational environments.

---

### ▶️ How to Run

#### 📦 Backend
```bash
cd backend
python app.py
```

#### 💻 Frontend
```bash
cd frontend
npm install
npm run dev
```

---

### ✅ Output Example
- Emotion Summary Card:
  - 😀 Happy – 85% → “Child feels comfortable in fun scenes”
  - 😨 Fear – 78% → “Child was scared during fighting scenes”
- Recommendations:
  - Avoid violent scenes
  - Use calm cartoon content
  - Monitor emotional triggers
