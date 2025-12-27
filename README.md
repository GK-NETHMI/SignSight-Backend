# Sign Sight (25-26J-404)

**Empowering Accessibility: A Tamil Sign Language (TSL) Recognition with Learning and Emotion Detection System Using Machine Learning**

This repository contains the project "Sign Sight," a multi-functional platform developed for the SLIIT Faculty of Computing. It is designed to bridge communication gaps for the hearing-impaired community in Sri Lanka.

## 🚩 About The Project

In Sri Lanka, an estimated 390,000 hearing-impaired citizens (approximately 1.7% of the population) use Tamil Sign Language to communicate. However, existing digital tools primarily support English Sign Language and often lack real-time accuracy, interactivity, and essential features like audio-to-sign conversion or vocal training.

Sign Sight is a multi-functional platform built to address this gap. It integrates learning, real-time detection, speech-to-sign conversion, and emotion detection training to create a single, holistic solution for education, communication, and rehabilitation.

## ✨ Key Features

The platform is composed of four main modules:

1.  **Interactive TSL Learning**

    - Provides structured, multi-level lessons for Tamil Sign Language, covering the alphabet, words, and phrase construction.
    - Includes an TSL Dictionary for quick lookups.
    - Features an evaluation system with randomized MCQ and Intermediate level assessments (~Tier based questions) to test comprehension.

2.  **Dynamic Real-Time Sign Detection**

    - Uses a webcam to capture user gestures in real-time.
    - Extracts key points using MediaPipe Holistic.
    - Provides instant sign language predictions using an LSTM-based machine learning model.

3.  **Audio & Video to TSL Conversion**

    - Converts Tamil audio or video files into TSL sign demonstrations.
    - The system processes audio, converts it to text, and then uses an ML model to predict and display the corresponding sign gestures.

4.  **Emotion Alert System**
    - Specifically designed for deaf children aged 3–10, focusing on visual emotional understanding.
    - Detects the child’s facial expressions via webcam while watching cartoon videos, and performs real-time emotion analysis.
    - Uses a deep learning model to classify emotions and delivers feedback to parents via email, supporting Tamil Sign Language (TSL) learning context.

## ⚙️ System Architecture

The system operates with a user interacting with a web interface. Inputs (Webcam, Text, Audio) are sent to a central Backend API, which orchestrates four primary services and communicates with a SQL database.

- **Learning System:** Manages lessons, MCQs, and Text-to-Sign interpretation.
- **Audio to Sign Conversion:** Handles audio-to-text and text-to-sign image conversion.
- **Emotion Alert System:** Manages facial expression input, feature extraction, emotion classification, and accuracy assessment for deaf children aged 3–10.
- **Dynamic Sign Detection:** Processes webcam input through an LSTM model for Emotional gesture recognition.
