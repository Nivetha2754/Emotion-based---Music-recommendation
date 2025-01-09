# Emotion-based---Music-recommendation
This project is an Emotion-Based Music Recommender System that uses real-time emotion detection to suggest personalized music playlists. By leveraging a trained neural network model and Mediapipe's facial and hand landmark detection, the system identifies the user's emotion and recommends a music 
# Emotion-Based Music Recommender

## Overview
This project is a **real-time emotion detection system** that recommends music based on the detected emotion. It uses Mediapipe for facial and hand landmark detection, along with a trained neural network to classify emotions and suggest personalized YouTube playlists.

## Features
- Detects emotions like **Happy**, **Sad**, **Angry**, and **Relaxed** in real-time.
- Recommends music playlists tailored to the detected emotion.
- Allows users to specify their **preferred language** and **favorite artist**.
- Automatically opens YouTube with a mood-matching playlist.

## Technologies Used
- **Python**
- **TensorFlow/Keras**: Emotion classification model.
- **Mediapipe**: Facial and hand landmark detection.
- **OpenCV**: Webcam integration and video feed.
- **Web Browser**: Opens YouTube playlists.

## How It Works
1. **Data Collection**: Use `data_collection.py` to collect emotion data.
2. **Model Training**: Use `data_training.py` to train a neural network.
3. **Emotion Detection & Music Recommendation**: Run `inference.py` to detect emotion and recommend a playlist.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Nivetha2754/Emotion based- Music recommendation.git



