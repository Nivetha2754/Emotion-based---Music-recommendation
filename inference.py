import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import webbrowser
from urllib.parse import quote

# Load the trained model
model = load_model("model.h5")

# Load the labels
labels = np.load("labels.npy", allow_pickle=True)

# Define emotion-to-mood mapping
emotion_to_mood = {
    "Happy": "upbeat",
    "Sad": "calm",
    "Angry": "energetic",
    "Relaxed": "soothing"
}

# Function to open a playlist on YouTube
def open_playlist(emotion):
    mood = emotion_to_mood.get(emotion, "upbeat")  # Default to 'upbeat' if emotion not found
    search_query = f"{language} {artist} {mood} playlist"
    url = f"https://www.youtube.com/results?search_query={quote(search_query)}"
    webbrowser.open(url)
    print(f"Opening playlist for: {search_query}")

# User preferences
language = input("Enter your preferred language (e.g., English, Hindi): ").strip()
artist = input("Enter your favorite artist (e.g., Ed Sheeran, Arijit Singh): ").strip()

# Initialize Mediapipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Starting emotion detection...")

while True:
    lst = []
    ret, frm = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Extract face landmarks
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)
    else:
        lst.extend([0.0] * (468 * 2))  # Zero padding if no face landmarks

    # Extract left hand landmarks
    if res.left_hand_landmarks:
        for i in res.left_hand_landmarks.landmark:
            lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
            lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
    else:
        lst.extend([0.0] * 42)  # Zero padding if no left hand landmarks

    # Extract right hand landmarks
    if res.right_hand_landmarks:
        for i in res.right_hand_landmarks.landmark:
            lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
            lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
    else:
        lst.extend([0.0] * 42)  # Zero padding if no right hand landmarks

    # Prepare input for the model
    lst = np.array(lst).reshape(1, -1)
    prediction = np.argmax(model.predict(lst))
    detected_emotion = labels[prediction]  # Use labels as a list

    # Display detected emotion
    print(f"Detected Emotion: {detected_emotion}")
    cv2.putText(frm, f"Emotion: {detected_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the video feed
    cv2.imshow("Emotion Detection", frm)

    # Stop detecting after the first emotion is detected
    open_playlist(detected_emotion)
    break

# Release resources
cap.release()
cv2.destroyAllWindows()


