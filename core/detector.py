import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def detect_emotion(frame):
    # ✅ Load model only when needed (lazy loading)
    model_path = os.path.join(os.path.dirname(__file__), "emotion_model", "model.h5")
    model = load_model(model_path)
    print("111111111!", model_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []

    for x, y, w, h in faces:
        roi = gray[y : y + h, x : x + w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        prediction = model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        results.append(emotion)
    return results
