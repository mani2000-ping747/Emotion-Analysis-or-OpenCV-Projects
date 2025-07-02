import time
from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.core.files.storage import default_storage
from django.conf import settings
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model_path = os.path.join("core", "emotion_model", "model.h5")
model = load_model(model_path, compile=False)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def index(request):
    return render(request, "webcam.html")


def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield (b"--frame\r\nContent-Type: text/plain\r\n\r\nCamera not found\r\n")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(preds)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def video_feed(request):
    return StreamingHttpResponse(
        gen_frames(), content_type="multipart/x-mixed-replace; boundary=frame"
    )


def predict_emotion(request):
    if request.method == "POST" and request.FILES.get("image"):
        uploaded_file = request.FILES["image"]
        file_path = default_storage.save("uploaded.jpg", uploaded_file)

        img_path = default_storage.path(file_path)
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion = "No face detected"
        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi = roi_gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            emotion = emotion_labels[np.argmax(preds)]

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                image,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        result_path = os.path.join(settings.MEDIA_ROOT, "result.jpg")
        cv2.imwrite(result_path, image)

        return render(
            request,
            "result.html",
            {
                "emotion": emotion,
                "image_url": settings.MEDIA_URL + "result.jpg",
            },
        )

    return render(request, "webcam.html")


def predict_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        import time

        video_file = request.FILES["video"]
        file_path = default_storage.save("uploaded_video.mp4", video_file)
        video_path = default_storage.path(file_path)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return render(
                request, "video_result.html", {"error": "Unable to open video"}
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = os.path.join(settings.MEDIA_ROOT, "processed_video.avi")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for x, y, w, h in faces:
                roi_gray = gray[y : y + h, x : x + w]
                roi_gray = cv2.resize(roi_gray, (64, 64))
                roi = roi_gray.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)

                preds = model.predict(roi, verbose=0)[0]
                label = emotion_labels[np.argmax(preds)]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)

        return render(
            request,
            "video_result.html",
            {"video_url": settings.MEDIA_URL + "processed_video.avi"},
        )

    return render(request, "upload_video.html")
