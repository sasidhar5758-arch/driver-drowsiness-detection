import cv2
import tensorflow as tf
import numpy as np
import os
import threading
from playsound import playsound

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenet_drowsiness.keras"
)

ALARM_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "alarm.wav"
)

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

# =========================
# CLASS LABELS
# =========================

classes = [
    "Closed",
    "Open",
    "no_yawn",
    "yawn"
]

# =========================
# FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =========================
# ALARM
# =========================

alarm_running = False


def play_alarm():
    global alarm_running

    if not alarm_running:
        alarm_running = True

        try:
            playsound(ALARM_PATH)
        except Exception as e:
            print("Alarm Error:", e)

        alarm_running = False


# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("Cannot open webcam!")
    exit()

print("Webcam started...")

# =========================
# DROWSINESS VARIABLES
# =========================

closed_counter = 0

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[
            y:y+h,
            x:x+w
        ]

        try:

            face = cv2.resize(
                face,
                (224, 224)
            )

            face = face.astype(
                "float32"
            ) / 255.0

            face = np.expand_dims(
                face,
                axis=0
            )

            prediction = model.predict(
                face,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction
            )

            label = classes[
                predicted_index
            ]

            confidence = float(
                np.max(prediction)
            )

            # =====================
            # DROWSINESS LOGIC
            # =====================

            if label == "Closed":

                closed_counter += 1

            else:

                closed_counter = 0

            if closed_counter >= 20:

                threading.Thread(
                    target=play_alarm,
                    daemon=True
                ).start()

                status = "DROWSY"

            else:

                status = "ALERT"

            # =====================
            # DRAW
            # =====================

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{label} : {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                f"Closed Count: {closed_counter}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        except Exception as e:

            print("Prediction Error:", e)

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()