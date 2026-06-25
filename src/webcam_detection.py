import cv2
import tensorflow as tf
import numpy as np
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenet_drowsiness.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    "Closed",
    "Open",
    "no_yawn",
    "yawn"
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (224, 224)
        )

        face = face / 255.0

        face = np.expand_dims(
            face,
            axis=0
        )

        pred = model.predict(
            face,
            verbose=0
        )

        label = classes[
            np.argmax(pred)
        ]

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()