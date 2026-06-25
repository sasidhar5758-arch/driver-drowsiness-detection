import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_DIR = os.path.join(BASE_DIR, "data", "test")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenet_drowsiness.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate(
    [y.numpy() for _, y in test_ds]
)

print("\nClassification Report\n")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_true, y_pred))