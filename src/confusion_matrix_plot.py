import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models", "mobilenet_drowsiness.keras")
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_DIR, "data", "test"),
    image_size=(224,224),
    batch_size=32,
    shuffle=False
)

preds = model.predict(test_ds)

y_pred = np.argmax(preds, axis=1)
y_true = np.concatenate([y.numpy() for _, y in test_ds])

cm = confusion_matrix(y_true, y_pred)

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")
)

plt.show()