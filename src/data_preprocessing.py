# data_preprocessing.py

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError(
        "TensorFlow is not installed in the current Python interpreter.\n"
        "Run:\n"
        "py -3.11 -m pip install tensorflow==2.16.1"
    ) from e

import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")

print("TensorFlow Version:", tf.__version__)
print("Train Path:", TRAIN_DIR)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataset Loaded Successfully")
print("Classes:", train_ds.class_names)

for images, labels in train_ds.take(1):
    print("Image Batch Shape:", images.shape)
    print("Label Batch Shape:", labels.shape)