# Driver Drowsiness Detection Using MobileNetV2

## Overview

This project implements a real-time Driver Drowsiness Detection System using Deep Learning, TensorFlow, MobileNetV2, and OpenCV.

The system detects:

* Open Eyes
* Closed Eyes
* Yawning
* No Yawning

and alerts the driver when signs of drowsiness are detected.

## Features

* Real-time webcam monitoring
* Face detection using OpenCV
* MobileNetV2-based image classification
* Drowsiness alert system
* Alarm notification
* High accuracy detection

## Technologies Used

* Python 3.11
* TensorFlow 2.16.1
* Keras
* OpenCV
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn

## Project Structure

```text
driver_drowsiness_detection/
├── data/
├── models/
├── outputs/
├── src/
├── requirements.txt
└── README.md
```

## Training

```bash
py -3.11 src/train_mobilenet.py
```

## Evaluation

```bash
py -3.11 src/evaluate.py
```

## Real-Time Detection

```bash
py -3.11 src/advanced_webcam_detection.py
```

## Results

* Training Accuracy: 98.0%
* Validation Accuracy: 98.9%
* Validation Loss: 0.0387

## Model

MobileNetV2

## Author

Sasidhar Y
