"""AdSwapAI R&D, 2025-03-03: one-line ultralytics YOLOv5n prediction on a test image."""

from ultralytics import YOLO

# Load the YOLOv5 nano model
model = YOLO("yolov5n.pt")  # Small model; larger variants: yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt

# Test the model (optional)
model.predict("test.jpg", save=True)
