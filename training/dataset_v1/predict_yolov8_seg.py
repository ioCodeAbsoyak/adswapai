"""AdSwapAI R&D, 2025-04-03: quick visual check of the trained YOLOv8-seg billboard model on a clip."""
import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/custom_seg_model4/weights/best.pt"   # output of train_yolov8_seg.py
VIDEO_PATH = "data/adVideo2.mp4"                        # sample clip, see docs/assets.md

# Load the trained weights
model = YOLO(MODEL_PATH)

# Predict as a stream
results = model.predict(source=VIDEO_PATH, imgsz=640, conf=0.5, stream=True)

# Show every 30th frame
frame_count = 0
for r in results:
    frame_count += 1
    if frame_count % 30 == 0:
        img = r.plot()  # frame with the predicted masks drawn on it
        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
