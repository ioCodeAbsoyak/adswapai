"""AdSwapAI R&D, 2025-04-03: train the single-class YOLOv8s-seg billboard model on the Roboflow export (dataset v1).

Produces runs/custom_seg_model*/weights/best.pt, the model used by the April 2025 tracking experiments.
"""
from ultralytics import YOLO
import multiprocessing

def train_model():
    model = YOLO("yolov8s-seg.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs",
        name="custom_seg_model"
    )
    print("Training complete!")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed for frozen executables on Windows
    train_model()
