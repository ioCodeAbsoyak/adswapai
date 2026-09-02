"""AdSwapAI R&D, 2025-04-03: YOLOv8s (COCO) + deep_sort_realtime benchmark with run statistics."""

import cv2
import numpy as np
import torch
import time
import warnings
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

VIDEO_PATH = "data/adVideo1.mp4"   # sample clip, see repo docs/assets.md

# Suppress the FutureWarnings from YOLOv5/PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress YOLO console output
os.environ["PYTHONIOENCODING"] = "utf-8"  # Needed for suppressing ultralytics prints
import sys
from io import StringIO

# Class to capture stdout and suppress it
class CaptureOutput:
    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self.mystdout = StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.old_stdout
    
    def get_output(self):
        return self.mystdout.getvalue()

# Path to your video
video_path = VIDEO_PATH

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Initialize YOLO model
print("Loading YOLO model...")
# Use YOLOv8 if installed, otherwise fall back to YOLOv5
try:
    from ultralytics import YOLO
    with CaptureOutput():  # Suppress model loading output
        model = YOLO('yolov8s.pt')  # Load YOLOv8 small model
    use_yolov8 = True
    print("Using YOLOv8")
except ImportError:
    # Fall back to YOLOv5
    with CaptureOutput():  # Suppress model loading output
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    use_yolov8 = False
    print("Using YOLOv5")

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=100,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True
)

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Performance tracking variables
start_time = time.time()
frame_count = 0
current_fps = 0
update_fps_every = 10  # Update FPS every 10 frames

# Tracking statistics
class_counts = defaultdict(int)
unique_tracks = set()
inference_times = []
process_times = []

# Define COCO class names for reference
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']

# Color map for visualization (each class gets a color)
color_map = {}

print(f"Starting processing of {total_frames} frames...")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Calculate FPS
    frame_count += 1
    frame_start_time = time.time()
    
    if frame_count % update_fps_every == 0:
        current_fps = update_fps_every / (frame_start_time - start_time)
        start_time = frame_start_time
    
    # Run object detection
    if use_yolov8:
        # YOLOv8 detection (suppress console output)
        with CaptureOutput():
            inference_start = time.time()
            results = model(frame, verbose=False)
            inference_end = time.time()
        inference_times.append(inference_end - inference_start)
        
        # Process detections
        detection_list = []
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Update class counts
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            class_counts[class_name] += 1
            
            # Create detection format for DeepSORT
            detection_list.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))
    else:
        # YOLOv5 detection
        with CaptureOutput():
            inference_start = time.time()
            results = model(frame, verbose=False)
            inference_end = time.time()
        inference_times.append(inference_end - inference_start)
        
        detections = results.xyxy[0].cpu().numpy()
        
        # Process detections
        detection_list = []
        for x1, y1, x2, y2, conf, cls_id in detections:
            if conf > 0.45:  # Confidence threshold
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Update class counts
                class_id = int(cls_id)
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                class_counts[class_name] += 1
                
                detection_list.append(([x, y, w, h], conf, int(cls_id)))
    
    # Update tracker
    tracks = tracker.update_tracks(detection_list, frame=frame)
    
    # Process and display tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        unique_tracks.add(track_id)
        ltrb = track.to_ltrb()  # [left, top, right, bottom]
        
        x1, y1, x2, y2 = map(int, ltrb)
        cls_id = track.get_det_class()
        
        # Generate consistent color for this track ID
        if track_id not in color_map:
            # Generate random color
            color_map[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        color = color_map[track_id]
        
        # Get class name
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"ID:{track_id} {class_name}"
        
        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Performance monitoring - record total frame processing time
    frame_end_time = time.time()
    process_times.append(frame_end_time - frame_start_time)
    
    # Draw FPS and progress on frame
    fps_text = f"FPS: {current_fps:.1f}"
    progress_text = f"Frame: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
    cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, progress_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Object Tracking", frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate and display statistics
total_time = sum(process_times)
avg_fps = frame_count / total_time if total_time > 0 else 0
avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
avg_process = sum(process_times) / len(process_times) if process_times else 0

print("\n===== TRACKING SUMMARY =====")
print(f"Total frames processed: {frame_count}")
print(f"Total unique objects tracked: {len(unique_tracks)}")
print("\nDetected objects by class:")
for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cls}: {count} detections")

print("\nPerformance statistics:")
print(f"  Average FPS: {avg_fps:.2f}")
print(f"  Average inference time: {avg_inference*1000:.2f} ms")
print(f"  Average frame processing time: {avg_process*1000:.2f} ms")
print("============================")