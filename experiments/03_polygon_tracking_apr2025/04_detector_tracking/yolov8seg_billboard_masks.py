"""AdSwapAI R&D, 2025-04-03: first use of the custom single-class YOLOv8s-seg "billboard"
model (best.pt); detected billboard masks shown with blur / color / pixelate effects."""

import cv2
import numpy as np
import torch
import time
import warnings
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# Suppress the FutureWarnings from YOLOv5/PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress YOLO console output
os.environ["PYTHONIOENCODING"] = "utf-8"
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

VIDEO_PATH = "data/adVideo2.mp4"   # sample clip, see repo docs/assets.md
MODEL_PATH = "best.pt"   # single-class YOLOv8s-seg "billboard" model, see docs/assets.md

# Path to your video
video_path = VIDEO_PATH

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Initialize YOLO model
print("Loading YOLO model...")
try:
    from ultralytics import YOLO
    with CaptureOutput():
        model = YOLO(MODEL_PATH)
    use_yolov8 = True
    print("Using YOLOv8 Segmantation")
except ImportError:
    with CaptureOutput():
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

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Performance tracking variables
start_time = time.time()
frame_count = 0
current_fps = 0
update_fps_every = 10

# Tracking statistics
class_counts = defaultdict(int)
unique_tracks = set()
inference_times = []
process_times = []

# COCO class names
class_names = ['billboard']

# Color map for visualization
color_map = {}

# Masking options - you can change these to adjust the effect
MASK_TYPE = "BLUR"  # Options: "BLUR", "COLOR", "PIXELATE"
BLUR_AMOUNT = 25    # Higher values = more blur
MASK_COLOR = (0, 0, 255)  # Red mask color (BGR)
PIXELATE_BLOCKS = 15  # Pixelation block size (larger = more pixelated)

print(f"Starting processing of {total_frames} frames...")
print(f"Masking billboards using {MASK_TYPE} method")

# Function to apply different masking effects
def apply_mask(frame, mask_info, mask_type):
    # Check if we have a segmentation mask or just a bounding box
    if isinstance(mask_info, tuple) and len(mask_info) == 2 and not isinstance(mask_info[0], (int, float)):
        # We have a segmentation mask with format (mask, bbox)
        segmentation_mask = True
        mask, bbox = mask_info
        x1, y1, x2, y2 = bbox
        x, y, w, h = x1, y1, x2-x1, y2-y1
    else:
        # We have a bounding box with format (x, y, w, h)
        segmentation_mask = False
        try:
            x, y, w, h = mask_info
        except ValueError:
            print(f"Error with mask_info: {mask_info}")
            return
    
    # Make sure the bbox is within frame boundaries
    x, y = max(0, x), max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    
    # Get the region of interest
    roi = frame[y:y+h, x:x+w]
    
    if roi.size == 0:
        return
    
    if mask_type == "COLOR" and segmentation_mask:
        # Apply color masking using segmentation mask
        # Create a color overlay
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colored_mask[:] = MASK_COLOR
        
        # Scale the mask to match the roi size
        mask_resized = cv2.resize(mask, (w, h))
        
        # Create 3-channel mask
        mask_3channel = np.stack([mask_resized]*3, axis=-1)
        
        # Blend color where mask is active
        alpha = 0.7  # Transparency
        # For segmentation masks, only apply color where mask > 0.5
        for c in range(3):  # For each color channel
            roi[:,:,c] = np.where(mask_resized > 0.5, 
                                  (alpha * colored_mask[:,:,c] + (1-alpha) * roi[:,:,c]).astype(np.uint8), 
                                  roi[:,:,c])
        
        frame[y:y+h, x:x+w] = roi
    elif mask_type == "COLOR":
        # Regular color masking for bounding box
        overlay = np.ones(roi.shape, dtype=np.uint8)
        overlay[:] = MASK_COLOR
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
        frame[y:y+h, x:x+w] = roi
    elif mask_type == "BLUR":
        # Apply Gaussian blur
        kernel_size = min(BLUR_AMOUNT, w - 1 if w % 2 == 0 else w, h - 1 if h % 2 == 0 else h)
        # Ensure kernel size is odd and at least 3
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size - 1)
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        frame[y:y+h, x:x+w] = blurred
    elif mask_type == "PIXELATE":
        # Pixelate the ROI
        h, w = roi.shape[:2]
        
        # Ensure minimum size for pixelation
        if w < 2 or h < 2:
            return
            
        # Calculate safe block size (prevent division by zero)
        block_w = max(1, w // PIXELATE_BLOCKS)
        block_h = max(1, h // PIXELATE_BLOCKS)
        
        # Reduce resolution
        if block_w >= 1 and block_h >= 1:
            temp = cv2.resize(roi, (max(1, w // PIXELATE_BLOCKS), max(1, h // PIXELATE_BLOCKS)),
                            interpolation=cv2.INTER_LINEAR)
            
            # Enlarge back to original size with nearest neighbor
            pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            
            frame[y:y+h, x:x+w] = pixelated

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
        with CaptureOutput():
            inference_start = time.time()
            results = model(frame, verbose=False)
            inference_end = time.time()
        inference_times.append(inference_end - inference_start)
        
        detection_list = []
        billboard_masks = []  # Clear the billboard masks list for this frame
        
        for r in results:
            boxes = r.boxes
            if hasattr(r, 'masks') and r.masks is not None:
                masks = r.masks.data
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                    class_counts[class_name] += 1
                    
                    detection_list.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))
                    
                    # For billboards, store the segmentation mask
                    if cls_id == 0:  # billboard class
                        try:
                            # Get the mask data - make sure indexing is correct
                            mask = masks[i].cpu().numpy()
                            billboard_masks.append((mask, (x1, y1, x2, y2)))
                        except Exception as e:
                            print(f"Mask error: {e}")
                            # Fallback to bounding box if mask extraction fails
                            billboard_masks.append((x1, y1, x2-x1, y2-y1))
            else:
                # Fallback to normal detection if no masks
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                    class_counts[class_name] += 1
                    
                    detection_list.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))
                    
                    # For billboards, store bounding box
                    if cls_id == 0:  # billboard class
                        billboard_masks.append((x1, y1, x2-x1, y2-y1))
    else:
        with CaptureOutput():
            inference_start = time.time()
            results = model(frame, verbose=False)
            inference_end = time.time()
        inference_times.append(inference_end - inference_start)
        
        detections = results.xyxy[0].cpu().numpy()
        
        detection_list = []
        for x1, y1, x2, y2, conf, cls_id in detections:
            if conf > 0.45:
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                class_id = int(cls_id)
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                class_counts[class_name] += 1
                
                detection_list.append(([x, y, w, h], conf, int(cls_id)))
    
    # Update tracker
    tracks = tracker.update_tracks(detection_list, frame=frame)
    
    # Dictionary to collect all masks before applying them
    # (this prevents masking artifacts when boxes overlap)
    tracked_billboard_masks = []  # Create a new list for tracking

    # Process and display tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        unique_tracks.add(track_id)
        ltrb = track.to_ltrb()  # [left, top, right, bottom]
        
        x1, y1, x2, y2 = map(int, ltrb)
        cls_id = track.get_det_class()
        
        # Get class name
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        
        # Generate consistent color for this track ID
        if track_id not in color_map:
            color_map[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        color = color_map[track_id]
        
        # For billboard class (0), collect mask information
        if cls_id == 0:  # billboard class
            # Find the corresponding mask from billboard_masks
            matching_mask = None
            for mask in billboard_masks:
                if isinstance(mask, tuple):
                    if len(mask) == 2 and not isinstance(mask[0], (int, float)):
                        # This is a segmentation mask with bbox
                        _, (mx1, my1, mx2, my2) = mask
                        # Check if this mask's bbox significantly overlaps with the tracked bbox
                        if (mx1 < x2 and mx2 > x1 and my1 < y2 and my2 > y1):
                            # Calculate IoU or just use the first match
                            matching_mask = mask
                            break

            if matching_mask:
                tracked_billboard_masks.append(matching_mask)
            else:
                # Fallback to using the bbox
                tracked_billboard_masks.append((x1, y1, x2-x1, y2-y1))

            # Draw ID labels above the billboard
            label = f"ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # For non-billboard objects, draw normal bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {class_name}"
            cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Apply masks to all billboards
    for mask_info in tracked_billboard_masks:
        apply_mask(frame, mask_info, MASK_TYPE)
    
    # Performance monitoring
    frame_end_time = time.time()
    process_times.append(frame_end_time - frame_start_time)
    
    # Draw FPS and progress
    fps_text = f"FPS: {current_fps:.1f}"
    progress_text = f"Frame: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
    masking_text = f"Masking: {MASK_TYPE}"
    cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, progress_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, masking_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Billboard Masking", frame)
    
    # Press 'm' to change masking type, 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        # Cycle through masking types
        if MASK_TYPE == "BLUR":
            MASK_TYPE = "COLOR"
        elif MASK_TYPE == "COLOR":
            MASK_TYPE = "PIXELATE"
        else:
            MASK_TYPE = "BLUR"
        print(f"Changed masking method to: {MASK_TYPE}")

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