"""AdSwapAI R&D, 2025-04-08: segmentation polygon -> approxPolyDP / minAreaRect quadrilateral
-> perspective warp replacement."""

import os
import cv2
import numpy as np
import math
import time
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = "data/adVideo2.mp4"   # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"
MODEL_PATH = "best.pt"   # single-class YOLOv8s-seg "billboard" model, see docs/assets.md

# Load the YOLO model (make sure your model is trained for segmentation)
model = YOLO(MODEL_PATH)

# Initialize the DeepSORT tracker with default parameters
tracker = DeepSort()

# Load the replacement image (verify the path)
replacement_img = cv2.imread(REPLACEMENT_PATH)
rep_h, rep_w = replacement_img.shape[:2]
# Define source polygon points for the replacement image (clockwise: top-left, top-right, bottom-right, bottom-left)
src_pts = np.array([[0, 0], [rep_w, 0], [rep_w, rep_h], [0, rep_h]], dtype=np.float32)

# Open the input video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Prepare the video writer for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%H%M%S')
output_path = os.path.join(OUTPUT_DIR, f"output_replaced_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Set blending factor: 1 means full replacement, no transparency
alpha = 1  
# Skip replacement if the detected billboard width is greater than 50% of the frame width
sizeLimit = 0.5

# Start timing for FPS calculation
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model(frame, augment=False)[0]
    
    detections = []  # For DeepSORT tracking
    # Check if segmentation masks are available (assumes model returns segmentation data)
    use_polygons = results.masks is not None and len(results.masks.xy) > 0

    for i, box in enumerate(results.boxes):
        # Get bounding box coordinates: [x1, y1, x2, y2]
        xyxy = box.xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = xyxy
        target_width = int(x2 - x1)
        target_height = int(y2 - y1)
        
        # Skip replacement if the detected billboard's width exceeds 50% of the frame width
        if target_width > sizeLimit * frame_width:
            conf = box.conf.cpu().numpy()[0]
            bbox = [x1, y1, target_width, target_height]
            detections.append([bbox, conf])
            continue

        # Use detected polygon from segmentation if available; otherwise, use the bounding box
        if use_polygons and i < len(results.masks.xy):
            # Directly use the numpy array without calling .cpu()
            pts = results.masks.xy[i].reshape(-1, 2).astype(np.float32)
            if pts.shape[0] >= 4:
                # Approximate the polygon to simplify it
                epsilon = 0.02 * cv2.arcLength(pts, True)
                approx = cv2.approxPolyDP(pts, epsilon, True)
                if len(approx) == 4:
                    dst_pts = approx.reshape(4, 2)
                else:
                    # When approximation does not return 4 points, use the minimum area rectangle
                    rect = cv2.minAreaRect(pts)
                    dst_pts = cv2.boxPoints(rect).astype(np.float32)
            else:
                dst_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        else:
            dst_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        
        # Check if tiling is needed for a billboard larger than the replacement image
        if target_width > rep_w or target_height > rep_h:
            # Calculate number of tiles required horizontally and vertically
            n_tiles_x = math.ceil(target_width / rep_w)
            n_tiles_y = math.ceil(target_height / rep_h)
            # Create a tiled image by repeating the replacement image
            tiled_img = np.tile(replacement_img, (n_tiles_y, n_tiles_x, 1))
            tiled_height, tiled_width = tiled_img.shape[:2]
            # Define source points for the tiled image
            tiled_src_pts = np.array([[0, 0], [tiled_width, 0], [tiled_width, tiled_height], [0, tiled_height]], dtype=np.float32)
            # Compute the perspective transformation matrix
            M = cv2.getPerspectiveTransform(tiled_src_pts, dst_pts)
            warped = cv2.warpPerspective(tiled_img, M, (frame_width, frame_height))
        else:
            # Standard replacement using the replacement image
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(replacement_img, M, (frame_width, frame_height))
        
        # Create a mask for the destination polygon region
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(dst_pts)], 255)
        mask_bool = mask.astype(bool)
        
        # Alpha blend the warped replacement onto the original frame
        # new_pixel = original_pixel * (1 - alpha) + warped_pixel * alpha
        frame_float = frame.astype(np.float32)
        warped_float = warped.astype(np.float32)
        blended = frame_float * (1 - alpha) + warped_float * alpha
        blended = blended.astype(np.uint8)
        frame[mask_bool] = blended[mask_bool]

        # Add detection for DeepSORT (format: [[x, y, width, height], confidence])
        conf = box.conf.cpu().numpy()[0]
        bbox = [x1, y1, target_width, target_height]
        detections.append([bbox, conf])
    
    # Update tracker; no IDs or boxes are drawn per instruction
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Write the processed frame to the output video
    out.write(frame)
    frame_count += 1

# Calculate average FPS over processed frames
elapsed_time = time.time() - start_time
avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video output file: {output_path}")
print(f"Average FPS: {avg_fps:.2f}")
