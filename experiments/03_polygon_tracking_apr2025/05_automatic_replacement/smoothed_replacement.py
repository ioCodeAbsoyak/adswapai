"""AdSwapAI R&D, 2025-04-08: orientation enforcement + exponential smoothing of the
detected polygons between frames."""

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

# ----- Helper functions for polygon ordering, orientation and smoothing -----

def order_points(pts):
    """
    Order a set of 4 points in the order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def enforce_orientation(pts):
    """
    Enforce the heuristic that ad areas have a short height and a long width.
    If the average vertical dimension is larger than the horizontal, rotate
    the ordered points by 90 degrees (clockwise).
    """
    ordered = order_points(pts)
    width_top = np.linalg.norm(ordered[0] - ordered[1])
    width_bot = np.linalg.norm(ordered[3] - ordered[2])
    height_left = np.linalg.norm(ordered[0] - ordered[3])
    height_right = np.linalg.norm(ordered[1] - ordered[2])
    avg_width = (width_top + width_bot) / 2.0
    avg_height = (height_left + height_right) / 2.0
    if avg_height > avg_width:
        # Compute center and rotate each point 90 degrees clockwise
        center = np.mean(ordered, axis=0)
        rotated = []
        for p in ordered:
            dx = p[0] - center[0]
            dy = p[1] - center[1]
            new_p = [center[0] + dy, center[1] - dx]  # 90 degrees clockwise
            rotated.append(new_p)
        ordered = order_points(np.array(rotated, dtype=np.float32))
    return ordered

def get_polygon_center(poly):
    """Compute the center point of a polygon."""
    return np.mean(poly, axis=0)

def smooth_polygon(current_poly, prev_poly, smoothing_factor):
    """
    Exponential smoothing: new = alpha * current + (1 - alpha) * previous.
    """
    return smoothing_factor * current_poly + (1 - smoothing_factor) * prev_poly

# ----- Global parameters for smoothing across frames -----
prev_polygons = []  # Will store list of (center, polygon) for previous frame
smoothing_factor = 0.8  # Weight for current polygon (higher = less smoothing)
matching_distance_threshold = 20  # Maximum distance in pixels to consider same area

# ----- Main Code -----

# Load the YOLO model (ensure your model is trained for segmentation)
model = YOLO(MODEL_PATH)

# Initialize DeepSORT tracker with default parameters
tracker = DeepSort()

# Load the replacement image; verify the path is correct
replacement_img = cv2.imread(REPLACEMENT_PATH)
rep_h, rep_w = replacement_img.shape[:2]
# Define source polygon for the replacement image (ordered clockwise)
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

# Blending factor: set to 1 for full replacement (change if you need transparency)
alpha = 1  
# Skip replacement if billboard width > 50% of frame width
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
    # Check if segmentation masks are available (if model returns polygon info)
    use_polygons = (results.masks is not None) and (len(results.masks.xy) > 0)
    
    current_polygons = []  # Will collect polygons for current frame (for smoothing)

    # Loop over detected boxes
    for i, box in enumerate(results.boxes):
        # Extract bounding box coordinates: [x1, y1, x2, y2]
        xyxy = box.xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = xyxy
        target_width = int(x2 - x1)
        target_height = int(y2 - y1)
        
        # Skip replacement if the billboard's width exceeds the limit
        if target_width > sizeLimit * frame_width:
            conf = box.conf.cpu().numpy()[0]
            bbox = [x1, y1, target_width, target_height]
            detections.append([bbox, conf])
            continue

        # Get destination polygon: try to use segmentation polygon if available,
        # otherwise fallback to bounding box corners.
        if use_polygons and i < len(results.masks.xy):
            pts = results.masks.xy[i].reshape(-1, 2).astype(np.float32)
            if pts.shape[0] >= 4:
                # Approximate to simplify polygon
                epsilon = 0.02 * cv2.arcLength(pts, True)
                approx = cv2.approxPolyDP(pts, epsilon, True)
                if len(approx) == 4:
                    dst_pts = approx.reshape(4, 2)
                else:
                    rect = cv2.minAreaRect(pts)
                    dst_pts = cv2.boxPoints(rect).astype(np.float32)
            else:
                dst_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        else:
            dst_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        # Enforce proper orientation so that the short side becomes vertical
        dst_pts = enforce_orientation(dst_pts)
        
        # Compute the center of the current polygon
        curr_center = get_polygon_center(dst_pts)
        smoothed_poly = dst_pts.copy()
        matched = False
        
        # Attempt to match with a polygon from the previous frame (using center proximity)
        for prev_center, prev_poly in prev_polygons:
            dist = np.linalg.norm(curr_center - prev_center)
            if dist < matching_distance_threshold:
                smoothed_poly = smooth_polygon(dst_pts, prev_poly, smoothing_factor)
                matched = True
                break
        
        # Save the (smoothed) polygon for current frame
        current_polygons.append((get_polygon_center(smoothed_poly), smoothed_poly))
        
        # Determine if tiling is needed for a billboard larger than the replacement image
        if target_width > rep_w or target_height > rep_h:
            n_tiles_x = math.ceil(target_width / rep_w)
            n_tiles_y = math.ceil(target_height / rep_h)
            tiled_img = np.tile(replacement_img, (n_tiles_y, n_tiles_x, 1))
            tiled_height, tiled_width = tiled_img.shape[:2]
            tiled_src_pts = np.array([[0, 0], [tiled_width, 0], [tiled_width, tiled_height], [0, tiled_height]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(tiled_src_pts, smoothed_poly)
            warped = cv2.warpPerspective(tiled_img, M, (frame_width, frame_height))
        else:
            M = cv2.getPerspectiveTransform(src_pts, smoothed_poly)
            warped = cv2.warpPerspective(replacement_img, M, (frame_width, frame_height))
        
        # Create a mask for the replaced region based on the (smoothed) polygon
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(smoothed_poly)], 255)
        mask_bool = mask.astype(bool)
        
        # Alpha blend the warped replacement onto the frame
        frame_float = frame.astype(np.float32)
        warped_float = warped.astype(np.float32)
        blended = frame_float * (1 - alpha) + warped_float * alpha
        blended = blended.astype(np.uint8)
        frame[mask_bool] = blended[mask_bool]
        
        # Add detection for DeepSORT (format: [[x, y, width, height], confidence])
        conf = box.conf.cpu().numpy()[0]
        bbox = [x1, y1, target_width, target_height]
        detections.append([bbox, conf])
    
    # Update tracker (no IDs or boxes drawn as per instructions)
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Write the processed frame to the output video
    out.write(frame)
    frame_count += 1
    # Update previous polygons with the ones from this frame for smoothing next frame
    prev_polygons = current_polygons.copy()

# Calculate and print average FPS
elapsed_time = time.time() - start_time
avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video output file: {output_path}")
print(f"Average FPS: {avg_fps:.2f}")
