"""AdSwapAI R&D, 2025-04-13: "known ads" memory - IoU matching of detections to
remembered boards plus EMA smoothing of their polygons across frames."""

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

# ----- Helper Functions -----

def order_points(pts):
    """
    Orders four points in the following order:
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
    If the vertical extent is larger than the horizontal extent,
    rotates the polygon 90 degrees clockwise.
    This brings the ad area's width-to-height ratio in line with expectations.
    """
    ordered = order_points(pts)
    width_top = np.linalg.norm(ordered[0] - ordered[1])
    width_bot = np.linalg.norm(ordered[3] - ordered[2])
    height_left = np.linalg.norm(ordered[0] - ordered[3])
    height_right = np.linalg.norm(ordered[1] - ordered[2])
    avg_width = (width_top + width_bot) / 2.0
    avg_height = (height_left + height_right) / 2.0
    if avg_height > avg_width:
        center = np.mean(ordered, axis=0)
        rotated = []
        for p in ordered:
            dx = p[0] - center[0]
            dy = p[1] - center[1]
            new_p = [center[0] + dy, center[1] - dx]  # rotate 90 degrees clockwise
            rotated.append(new_p)
        ordered = order_points(np.array(rotated, dtype=np.float32))
    return ordered

def get_polygon_center(poly):
    """Computes the center point of a polygon."""
    return np.mean(poly, axis=0)

def smooth_polygon(current_poly, prev_poly, smoothing_factor):
    """
    Applies exponential smoothing: new = alpha * current + (1 - alpha) * previous.
    This reduces sudden changes in the polygon's position.
    """
    return smoothing_factor * current_poly + (1 - smoothing_factor) * prev_poly

def polygon_iou(poly1, poly2):
    """
    Computes the IoU (Intersection over Union) ratio of two polygons.
    Note: the polygons' corners are assumed to be ordered and (likely) convex.
    """
    poly1 = order_points(poly1)
    poly2 = order_points(poly2)
    area1 = cv2.contourArea(poly1)
    area2 = cv2.contourArea(poly2)
    inter_area, _ = cv2.intersectConvexConvex(poly1, poly2)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0
    return inter_area / union_area

# ----- Global Parameters and Memory -----

# List for storing ads detected in memory (each element is a dict: 'polygon', 'center')
known_ads = []

# Exponential smoothing coefficient - a higher value (e.g. 0.2) makes the overlay look more stable
smoothing_factor = 0.2

# IoU threshold between a remembered ad and a new detection
match_threshold = 0.50

# (Optional) matching distance threshold for DeepSORT
matching_distance_threshold = 30

# Blending coefficient for a full replacement (1: fully replaced)
alpha = 1

# Skip replacement if the detected ad area is larger than 50% of the video width
sizeLimit = 0.5

# ----- Loading the Model and Resources -----

# Load the YOLO model (make sure it's trained for segmentation)
model = YOLO(MODEL_PATH)

# (Optional) DeepSORT tracker
tracker = DeepSort()

# Load the ad image to use for replacement (check the file path)
replacement_img = cv2.imread(REPLACEMENT_PATH)
rep_h, rep_w = replacement_img.shape[:2]
src_pts = np.array([[0, 0], [rep_w, 0], [rep_w, rep_h], [0, rep_h]], dtype=np.float32)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# VideoWriter settings for the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%H%M%S')
output_path = os.path.join(OUTPUT_DIR, f"output_replaced_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Global counters: total new ads found and total tracked ads
total_new_ads = 0
total_tracked_ads = 0

start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run the YOLO model to get detection results
    results = model(frame, augment=False)[0]

    detections = []  # (Optional) detection data for DeepSORT
    use_polygons = (results.masks is not None) and (len(results.masks.xy) > 0)

    # List for storing the polygons of ads detected in this frame
    current_frame_ads = []

    # Per-frame counters
    new_ads_count = 0
    tracked_ads_count = 0

    for i, box in enumerate(results.boxes):
        # Get the bounding box coordinates
        xyxy = box.xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = xyxy
        target_width = int(x2 - x1)
        target_height = int(y2 - y1)

        # Skip replacement if the area is larger than a certain fraction of the video width
        if target_width > sizeLimit * frame_width:
            conf = box.conf.cpu().numpy()[0]
            bbox = [x1, y1, target_width, target_height]
            detections.append([bbox, conf])
            continue

        # Get the polygon info (if segmentation is available)
        if use_polygons and i < len(results.masks.xy):
            pts = results.masks.xy[i].reshape(-1, 2).astype(np.float32)
            if pts.shape[0] >= 4:
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

        # Fix the ad polygon's orientation (an ad's width should generally exceed its height)
        dst_pts = enforce_orientation(dst_pts)

        # Check the IoU against ads already known in memory
        match_found = False
        smoothed_poly = dst_pts.copy()
        for ad in known_ads:
            iou_score = polygon_iou(ad['polygon'], dst_pts)
            if iou_score >= match_threshold:
                # Match found: track the remembered ad with a smooth update
                smoothed_poly = smooth_polygon(dst_pts, ad['polygon'], smoothing_factor)
                ad['polygon'] = smoothed_poly
                ad['center'] = get_polygon_center(smoothed_poly)
                match_found = True
                break

        if match_found:
            tracked_ads_count += 1
            total_tracked_ads += 1
        else:
            # New ad: add it to memory
            known_ads.append({
                'polygon': dst_pts,
                'center': get_polygon_center(dst_pts)
            })
            new_ads_count += 1
            total_new_ads += 1
            smoothed_poly = dst_pts

        current_frame_ads.append((get_polygon_center(smoothed_poly), smoothed_poly))

        # Replacement step:
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

        # Build the mask and blend in the replacement image
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(smoothed_poly)], 255)
        mask_bool = mask.astype(bool)
        frame_float = frame.astype(np.float32)
        warped_float = warped.astype(np.float32)
        blended = frame_float * (1 - alpha) + warped_float * alpha
        blended = blended.astype(np.uint8)
        frame[mask_bool] = blended[mask_bool]

        # Prepare the detection data for DeepSORT (optional)
        conf = box.conf.cpu().numpy()[0]
        bbox = [x1, y1, target_width, target_height]
        detections.append([bbox, conf])

    # (Optional) update the objects detected by the DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Write the frame to the output video
    out.write(frame)
    frame_count += 1

    # Summary log at the end of each frame: how many new ads were found this frame, how many old ads were tracked?
    print(f"Frame {frame_count}: Found new ads: {new_ads_count}, Tracked old ads: {tracked_ads_count}")

# Overall summary once video processing is complete
elapsed_time = time.time() - start_time
avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video output file: {output_path}")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Total new ads found: {total_new_ads}")
print(f"Total tracked old ads: {total_tracked_ads}")
