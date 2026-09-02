"""AdSwapAI R&D, 2025-04-08: YOLOv8-seg best.pt detections drive the replacement
automatically via a bounding-box warp."""

import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = "data/adVideo2.mp4"   # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"
MODEL_PATH = "best.pt"   # single-class YOLOv8s-seg "billboard" model, see docs/assets.md

# Load the YOLO model (best.pt)
model = YOLO(MODEL_PATH)

# Initialize DeepSORT with default parameters
tracker = DeepSort()

# Load the replacement image
replacement_img = cv2.imread(REPLACEMENT_PATH)
rep_h, rep_w = replacement_img.shape[:2]
# Define the source points of the replacement image (top-left, top-right, bottom-right, bottom-left)
src_pts = np.array([[0, 0], [rep_w, 0], [rep_w, rep_h], [0, rep_h]], dtype=np.float32)

# Open the input video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties and prepare the output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%H%M%S')
output_path = os.path.join(OUTPUT_DIR, f"output_replaced_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection using YOLO; we assume it properly detects billboards.
    results = model(frame, augment=False)[0]
    
    detections = []  # List to store detections for DeepSORT
    if results.boxes is not None:
        for box in results.boxes:
            # Extract bounding box coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy.cpu().numpy().flatten()  # e.g. [x1, y1, x2, y2]
            x1, y1, x2, y2 = xyxy
            # Create destination points (assume a rectangle here)
            dst_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

            # Compute perspective transformation matrix and warp the replacement image to fit the billboard
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(replacement_img, M, (frame_width, frame_height))

            # Create a mask for the billboard region and apply the warped replacement image
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(dst_pts)], 255)
            mask_bool = mask.astype(bool)
            frame[mask_bool] = warped[mask_bool]

            # For DeepSORT: structure the detection as [[x, y, width, height], confidence]
            conf = box.conf.cpu().numpy()[0]
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append([bbox, conf])

    # Update the tracker with the formatted detections
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()  # Format: left, top, right, bottom
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # TODO: Later, add occlusion handling so that areas with people, referees, or the ball are excluded from replacement

    # Write the processed frame to the output video
    out.write(frame)
    
    # Optional: display the frame in real time; press 'q' to quit
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video output file: {output_path}")
