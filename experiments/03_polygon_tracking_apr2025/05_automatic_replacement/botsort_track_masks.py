"""AdSwapAI R&D, 2025-04-08: Ultralytics model.track (BoT-SORT) with a per-track
colored mask overlay."""

import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
import time
from datetime import datetime

VIDEO_PATH = "data/adVideo2.mp4"   # sample clip, see repo docs/assets.md
OUTPUT_DIR = "output"
MODEL_PATH = "best.pt"   # single-class YOLOv8s-seg "billboard" model, see docs/assets.md

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device('cuda')
else:
    print("CUDA is not available, using CPU")
    DEVICE = torch.device('cpu')

def get_color(track_id):
    """
    Returns a distinct color for each track ID.
    Colors are in OpenCV BGR format.
    """
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0)   # Olive
    ]
    return colors[track_id % len(colors)]

def process_video(video_path, model_path, output_path=None, confidence=0.5):
    """
    Processes the video, detects ad panels, and applies a distinct-color, semi-transparent
    overlay (mask) covering exactly the billboard area.

    Args:
        video_path: input video file path.
        model_path: YOLOv8 model file path (.pt).
        output_path: output video file path (None means no output is saved).
        confidence: confidence threshold for detection.

    Returns:
        Dictionary containing the detected track data.
    """
    # Load the YOLO model
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    if CUDA_AVAILABLE:
        model.to(DEVICE)

    # Open the video
    print(f"Opening video from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return {}

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Output video writer (if any)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Writing output to {output_path}")

    # Variables for tracking
    tracks = {}  # {track_id: {'points': [], 'boxes': [], 'class': int, 'first_frame': int, 'last_frame': int, 'masks': [optional]}}
    next_track_id = 1

    # For FPS calculation
    start_time = time.time()
    frames_processed = 0

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frames_processed += 1

            # Detection and tracking with YOLOv8
            results = model.track(frame, persist=True, verbose=False, conf=confidence)
            if results:
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    boxes = results[0].boxes
                    # Get tracking IDs (if any)
                    track_ids = boxes.id.int().cpu().tolist() if hasattr(boxes, 'id') and boxes.id is not None else None

                    for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                        x1, y1, x2, y2 = map(int, box)
                        # Use the existing tracking ID if present, otherwise assign a new one
                        if track_ids and i < len(track_ids):
                            track_id = track_ids[i]
                        else:
                            track_id = next_track_id
                            next_track_id += 1

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if track_id not in tracks:
                            tracks[track_id] = {
                                'points': [],
                                'boxes': [],
                                'class': int(boxes.cls[i].item()) if hasattr(boxes, 'cls') else 0,
                                'first_frame': frame_idx
                            }
                        tracks[track_id]['points'].append((center_x, center_y))
                        tracks[track_id]['boxes'].append((x1, y1, x2, y2))
                        tracks[track_id]['last_frame'] = frame_idx

                        # Determine color
                        color = get_color(track_id)

                        # For each detection, we build the ad-area mask.
                        # First, we create an empty single-channel mask.
                        detection_mask = np.zeros((height, width), dtype=np.uint8)

                        if hasattr(results[0], 'masks') and results[0].masks is not None:
                            try:
                                seg_mask = results[0].masks.data[i].cpu().numpy()
                                # Resize the segmentation mask if it doesn't match the original frame size.
                                if seg_mask.shape[:2] != (height, width):
                                    seg_mask = cv2.resize(seg_mask, (width, height))
                                # Apply threshold: values above 0.5 are accepted as the ad area
                                binary_mask = (seg_mask > 0.5).astype(np.uint8) * 255
                                detection_mask = binary_mask
                            except Exception as e:
                                # On error, fall back to the bounding box area
                                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                                cv2.fillPoly(detection_mask, [pts], 255)
                        else:
                            # If there's no segmentation mask, the entire bounding box is used as the ad area.
                            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                            cv2.fillPoly(detection_mask, [pts], 255)

                        # detection_mask now marks the ad area as a single channel (0 or 255).
                        # Let's convert this to three channels:
                        mask_3c = cv2.merge([detection_mask, detection_mask, detection_mask])

                        # Create a fully-colored overlay for the ad area.
                        overlay = np.full_like(frame, color, dtype=np.uint8)

                        # Apply alpha-blending only within the mask area:
                        alpha = 0.4  # Transparency ratio
                        frame_float = frame.astype(np.float32)
                        overlay_float = overlay.astype(np.float32)
                        mask_norm = (mask_3c.astype(np.float32) / 255.0)  # mask with values of 0 or 1
                        # Blending is applied only where the mask is set; other regions stay unchanged.
                        blended = frame_float * (1 - mask_norm * alpha) + overlay_float * (mask_norm * alpha)
                        blended = np.clip(blended, 0, 255).astype(np.uint8)
                        frame = blended

            # If there's an output video, save the frame
            if writer:
                writer.write(frame)

            # Print progress every 10 frames
            if frame_idx % 10 == 0:
                elapsed = time.time() - start_time
                fps_rate = frames_processed / elapsed if elapsed > 0 else 0
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"Processing: {progress:.1f}% complete, {fps_rate:.1f} FPS", end='\r')
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    
    total_time = time.time() - start_time
    avg_fps = frames_processed / total_time if total_time > 0 else 0
    print(f"\nProcessing completed: {frames_processed} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")
    
    return tracks

def main():
    # File paths
    video_path = VIDEO_PATH
    model_path = MODEL_PATH
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f"output_masked_{timestamp}.mp4")

    # Process the video
    tracks = process_video(video_path, model_path, output_path, confidence=0.3)
    
    print(f"Detected and tracked {len(tracks)} advertisement panels")
    for track_id, track_data in tracks.items():
        duration = track_data['last_frame'] - track_data['first_frame']
        print(f"  Track {track_id}: {len(track_data['points'])} points, duration: {duration} frames")

if __name__ == "__main__":
    main()
