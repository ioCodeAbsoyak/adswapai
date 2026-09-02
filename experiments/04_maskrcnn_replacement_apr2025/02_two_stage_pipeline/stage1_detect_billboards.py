"""AdSwapAI R&D, 2025-04-22: two-stage pipeline, stage 1 - detect billboards every
N frames, track ids, apply temporal smoothing, and write detections to a JSON file
(Detectron2 Mask R-CNN billboard model)."""

import cv2
import argparse
import numpy as np
import os
import json
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
import time

def setup_predictor(model_path):
    """Set up the detectron2 predictor with the model path"""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (billboard)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Increased threshold for more confident detections
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Register a dummy dataset to get metadata
    if "billboard_test" not in DatasetCatalog.list():
        DatasetCatalog.register("billboard_test", lambda: [])
        MetadataCatalog.get("billboard_test").set(thing_classes=["billboard"])
    
    return DefaultPredictor(cfg)

def find_corners_from_mask(mask):
    """Find four corners from a binary mask using only OpenCV"""
    # Convert to binary mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If not 4 points, use minimum area rectangle
    if len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box).astype(np.int32)
        return box
    
    return approx.reshape(-1, 2)

def main(args):
    # Setup predictor
    predictor = setup_predictor(args.model_path)
    
    # Open video to get dimensions
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {args.input_video}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize detection data structure
    # Format: { frame_idx: [list of detections] }
    # Each detection: {id, mask, bbox, corners}
    detections_data = {}
    
    # Initialize billboard ID tracking
    next_billboard_id = 1
    active_billboards = {}  # id -> {last_frame, bbox, history, iou_threshold, etc}
    
    # Define the keyframe interval
    keyframe_interval = 5  # Only detect on every 5th frame
    
    # For FPS calculation
    frame_count = 0
    start_time = time.time()
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_detections = []
        
        # Only run detection on keyframes
        is_keyframe = (frame_count % keyframe_interval == 0)
        
        # Process detections
        if is_keyframe:
            # Run detection
            outputs = predictor(frame)
            instances = outputs["instances"].to("cpu")
            
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()
                boxes = instances.pred_boxes.tensor.numpy()
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    x1, y1, x2, y2 = box.astype(int)
                    w, h = x2 - x1, y2 - y1
                    
                    # Skip tiny detections
                    if w < 20 or h < 20:
                        continue
                    
                    # Convert mask to polygon
                    try:
                        # Get corners
                        corners = find_corners_from_mask(mask)
                        if corners is None:
                            continue
                            
                        # Encode mask using indices for compact storage
                        mask_indices = np.where(mask)
                        mask_points = list(zip(mask_indices[0].tolist(), mask_indices[1].tolist()))
                        
                        # Try to match with existing billboards
                        matched_id = None
                        best_iou = 0
                        
                        for billboard_id, billboard_info in active_billboards.items():
                            last_frame, last_bbox, iou_threshold = billboard_info['last_frame'], billboard_info['bbox'], billboard_info['iou_threshold']
                            
                            # Skip if it's been too many frames
                            if frame_count - last_frame > keyframe_interval * 3:
                                continue
                            
                            # Calculate IoU between current detection and last bbox
                            old_x1, old_y1, old_w, old_h = last_bbox
                            old_x2, old_y2 = old_x1 + old_w, old_y1 + old_h
                            
                            # Intersection
                            inter_x1 = max(x1, old_x1)
                            inter_y1 = max(y1, old_y1)
                            inter_x2 = min(x2, old_x2)
                            inter_y2 = min(y2, old_y2)
                            
                            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                                area1 = w * h
                                area2 = old_w * old_h
                                iou = inter_area / float(area1 + area2 - inter_area)
                                
                                if iou > iou_threshold and iou > best_iou:
                                    matched_id = billboard_id
                                    best_iou = iou
                        
                        # If no match, create new billboard ID
                        if matched_id is None:
                            matched_id = next_billboard_id
                            next_billboard_id += 1
                            # Initialize new billboard with history
                            active_billboards[matched_id] = {
                                'last_frame': frame_count,
                                'bbox': (x1, y1, w, h),
                                'iou_threshold': 0.3,
                                'corners_history': [corners.tolist()],
                                'mask_history': [mask_points]
                            }
                        else:
                            # Update existing billboard info
                            # Gradually reduce threshold as we track longer
                            current_threshold = active_billboards[matched_id]['iou_threshold']
                            new_threshold = max(0.2, current_threshold - 0.01)
                            
                            # Update history
                            if 'corners_history' not in active_billboards[matched_id]:
                                active_billboards[matched_id]['corners_history'] = []
                            if 'mask_history' not in active_billboards[matched_id]:
                                active_billboards[matched_id]['mask_history'] = []
                                
                            active_billboards[matched_id]['corners_history'].append(corners.tolist())
                            active_billboards[matched_id]['mask_history'].append(mask_points)
                            
                            # Keep only the last 3 frames of history
                            if len(active_billboards[matched_id]['corners_history']) > 3:
                                active_billboards[matched_id]['corners_history'] = active_billboards[matched_id]['corners_history'][-3:]
                            if len(active_billboards[matched_id]['mask_history']) > 3:
                                active_billboards[matched_id]['mask_history'] = active_billboards[matched_id]['mask_history'][-3:]
                                
                            # Update basic info
                            active_billboards[matched_id].update({
                                'last_frame': frame_count,
                                'bbox': (x1, y1, w, h),
                                'iou_threshold': new_threshold
                            })
                        
                        # Apply temporal smoothing to corners and mask if we have history
                        if len(active_billboards[matched_id]['corners_history']) >= 2:
                            # Average the corners
                            corners_history = active_billboards[matched_id]['corners_history']
                            avg_corners = np.mean([np.array(c) for c in corners_history], axis=0)
                            corners = avg_corners.astype(np.int32)
                            
                            # Union of masks from history
                            mask_history = active_billboards[matched_id]['mask_history']
                            all_points = set()
                            for points in mask_history:
                                all_points.update([tuple(p) for p in points])
                            mask_points = list(map(list, all_points))
                        
                        # Add the detection with temporal smoothing applied
                        detection = {
                            'id': matched_id,
                            'bbox': [int(x1), int(y1), int(w), int(h)],
                            'corners': corners.tolist() if isinstance(corners, np.ndarray) else corners,
                            'mask_points': mask_points
                        }
                        frame_detections.append(detection)
                        
                    except Exception as e:
                        print(f"Error processing detection in frame {frame_count}: {e}")
                        continue
        else:
            # For non-keyframes, interpolate detections from previous frames
            billboards_to_show = []
            
            # Find billboards that were recently detected
            for billboard_id, info in active_billboards.items():
                last_frame = info['last_frame']
                
                # Only consider billboards seen recently
                if frame_count - last_frame <= keyframe_interval:
                    billboards_to_show.append(billboard_id)
            
            # Add interpolated detections
            for billboard_id in billboards_to_show:
                info = active_billboards[billboard_id]
                
                # Simply use the last known position and appearance
                detection = {
                    'id': billboard_id,
                    'bbox': list(info['bbox']),
                    'corners': info['corners_history'][-1],
                    'mask_points': info['mask_history'][-1]
                }
                frame_detections.append(detection)
        
        # Store frame detections
        if frame_detections:
            detections_data[frame_count] = frame_detections
        
        # Print progress
        if frame_count % 20 == 0:
            elapsed_time = time.time() - start_time
            fps_calc = frame_count / elapsed_time
            print(f"Processed {frame_count}/{total_frames} frames... FPS: {fps_calc:.2f}")
    
    # Clean up
    cap.release()
    
    # Save detections to JSON file
    with open(args.output_json, 'w') as f:
        json.dump(
            {
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'total_frames': total_frames
                },
                'detections': {str(k): v for k, v in detections_data.items()}  # Convert keys to strings for JSON
            },
            f
        )
    
    print(f"Detection completed! Data saved to {args.output_json}")
    print(f"Total billboards detected: {next_billboard_id - 1}")
    
    # Create a simple visualization if requested
    if args.create_vis:
        vis_output = args.output_json.replace('.json', '_vis.mp4')
        print(f"Creating visualization: {vis_output}")
        
        # Reopen the video
        cap = cv2.VideoCapture(args.input_video)
        out = cv2.VideoWriter(vis_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        
        frame_idx = 0
        
        # Create a color map for billboard IDs
        color_map = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Draw detections for this frame
            if str(frame_idx) in detections_data:
                for det in detections_data[str(frame_idx)]:
                    billboard_id = det['id']
                    
                    # Assign a color for this billboard if not assigned yet
                    if billboard_id not in color_map:
                        color_map[billboard_id] = (
                            np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255)
                        )
                    
                    color = color_map[billboard_id]
                    
                    # Draw bounding box
                    x, y, w, h = det['bbox']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw corners
                    corners = np.array(det['corners'])
                    cv2.polylines(frame, [corners.astype(np.int32)], True, color, 2)
                    
                    # Draw ID
                    cv2.putText(frame, f"ID: {billboard_id}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame)
            
            if frame_idx % 100 == 0:
                print(f"Visualization: {frame_idx}/{total_frames}")
        
        cap.release()
        out.release()
        print(f"Visualization saved to {vis_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and track billboards in video with temporal smoothing")
    parser.add_argument("--model-path", default="model_final.pth", help="Path to model weights (custom billboard model, see docs/assets.md)")
    parser.add_argument("--input-video", default="data/adVideo1.mp4", help="Path to input video (sample clip, see repo docs/assets.md)")
    parser.add_argument("--output-json", default="output/detections.json", help="Path to output JSON file")
    parser.add_argument("--create-vis", action="store_true", help="Create visualization video")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    main(args)