"""AdSwapAI R&D, 2025-04-24: detect + DeepSORT-style tracking (Kalman filter,
MobileNetV2 appearance features, Hungarian matching) and billboard replacement
(Detectron2 Mask R-CNN billboard model)."""

import cv2
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from scipy.optimize import linear_sum_assignment
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

class FeatureExtractor(nn.Module):
    """CNN for extracting deep appearance features for tracking"""
    def __init__(self, use_cuda=True):
        super(FeatureExtractor, self).__init__()
        # Use a simpler feature extractor to improve speed
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        # Remove the last layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Normalization for the model
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.norm
        ])
    
    def extract(self, image_patches):
        """Extract features from image patches"""
        if not image_patches:
            return []
            
        tensors = []
        for patch in image_patches:
            if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                # Create a small black patch if empty
                patch = np.zeros((10, 10, 3), dtype=np.uint8)
            tensor = self.transform(patch).unsqueeze(0).to(self.device)
            tensors.append(tensor)
            
        batched_tensors = torch.cat(tensors, dim=0)
        
        with torch.no_grad():
            features = self.model(batched_tensors)
            features = features.squeeze().cpu().numpy()
            
        # Normalize features
        if len(features.shape) == 1:  # Only one sample
            features = features.reshape(1, -1)
            
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        return features

class KalmanTracker:
    """Kalman filter for tracking billboards"""
    def __init__(self, bbox, track_id, mask=None, corners=None, max_age=5):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h) format
        self.mask = mask
        self.corners = corners
        self.max_age = max_age
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.features = []
        self.visible = True
        self.last_mask = mask
        self.last_corners = corners
        self.consistency_count = 0  # Used to track stability
        
        # Initialize Kalman filter (8 state, 4 measurement)
        self.kf = cv2.KalmanFilter(8, 4)
        
        # State: [x, y, w, h, vx, vy, vw, vh]
        # Measurement: [x, y, w, h]
        
        # Transition matrix F
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ], dtype=np.float32)
        
        # Observation matrix H
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0],  # w
            [0, 0, 0, 1, 0, 0, 0, 0],  # h
        ], dtype=np.float32)
        
        # Process noise covariance Q
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Measurement noise covariance R
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Error covariance P
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0
        
        # Initialize state
        x, y, w, h = bbox
        self.kf.statePost = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
    
    def predict(self):
        """Predict next state using Kalman filter"""
        prediction = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        # Extract predicted bbox
        x = max(0, prediction[0][0])
        y = max(0, prediction[1][0])
        w = max(1, prediction[2][0])
        h = max(1, prediction[3][0])
        
        self.bbox = (int(x), int(y), int(w), int(h))
        return self.bbox
    
    def update(self, bbox, mask=None, corners=None, feature=None):
        """Update tracker with new detection"""
        x, y, w, h = bbox
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)
        
        # Update Kalman filter
        self.kf.correct(measurement)
        
        # Update state
        state = self.kf.statePost
        x = max(0, state[0][0])
        y = max(0, state[1][0])
        w = max(1, state[2][0])
        h = max(1, state[3][0])
        
        self.bbox = (int(x), int(y), int(w), int(h))
        if mask is not None:
            self.mask = mask
            self.last_mask = mask
            
        if corners is not None:
            self.corners = corners
            self.last_corners = corners
            
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 30:  # Limit feature history
                self.features.pop(0)
                
        self.hits += 1
        self.time_since_update = 0
        self.visible = True
        self.consistency_count += 1
        return self.bbox
    
    def get_feature(self):
        """Get average feature vector"""
        if not self.features:
            return None
        return np.mean(self.features, axis=0)
    
    def is_confirmed(self):
        """Check if tracker is confirmed (reliable)"""
        return self.hits >= 3
    
    def is_deleted(self):
        """Check if tracker should be deleted"""
        return self.time_since_update > self.max_age
        
    def check_visibility(self, frame_width, frame_height, margin=0):
        """Check if the billboard is still visible in the frame"""
        x, y, w, h = self.bbox
        
        # Check if the billboard is completely outside the frame
        if (x + w < -margin or x > frame_width + margin or 
            y + h < -margin or y > frame_height + margin):
            self.visible = False
            return False
        
        # Check if the billboard is mostly outside the frame
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Billboard must have its center in the frame
        if (center_x < -margin or center_x > frame_width + margin or
            center_y < -margin or center_y > frame_height + margin):
            self.visible = False
            return False
            
        return True

class BillboardTracker:
    """Tracker for billboards"""
    def __init__(self, frame_width, frame_height, max_age=5, feature_matching=True, use_cuda=True):
        self.max_age = max_age
        self.feature_matching = feature_matching
        self.trackers = []
        self.frame_count = 0
        self.next_id = 1
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Feature extractor for appearance matching
        if feature_matching:
            self.feature_extractor = FeatureExtractor(use_cuda)
        else:
            self.feature_extractor = None
    
    def update(self, frame, detections):
        """Update trackers with new detections"""
        self.frame_count += 1
        
        # Predict new locations of all trackers
        for tracker in self.trackers:
            tracker.predict()
            # Check if tracker is still in frame with a small margin
            tracker.check_visibility(self.frame_width, self.frame_height, margin=-10)
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if not t.is_deleted()]
        
        # Extract features from detection patches if feature matching is enabled
        if self.feature_matching and detections and self.feature_extractor:
            patches = []
            for det in detections:
                x, y, w, h = det['bbox']
                if x >= 0 and y >= 0 and w > 0 and h > 0 and x < self.frame_width and y < self.frame_height:
                    # Ensure we don't go outside frame boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, self.frame_width - x)
                    h = min(h, self.frame_height - y)
                    patch = frame[y:y+h, x:x+w]
                    patches.append(patch)
                else:
                    patches.append(np.zeros((10, 10, 3), dtype=np.uint8))
                    
            if patches:
                try:
                    features = self.feature_extractor.extract(patches)
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    features = [None] * len(detections)
            else:
                features = [None] * len(detections)
        else:
            features = [None] * len(detections)
        
        # Match detections to existing trackers
        if self.trackers and detections:
            # Calculate cost matrix
            cost_matrix = np.zeros((len(detections), len(self.trackers)))
            
            for i, (det, feat) in enumerate(zip(detections, features)):
                bbox1 = det['bbox']
                for j, tracker in enumerate(self.trackers):
                    bbox2 = tracker.bbox
                    
                    # IoU distance
                    iou_dist = 1.0 - self._iou(bbox1, bbox2)
                    
                    # Feature distance (if available)
                    feat_dist = 0.0
                    if self.feature_matching and feat is not None and tracker.get_feature() is not None:
                        try:
                            feat_dist = np.linalg.norm(feat - tracker.get_feature())
                            feat_dist = min(1.0, feat_dist)  # Cap at 1.0
                        except:
                            feat_dist = 0.5  # Default if error
                    
                    # Combined distance (70% IoU, 30% feature)
                    cost_matrix[i, j] = 0.7 * iou_dist + 0.3 * feat_dist
            
            # Use Hungarian algorithm for assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Filter matches with high cost
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_trackers = list(range(len(self.trackers)))
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.7:  # Threshold for matching
                    matches.append((r, c))
                    if r in unmatched_detections:
                        unmatched_detections.remove(r)
                    if c in unmatched_trackers:
                        unmatched_trackers.remove(c)
        else:
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_trackers = list(range(len(self.trackers)))
        
        # Update matched trackers
        for det_idx, trk_idx in matches:
            det = detections[det_idx]
            self.trackers[trk_idx].update(
                det['bbox'], 
                det.get('mask', None),
                det.get('corners', None),
                features[det_idx] if self.feature_matching else None
            )
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_tracker = KalmanTracker(
                det['bbox'],
                self.next_id,
                det.get('mask', None),
                det.get('corners', None),
                self.max_age
            )
            
            if self.feature_matching and features[det_idx] is not None:
                new_tracker.features.append(features[det_idx])
                
            self.trackers.append(new_tracker)
            self.next_id += 1
        
        # Return active and visible trackers with sufficient stability
        confirmed_trackers = [t for t in self.trackers if t.is_confirmed() and t.visible]
        return confirmed_trackers
    
    def _iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        # Convert from (x, y, w, h) to (x1, y1, x2, y2)
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_1 = w1 * h1
        area_2 = w2 * h2
        
        # Calculate IoU
        iou = area_i / float(area_1 + area_2 - area_i)
        return iou


def setup_predictor(model_path):
    """Set up the detectron2 predictor with the model path"""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (billboard)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
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

def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Compute the difference between the points
    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def replace_billboard(frame, mask, corners, replacement_img):
    """Replace billboard in the frame with the replacement image"""
    if corners is None or len(corners) != 4:
        # If no corners are provided, try to find them from the mask
        corners = find_corners_from_mask(mask)
        if corners is None or len(corners) != 4:
            return frame
    
    # Make sure corners are ordered correctly
    try:
        corners = order_points(corners.astype(np.float32))
    except:
        return frame
    
    # Get dimensions of replacement image
    h_repl, w_repl = replacement_img.shape[:2]
    
    # Define source points (corners of the replacement image)
    src_points = np.array([
        [0, 0],
        [w_repl - 1, 0],
        [w_repl - 1, h_repl - 1],
        [0, h_repl - 1]
    ], dtype=np.float32)
    
    # Calculate perspective transform
    try:
        M = cv2.getPerspectiveTransform(src_points, corners)
    except:
        return frame
    
    # Create a warped version of the replacement image
    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(replacement_img, M, (w, h))
    
    # Create a mask from the warped image
    warp_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Ensure corners are valid for fillPoly
    if np.any(np.isnan(corners)) or np.any(np.isinf(corners)):
        return frame
        
    cv2.fillPoly(warp_mask, [corners.astype(np.int32)], 255)
    warp_mask = warp_mask.astype(bool)
    
    # Combine the original and replacement images
    result = frame.copy()
    for c in range(3):
        result[:, :, c] = np.where(warp_mask, warped[:, :, c], result[:, :, c])
    
    return result

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
    
    # Setup tracker with frame dimensions
    tracker = BillboardTracker(width, height, max_age=3, feature_matching=True, use_cuda=True)
    
    # Load replacement image
    replacement_img = cv2.imread(args.replace_img)
    if replacement_img is None:
        raise ValueError(f"Could not load replacement image: {args.replace_img}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_video) or '.', exist_ok=True)
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    # Initialize billboard ID to replacement image mapping
    billboard_images = {}
    
    # For FPS calculation
    frame_count = 0
    start_time = time.time()
    
    # Maintain cache of previous replacements to ensure consistency
    replacement_cache = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        original_frame = frame.copy()
        result_frame = frame.copy()
        
        # Run object detection on the original frame (every 2 frames to improve speed)
        if frame_count % 2 == 0:
            outputs = predictor(original_frame)
            instances = outputs["instances"].to("cpu")
            
            # Check if we have masks
            if instances.has("pred_masks"):
                # Get detection information
                masks = instances.pred_masks.numpy()
                boxes = instances.pred_boxes.tensor.numpy()
                
                # Convert to the format expected by the tracker
                detections = []
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    x1, y1, x2, y2 = box.astype(int)
                    w, h = x2 - x1, y2 - y1
                    
                    # Skip detections that are too small or invalid
                    if w < 20 or h < 20 or w > width or h > height:
                        continue
                    
                    # Find corners for this mask
                    corners = None
                    try:
                        mask_roi = mask[y1:y2, x1:x2]
                        corners = find_corners_from_mask(mask_roi)
                        if corners is not None:
                            # Adjust corners to global image coordinates
                            corners[:, 0] += x1
                            corners[:, 1] += y1
                    except Exception as e:
                        if args.debug:
                            print(f"Error finding corners: {e}")
                    
                    detections.append({
                        'bbox': (x1, y1, w, h),
                        'mask': mask,
                        'corners': corners
                    })
                
                # Update trackers with new detections
                tracked_objects = tracker.update(original_frame, detections)
            else:
                # If no masks detected, just update tracker with empty detections
                tracked_objects = tracker.update(original_frame, [])
        else:
            # On off frames, just update tracker with no new detections
            tracked_objects = tracker.update(original_frame, [])
        
        # Process each tracked object
        for obj in tracked_objects:
            track_id = obj.track_id
            
            # If this is a new billboard, assign a replacement image
            if track_id not in billboard_images:
                billboard_images[track_id] = replacement_img
            
            # Check if tracker has valid mask and is visible
            if obj.visible and obj.consistency_count >= 2:  # Must be stable for at least 2 frames
                # Use the most recent mask if available, or last valid mask
                current_mask = obj.mask if obj.mask is not None else obj.last_mask
                current_corners = obj.corners if obj.corners is not None else obj.last_corners
                
                if current_mask is not None:
                    try:
                        # Apply the replacement
                        result_frame = replace_billboard(
                            result_frame, 
                            current_mask, 
                            current_corners,
                            billboard_images[track_id]
                        )
                        
                        # Cache this replacement
                        replacement_cache[track_id] = {
                            'frame': frame_count,
                            'mask': current_mask,
                            'corners': current_corners
                        }
                    except Exception as e:
                        if args.debug:
                            print(f"Error replacing billboard {track_id}: {e}")
                    
                    # Draw bounding box for debugging
                    if args.debug:
                        x, y, w, h = obj.bbox
                        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(result_frame, f"ID: {track_id}", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write the frame
        out.write(result_frame)
        
        # Calculate and print FPS
        if frame_count % 20 == 0:
            elapsed_time = time.time() - start_time
            fps_calc = frame_count / elapsed_time
            print(f"Processed {frame_count} frames... FPS: {fps_calc:.2f}")
    
    # Clean up
    cap.release()
    out.release()
    print(f"Done! Output written to {args.output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace billboards in video with improved tracking")
    parser.add_argument("--model-path", default="model_final.pth", help="Path to model weights (custom billboard model, see docs/assets.md)")
    parser.add_argument("--input-video", default="data/adVideo1.mp4", help="Path to input video (sample clip, see repo docs/assets.md)")
    parser.add_argument("--replace-img", default="data/replace.jpg", help="Path to replacement image")
    parser.add_argument("--output-video", default="output/result.mp4", help="Path to output video")
    parser.add_argument("--debug", action="store_true", help="Show tracking debug info")

    args = parser.parse_args()
    main(args)