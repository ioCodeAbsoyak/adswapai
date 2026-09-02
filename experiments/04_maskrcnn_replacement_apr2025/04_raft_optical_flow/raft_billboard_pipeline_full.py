#!/usr/bin/env python3
"""AdSwapAI R&D, 2025-04-26: full-featured RAFT optical-flow billboard replacement
pipeline (dataclasses, logging, tqdm progress bar, CLI flags --mask-mode,
--respect-foreground, --show-flow, --batch-size; Detectron2 Mask R-CNN billboard
model + torchvision RAFT)."""

import cv2
import argparse
import numpy as np
import os
import sys
import torch
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
from torchvision.transforms.functional import to_tensor
from torchvision.utils import flow_to_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BillboardReplacer")

# Try to import tqdm for progress bar, but don't fail if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.info("tqdm not available, using simple progress reporting")

@dataclass
class BillboardDetection:
    """Data class for billboard detection results"""
    mask: np.ndarray
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    score: float

@dataclass
class BillboardTrack:
    """Data class for tracked billboard"""
    id: int
    mask: np.ndarray
    score: float
    box: Tuple[int, int, int, int]
    age: int
    time_since_detection: int
    color: Tuple[int, int, int]

class Timer:
    """Simple timer for performance measurements"""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        logger.debug(f"{self.name}: {elapsed:.4f} seconds")

def setup_billboard_predictor(model_path: str, conf_thresh: float = 0.6) -> Tuple[DefaultPredictor, Any]:
    """Set up the billboard detector using custom model
    
    Args:
        model_path: Path to the model weights
        conf_thresh: Confidence threshold for detections
        
    Returns:
        Tuple of predictor and metadata
    """
    logger.info(f"Setting up billboard predictor with model: {model_path}")
    
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (billboard)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh
    
    # Performance optimization: increase IOU threshold to reduce number of predictions
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
    
    # Use FP16 precision if available
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.FP16_ENABLED = True
    else:
        cfg.MODEL.DEVICE = "cpu"
    
    # Register dataset metadata
    if "billboard_test" not in DatasetCatalog.list():
        DatasetCatalog.register("billboard_test", lambda: [])
        MetadataCatalog.get("billboard_test").set(thing_classes=["billboard"])
    
    try:
        predictor = DefaultPredictor(cfg)
        return predictor, MetadataCatalog.get("billboard_test")
    except Exception as e:
        logger.error(f"Failed to initialize billboard predictor: {e}")
        raise

def setup_coco_predictor(conf_thresh: float = 0.7) -> Tuple[DefaultPredictor, Any]:
    """Set up standard COCO detector for people and sports ball
    
    Args:
        conf_thresh: Confidence threshold for detections
        
    Returns:
        Tuple of predictor and metadata
    """
    logger.info("Setting up COCO predictor")
    
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh
    
    # Only detect specific classes: person (0) and sports ball (32)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # Stricter NMS for better performance
    
    # Use FP16 precision if available
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.FP16_ENABLED = True
    else:
        cfg.MODEL.DEVICE = "cpu"
    
    try:
        predictor = DefaultPredictor(cfg)
        return predictor, MetadataCatalog.get("coco_2017_val")
    except Exception as e:
        logger.error(f"Failed to initialize COCO predictor: {e}")
        raise

@torch.no_grad()
def setup_raft_model(use_small: bool = False) -> Tuple[torch.nn.Module, torch.device]:
    """Set up RAFT optical flow model with optimizations
    
    Args:
        use_small: Whether to use the small RAFT model for speed
        
    Returns:
        Tuple of model and device
    """
    logger.info(f"Setting up RAFT model ({'small' if use_small else 'large'})")
    
    # Use small model for speed if specified, otherwise use large model for accuracy
    if use_small:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights)
    else:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # JIT compilation removed - it was causing problems
    logger.info("Using RAFT model without JIT optimization")
    
    model.eval()
    return model, device

def calculate_mask_area(mask: np.ndarray) -> int:
    """Calculate area of a binary mask
    
    Args:
        mask: Binary mask
        
    Returns:
        Area of the mask
    """
    return np.sum(mask)

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union between two masks
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def warp_mask_with_flow(mask: np.ndarray, flow: torch.Tensor) -> np.ndarray:
    """Warp a mask according to optical flow using OpenCV
    
    Args:
        mask: Binary mask to warp
        flow: Optical flow tensor
        
    Returns:
        Warped binary mask
    """
    h, w = mask.shape
    
    # Generate coordinate grid with explicit indexing
    y_coords, x_coords = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing='ij'
    )

    # Add flow to coordinates
    flow_x = flow[0].cpu().numpy()
    flow_y = flow[1].cpu().numpy()

    # Calculate new coordinates after flow
    new_x = x_coords + flow_x
    new_y = y_coords + flow_y

    # Ensure coordinates are within bounds
    new_x = np.clip(new_x, 0, w - 1)
    new_y = np.clip(new_y, 0, h - 1)

    # Use OpenCV's remap for high-quality warping
    flow_map = np.stack([new_x, new_y], axis=-1)
    
    # Apply warping with bilinear interpolation
    try:
        warped_mask = cv2.remap(mask.astype(np.float32), flow_map, None, cv2.INTER_LINEAR)
        return warped_mask > 0.5  # Convert back to binary mask
    except cv2.error as e:
        logger.error(f"Error warping mask: {e}")
        return mask  # Return original mask on error

class BillboardTracker:
    """Track billboards using RAFT optical flow with improved tracking stability"""
    
    def __init__(self, raft_model: torch.nn.Module, device: torch.device, max_age: int = 10):
        """Initialize billboard tracker
        
        Args:
            raft_model: RAFT optical flow model
            device: Torch device
            max_age: Maximum number of frames to keep track without detection
        """
        self.raft_model = raft_model
        self.device = device
        self.max_age = max_age
        
        # Tracking state
        self.prev_frame = None
        self.billboard_tracks: List[BillboardTrack] = []
        self.next_id = 1
        
        # For debugging/visualization
        self.flow_vis = None
        
        # Store mask history for smoothing
        self.mask_history = {}  # track_id -> list of past masks
        self.mask_history_length = 3  # Store this many past masks
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert OpenCV BGR frame to RAFT input tensor
        
        Args:
            frame: OpenCV BGR frame
            
        Returns:
            Preprocessed tensor ready for RAFT
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        frame_tensor = to_tensor(frame_rgb).to(self.device)
        
        # Add batch dimension if needed
        if frame_tensor.dim() == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
            
        return frame_tensor
    
    @torch.no_grad()
    def calculate_flow(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Calculate optical flow between prev_frame and current frame
        
        Args:
            frame: Current frame
            
        Returns:
            Optical flow tensor or None for first frame
        """
        if self.prev_frame is None:
            # For first frame, just store and return None
            self.prev_frame = self.preprocess_frame(frame)
            return None
        
        # Preprocess current frame
        current_frame = self.preprocess_frame(frame)
        
        # Calculate flow using RAFT
        try:
            # Make the mixed precision code safer
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=True):
                    flow_predictions = self.raft_model(self.prev_frame, current_frame)
            else:
                flow_predictions = self.raft_model(self.prev_frame, current_frame)
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Handle OOM error by clearing cache and retrying with smaller batch
                logger.warning("GPU OOM error in RAFT. Clearing cache and retrying.")
                torch.cuda.empty_cache()
                # Try again with reduced size
                h, w = current_frame.shape[2:]
                scale_factor = 0.75
                resized_prev = torch.nn.functional.interpolate(
                    self.prev_frame, scale_factor=scale_factor, mode='bilinear')
                resized_curr = torch.nn.functional.interpolate(
                    current_frame, scale_factor=scale_factor, mode='bilinear')
                
                # Make the mixed precision code safer
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda", enabled=True):
                        flow_predictions = self.raft_model(resized_prev, resized_curr)
                else:
                    flow_predictions = self.raft_model(resized_prev, resized_curr)
                
                # Upscale flow back to original resolution
                flow = flow_predictions[-1][0]  # Get last prediction, first batch
                flow = torch.nn.functional.interpolate(
                    flow.unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)
                flow[0] *= (1/scale_factor)  # Scale flow values appropriately
                flow[1] *= (1/scale_factor)
            else:
                logger.error(f"Error calculating optical flow: {e}")
                # Return None to indicate flow calculation failure
                return None
        else:
            # Normal path (no errors)
            flow = flow_predictions[-1][0]  # Get last prediction, first batch
        
        # Store current frame for next iteration
        self.prev_frame = current_frame
        
        # Convert flow for visualization (for debugging)
        with torch.no_grad():
            self.flow_vis = flow_to_image(flow).cpu().numpy()
            self.flow_vis = np.transpose(self.flow_vis, (1, 2, 0))
        
        return flow
    
    def update_mask_history(self, track_id: int, mask: np.ndarray):
        """Update mask history for smooth transitions
        
        Args:
            track_id: Billboard track ID
            mask: New mask
        """
        if track_id not in self.mask_history:
            self.mask_history[track_id] = []
        
        # Add new mask to history
        self.mask_history[track_id].append(mask)
        
        # Keep only the most recent masks
        if len(self.mask_history[track_id]) > self.mask_history_length:
            self.mask_history[track_id].pop(0)
    
    def get_smoothed_mask(self, track_id: int) -> Optional[np.ndarray]:
        """Get smoothed mask from history using exponential decay
        
        Args:
            track_id: Billboard track ID
            
        Returns:
            Smoothed mask or None if no history
        """
        if track_id not in self.mask_history or not self.mask_history[track_id]:
            return None
            
        # Use exponential weighting for temporal smoothing
        # Most recent mask has highest weight
        masks = self.mask_history[track_id]
        weights = [0.5**i for i in range(len(masks))]
        weights.reverse()  # Highest weight for most recent mask
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Initialize smoothed mask with zeros
        smoothed_mask = np.zeros_like(masks[0], dtype=np.float32)
        
        # Apply weighted average
        for mask, weight in zip(masks, weights):
            smoothed_mask += weight * mask.astype(np.float32)
        
        # Convert back to binary mask
        return smoothed_mask > 0.5
    
    def track_billboards(self, frame: np.ndarray, billboard_detections: List[BillboardDetection]) -> List[BillboardTrack]:
        """Update billboard tracks with current detections and flow
        
        Args:
            frame: Current video frame
            billboard_detections: List of detected billboards
            
        Returns:
            List of active billboard tracks
        """
        # Calculate flow between frames
        with Timer("Flow calculation"):
            flow = self.calculate_flow(frame)
        
        # For first frame, just initialize tracks
        if flow is None:
            new_tracks = []
            for i, detection in enumerate(billboard_detections):
                # Create new track
                new_track = BillboardTrack(
                    id=self.next_id,
                    mask=detection.mask,
                    score=detection.score,
                    box=detection.box,
                    age=0,
                    time_since_detection=0,
                    color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                )
                
                # Initialize mask history
                self.update_mask_history(new_track.id, detection.mask)
                
                new_tracks.append(new_track)
                self.next_id += 1
            
            self.billboard_tracks = new_tracks
            return self.billboard_tracks
        
        # Predict new positions of existing tracks using optical flow
        with Timer("Track prediction"):
            predicted_tracks = []
            for track in self.billboard_tracks:
                # Warp mask using optical flow
                warped_mask = warp_mask_with_flow(track.mask, flow)
                
                # Update track with warped mask
                updated_track = BillboardTrack(
                    id=track.id,
                    mask=warped_mask,
                    score=track.score,
                    box=track.box,  # Will be updated below
                    age=track.age + 1,
                    time_since_detection=track.time_since_detection + 1,
                    color=track.color
                )
                
                # Calculate new bounding box from warped mask
                y_indices, x_indices = np.where(warped_mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x1, y1 = np.min(x_indices), np.min(y_indices)
                    x2, y2 = np.max(x_indices), np.max(y_indices)
                    updated_track.box = (x1, y1, x2 - x1, y2 - y1)
                
                predicted_tracks.append(updated_track)
        
        # Match current detections with predicted tracks
        with Timer("Track matching"):
            unmatched_tracks = list(range(len(predicted_tracks)))
            unmatched_detections = list(range(len(billboard_detections)))
            matches = []
            
            # Calculate IoU between each detection and track
            if predicted_tracks and billboard_detections:
                iou_matrix = np.zeros((len(billboard_detections), len(predicted_tracks)))
                
                for d_idx, detection in enumerate(billboard_detections):
                    for t_idx, track in enumerate(predicted_tracks):
                        iou_matrix[d_idx, t_idx] = calculate_iou(detection.mask, track.mask)
                
                # Find matches using greedy algorithm with improved IoU threshold
                while True:
                    # Find highest IoU
                    max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0
                    if max_iou < 0.3:  # Minimum IoU threshold
                        break
                    
                    # Get indices of max IoU
                    d_idx, t_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                    
                    # Add to matches
                    matches.append((d_idx, t_idx))
                    
                    # Mark as matched
                    unmatched_detections.remove(d_idx)
                    unmatched_tracks.remove(t_idx)
                    
                    # Set IoU to zero to prevent rematch
                    iou_matrix[d_idx, :] = 0
                    iou_matrix[:, t_idx] = 0
        
        # Update matched tracks
        result_tracks = []
        
        # Process matched tracks first
        for d_idx, t_idx in matches:
            detection = billboard_detections[d_idx]
            track = predicted_tracks[t_idx]
            
            # ANTI-FLICKER: Smooth transition between track and detection
            # Update mask history with new detection
            self.update_mask_history(track.id, detection.mask)
            
            # Get temporally smoothed mask
            smoothed_mask = self.get_smoothed_mask(track.id)
            if smoothed_mask is None:
                # Fallback if no history
                smoothed_mask = detection.mask
            
            # Create updated track with smoothed mask
            updated_track = BillboardTrack(
                id=track.id,
                mask=smoothed_mask,  # Use smoothed mask
                score=detection.score,  # Use new score
                box=detection.box,     # Use new box
                age=track.age,         # Keep age counter
                time_since_detection=0,  # Reset detection counter
                color=track.color      # Keep color
            )
            
            result_tracks.append(updated_track)
        
        # Process unmatched tracks (but reduce their score)
        for t_idx in unmatched_tracks:
            track = predicted_tracks[t_idx]
            
            # Apply exponential decay to confidence score
            decay_factor = 0.85  # Faster decay for better track termination
            new_score = track.score * decay_factor
            
            # Keep track if it's still reliable
            if track.time_since_detection < self.max_age and new_score > 0.3:
                # Update track with decayed score
                updated_track = BillboardTrack(
                    id=track.id,
                    mask=track.mask,
                    score=new_score,
                    box=track.box,
                    age=track.age,
                    time_since_detection=track.time_since_detection,
                    color=track.color
                )
                result_tracks.append(updated_track)
            else:
                # Remove mask history for tracks we're discarding
                if track.id in self.mask_history:
                    del self.mask_history[track.id]
        
        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            detection = billboard_detections[d_idx]
            
            # Only create new tracks if detection is confident enough
            if detection.score > 0.4:  # Higher threshold for new tracks
                # Create new track
                new_track = BillboardTrack(
                    id=self.next_id,
                    mask=detection.mask,
                    score=detection.score,
                    box=detection.box,
                    age=0,
                    time_since_detection=0,
                    color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                )
                
                # Initialize mask history
                self.update_mask_history(new_track.id, detection.mask)
                
                result_tracks.append(new_track)
                self.next_id += 1
        
        # Update billboard tracks
        self.billboard_tracks = result_tracks
        return result_tracks

def load_replacement_image(image_path: str) -> np.ndarray:
    """Load and preprocess replacement image with improved error handling
    
    Args:
        image_path: Path to replacement image
        
    Returns:
        Loaded image or default image if loading fails
    """
    # Create a default replacement image
    def create_default_image(width=300, height=100):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, "REPLACEMENT AD", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img
    
    if not image_path or not os.path.exists(image_path):
        logger.warning(f"Replacement image not found at {image_path}")
        return create_default_image()
    
    # Load image
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Basic image validation
        if img.shape[0] < 10 or img.shape[1] < 10:
            raise ValueError(f"Image too small: {img.shape}")
        
        # Basic preprocessing: ensure 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
        
    except Exception as e:
        logger.error(f"Error loading replacement image: {e}")
        return create_default_image()

class BillboardReplacer:
    """Class for replacing billboards in video frames"""
    
    def __init__(self, replacement_img: np.ndarray):
        """Initialize with replacement image
        
        Args:
            replacement_img: Image to use for billboard replacement
        """
        self.replacement_img = replacement_img
        self.last_transform = {}  # Store last perspective transform per billboard ID
        
        # Optional: prepare alternative versions of the replacement image for variety
        self.replacement_variants = []
        
        # Create variations with different brightness/contrast (optional)
        for i in range(3):
            variant = replacement_img.copy()
            # Adjust brightness slightly
            factor = 0.9 + i * 0.1  # 0.9, 1.0, 1.1
            variant = cv2.convertScaleAbs(variant, alpha=factor, beta=5)
            self.replacement_variants.append(variant)
    
    def get_replacement_for_track(self, track_id: int) -> np.ndarray:
        """Get appropriate replacement image for a track
        
        Args:
            track_id: ID of the billboard track
            
        Returns:
            Replacement image for this track
        """
        # Use track ID to select a consistent variant for this billboard
        if not self.replacement_variants:
            return self.replacement_img
            
        variant_idx = track_id % len(self.replacement_variants)
        return self.replacement_variants[variant_idx]
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in top-left, top-right, bottom-right, bottom-left order
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered points
        """
        # Sort by sum of coordinates (smallest sum is top-left, largest is bottom-right)
        sum_pts = pts.sum(axis=1)
        rect = np.zeros((4, 2), dtype=pts.dtype)
        
        # Top-left point will have smallest sum
        rect[0] = pts[np.argmin(sum_pts)]
        # Bottom-right point will have largest sum
        rect[2] = pts[np.argmax(sum_pts)]
        
        # Calculate difference between coordinates
        diff = np.diff(pts, axis=1)
        # Top-right will have smallest difference (y - x)
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left will have largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _filter_transform_matrix(self, M, weight=0.7):
        """Smooth transform matrix to reduce flicker
        
        Args:
            M: Transform matrix
            weight: Weight for current matrix (0.0-1.0)
            
        Returns:
            Filtered transform matrix
        """
        # No need to filter if matrix is None or weight is 1.0
        if M is None:
            return M
        
        return M
    
    def replace_billboard(self, 
                        frame: np.ndarray, 
                        track: BillboardTrack, 
                        foreground_mask: Optional[np.ndarray] = None,
                        respect_foreground: bool = True) -> np.ndarray:
        """Replace billboard in frame with replacement image using perspective transform
        
        Args:
            frame: Video frame
            track: Billboard track
            foreground_mask: Mask of foreground objects
            respect_foreground: Whether to respect foreground objects when replacing
            
        Returns:
            Frame with replaced billboard
        """
        # Skip if mask is empty
        mask = track.mask
        if not np.any(mask):
            return frame
        
        # Create output frame
        result = frame.copy()
        
        # Get mask as uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours of the mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame
            
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Minimum points check
        if len(contour) < 4:
            return frame
        
        # Get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        
        # Minimum size check
        if w < 10 or h < 10:
            return frame
        
        # Get replacement image appropriate for this billboard
        replacement_img = self.get_replacement_for_track(track.id)
        
        # Try to use perspective transform for better visual quality and stability
        try:
            # Try to approximate a quadrilateral
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If we have a reasonable quadrilateral, use perspective transform
            if len(approx) >= 4 and len(approx) <= 6:
                # Get convex hull to ensure proper ordering
                hull = cv2.convexHull(approx)
                
                if len(hull) == 4:
                    # We have a proper quadrilateral
                    src_pts = hull.reshape(-1, 2).astype(np.float32)
                    
                    # Make sure points are in correct order: top-left, top-right, bottom-right, bottom-left
                    src_pts = self._order_points(src_pts)
                    
                    # FLICKER REDUCTION: If we have a previous transform for this billboard, use interpolation
                    # to smooth transition between transforms
                    if track.id in self.last_transform:
                        prev_pts = self.last_transform[track.id]
                        # Blend previous and current points to reduce jitter (80% current, 20% previous)
                        # Use less previous data for faster reactions but smoother transitions
                        alpha = 0.8
                        src_pts = alpha * src_pts + (1 - alpha) * prev_pts
                    
                    # Store current transform for next frame
                    self.last_transform[track.id] = src_pts.copy()
                    
                    # Set destination points (target size of the replacement image)
                    dst_pts = np.array([
                        [0, 0],
                        [replacement_img.shape[1] - 1, 0],
                        [replacement_img.shape[1] - 1, replacement_img.shape[0] - 1],
                        [0, replacement_img.shape[0] - 1]
                    ], dtype=np.float32)
                    
                    # Calculate perspective transform
                    M = cv2.getPerspectiveTransform(dst_pts, src_pts)
                    
                    # Warp replacement image to fit the billboard
                    warped_replacement = cv2.warpPerspective(
                        replacement_img, 
                        M, 
                        (frame.shape[1], frame.shape[0]),
                        borderMode=cv2.BORDER_TRANSPARENT
                    )
                    
                    # Create mask for warped replacement
                    warped_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    cv2.fillPoly(warped_mask, [hull], 255)
                    
                    # Apply foreground mask if needed
                    if respect_foreground and foreground_mask is not None:
                        # Only replace where mask is active AND there is no foreground
                        valid_mask = (warped_mask == 255) & (foreground_mask == 0)
                    else:
                        valid_mask = (warped_mask == 255)
                    
                    # Apply replacement using mask
                    for c in range(3):
                        result[:, :, c][valid_mask] = warped_replacement[:, :, c][valid_mask]
                    
                    return result
        except Exception as e:
            logger.debug(f"Perspective transform failed: {e}. Falling back to regular replacement.")
            # Fall back to regular replacement
            pass
        
        # STANDARD REPLACEMENT: Fall back to simple replacement if perspective transform fails
        # Prepare the replacement image
        replacement_resized = cv2.resize(replacement_img, (w, h))
        
        # Create mask for the region
        region_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour.copy()
        shifted_contour[:, :, 0] = contour[:, :, 0] - x
        shifted_contour[:, :, 1] = contour[:, :, 1] - y
        cv2.drawContours(region_mask, [shifted_contour], 0, 255, -1)
        
        # Apply the replacement image within the mask
        for c in range(3):
            # Extract the region from the frame
            region = result[y:y+h, x:x+w, c].copy()
            
            # Apply replacement only where the mask is active
            if respect_foreground and foreground_mask is not None:
                # Get foreground in the region
                foreground_region = foreground_mask[y:y+h, x:x+w]
                # Only replace where the mask is active AND there is no foreground
                valid_mask = (region_mask == 255) & (foreground_region == 0)
                region[valid_mask] = replacement_resized[:, :, c][valid_mask]
            else:
                # Only replace where the mask is active
                valid_mask = (region_mask == 255)
                region[valid_mask] = replacement_resized[:, :, c][valid_mask]
                
            # Put the region back to the result
            result[y:y+h, x:x+w, c] = region
        
        return result

class VideoProcessor:
    """Main class for processing videos with billboard replacement"""
    
    def __init__(self, args):
        """Initialize video processor with command-line arguments
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.billboard_predictor = None
        self.coco_predictor = None
        self.billboard_metadata = None
        self.coco_metadata = None
        self.raft_model = None
        self.device = None
        self.replacement_img = None
        self.billboard_tracker = None
        self.billboard_replacer = None
        
        # Set default values (used as fallbacks on error)
        if not hasattr(self.args, 'max_age'):
            self.args.max_age = 5
        if not hasattr(self.args, 'show_frame_num'):
            self.args.show_frame_num = False
        if not hasattr(self.args, 'show_ids'):
            self.args.show_ids = False
        if not hasattr(self.args, 'show_flow'):
            self.args.show_flow = False
        if not hasattr(self.args, 'show_foreground'):
            self.args.show_foreground = False
        if not hasattr(self.args, 'show_outlines'):
            self.args.show_outlines = False
        if not hasattr(self.args, 'respect_foreground'):
            self.args.respect_foreground = False
        if not hasattr(self.args, 'mask_mode'):
            self.args.mask_mode = False
    
    def setup_models(self):
        """Set up all required models and resources"""
        logger.info("Setting up models and resources...")
        
        # Load billboard model
        self.billboard_predictor, self.billboard_metadata = setup_billboard_predictor(
            self.args.billboard_model, conf_thresh=self.args.confidence)
        
        # Load COCO model
        self.coco_predictor, self.coco_metadata = setup_coco_predictor(
            conf_thresh=self.args.coco_conf)
        
        # Load RAFT model
        self.raft_model, self.device = setup_raft_model(use_small=self.args.fast_mode)
        
        # Load replacement image
        self.replacement_img = load_replacement_image(self.args.replacement_image)
        
        # Create billboard tracker
        self.billboard_tracker = BillboardTracker(self.raft_model, self.device, max_age=self.args.max_age)
        
        # Create billboard replacer
        self.billboard_replacer = BillboardReplacer(self.replacement_img)
        
        logger.info("Models and resources loaded successfully")
    
    def detect_billboards(self, frame):
        """Detect billboards in a frame
        
        Args:
            frame: Video frame
            
        Returns:
            List of billboard detections
        """
        try:
            # Make the mixed precision code safer
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", enabled=True):
                    outputs = self.billboard_predictor(frame)
            else:
                outputs = self.billboard_predictor(frame)
            
            instances = outputs["instances"].to("cpu")
            billboard_detections = []
            
            if instances.has("pred_masks") and len(instances) > 0:
                # Extract billboard predictions
                masks = instances.pred_masks.numpy()
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                
                for mask, box, score in zip(masks, boxes, scores):
                    x1, y1, x2, y2 = box
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    billboard_detections.append(BillboardDetection(
                        mask=mask,
                        box=(x, y, w, h),
                        score=score
                    ))
            
            return billboard_detections
            
        except Exception as e:
            logger.error(f"Error detecting billboards: {e}")
            return []
    
    def detect_foreground(self, frame):
        """Detect people and sports balls as foreground
        
        Args:
            frame: Video frame
            
        Returns:
            Foreground mask and instances
        """
        try:
            # Make the mixed precision code safer
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", enabled=True):
                    outputs = self.coco_predictor(frame)
            else:
                outputs = self.coco_predictor(frame)
            
            instances = outputs["instances"].to("cpu")
            
            # Filter to only include people and sports balls
            if len(instances) > 0:
                # Get indices of people and sports balls
                classes = instances.pred_classes.numpy()
                # Keep only person (0) and sports ball (32) in detectron2 indexing
                keep_indices = [i for i, c in enumerate(classes) if c == 0 or c == 32]
                
                foreground_instances = instances[keep_indices]
            else:
                foreground_instances = instances
            
            # Create foreground mask
            h, w = frame.shape[:2]
            foreground_mask = np.zeros((h, w), dtype=np.uint8)
            
            if foreground_instances.has("pred_masks") and len(foreground_instances) > 0:
                # Extract all foreground masks and combine them
                for mask in foreground_instances.pred_masks.numpy():
                    foreground_mask |= (mask * 255).astype(np.uint8)
            
            return foreground_mask, foreground_instances
            
        except Exception as e:
            logger.error(f"Error detecting foreground: {e}")
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), None
    
    def process_video(self):
        """Process input video and generate output video with billboard replacement"""
        # Set up all required models
        setup_start = time.time()
        self.setup_models()
        setup_time = time.time() - setup_start
        logger.info(f"Setup completed in {setup_time:.2f} seconds")
        
        # Open input video
        try:
            cap = cv2.VideoCapture(self.args.input_video)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video {self.args.input_video}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.args.output_video) or '.', exist_ok=True)
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.args.output_video, fourcc, fps, (width, height))
            
            # Process frames
            frame_count = 0
            processing_start = time.time()
            
            # Use tqdm for progress bar if available
            iterator = tqdm(total=total_frames) if TQDM_AVAILABLE else None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Update progress bar
                if iterator:
                    iterator.update(1)
                elif frame_count % 20 == 0:
                    elapsed = time.time() - processing_start
                    fps_avg = frame_count / elapsed
                    eta = (total_frames - frame_count) / fps_avg if fps_avg > 0 else 0
                    logger.info(f"Frame {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%) - {fps_avg:.2f} FPS - ETA: {eta/60:.1f} min")
                
                # Process frame
                try:
                    result_frame = self.process_frame(frame, frame_count)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    result_frame = frame  # Use original frame on error
                
                # Write output frame
                out.write(result_frame)
            
            # Close progress bar
            if iterator:
                iterator.close()
            
            # Clean up
            cap.release()
            out.release()
            
            # Print summary
            total_time = time.time() - setup_start
            processing_time = time.time() - processing_start
            logger.info(f"Processing complete!")
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Setup time: {setup_time:.2f} seconds")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"Average FPS: {frame_count / processing_time:.2f}")
            logger.info(f"Total unique billboards detected: {self.billboard_tracker.next_id - 1}")
            logger.info(f"Output written to: {self.args.output_video}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False
    
    def process_frame(self, frame, frame_count):
        """Process a single frame
        
        Args:
            frame: Video frame
            frame_count: Current frame number
            
        Returns:
            Processed frame with replaced billboards
        """
        # Create a copy of the frame for results
        result_frame = frame.copy()
        
        # Detect billboards
        with Timer("Billboard detection"):
            billboard_detections = self.detect_billboards(frame)
        
        # Detect foreground (people and sports balls)
        with Timer("Foreground detection"):
            foreground_mask, foreground_instances = self.detect_foreground(frame)
        
        # Update tracker
        with Timer("Billboard tracking"):
            active_billboards = self.billboard_tracker.track_billboards(frame, billboard_detections)
        
        # Process billboards based on mode
        if self.args.mask_mode:
            # Only visualize billboard masks with colors
            for track in active_billboards:
                # Only visualize if confidence is high enough
                if track.score > 0.4:
                    mask = track.mask
                    color = track.color
                    
                    # Apply the mask with color (random for each billboard)
                    mask_bool = mask.astype(bool)
                    
                    # Remove foreground from mask if requested
                    if self.args.respect_foreground:
                        visible_mask = mask_bool & (foreground_mask == 0)
                    else:
                        visible_mask = mask_bool
                    
                    # Apply color with alpha blending
                    for c in range(3):
                        result_frame[:, :, c][visible_mask] = (
                            0.1 * result_frame[:, :, c][visible_mask] + 
                            0.5 * color[c]
                        )
                    
                    # Add ID text if requested
                    if self.args.show_ids:
                        x, y, w, h = track.box
                        cv2.putText(result_frame, f"ID: {track.id} ({track.score:.2f})", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Replace billboards with advertisement
            for track in active_billboards:
                # Replace only if confidence is high enough
                if track.score > 0.4:
                    # Apply replacement
                    result_frame = self.billboard_replacer.replace_billboard(
                        result_frame, 
                        track, 
                        foreground_mask if self.args.respect_foreground else None,
                        self.args.respect_foreground
                    )
                    
                    # Add ID text if requested
                    if self.args.show_ids:
                        x, y, w, h = track.box
                        cv2.putText(result_frame, f"ID: {track.id} ({track.score:.2f})", 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2)
        
        # If requested, show flow visualization
        if self.args.show_flow and self.billboard_tracker.flow_vis is not None:
            flow_vis = self.billboard_tracker.flow_vis
            # Fixed scaling code
            flow_vis = cv2.resize(flow_vis, (frame.shape[1] // 4, frame.shape[0] // 4))
            result_frame[0:flow_vis.shape[0], 0:flow_vis.shape[1]] = flow_vis
        
        # If requested, visualize foreground objects
        if self.args.show_foreground and not self.args.show_outlines:
            # Add a small semi-transparent overlay on foreground objects
            foreground_overlay = np.zeros_like(result_frame)
            foreground_overlay[:, :, 1] = 30  # Green tint
            
            # Apply overlay only to foreground areas
            alpha = 0.3
            foreground_bool = (foreground_mask > 0)
            for c in range(3):
                result_frame[:, :, c][foreground_bool] = (
                    (1 - alpha) * result_frame[:, :, c][foreground_bool] + 
                    alpha * foreground_overlay[:, :, c][foreground_bool]
                )
        
        # If specifically requested, show foreground outlines
        if self.args.show_outlines and foreground_instances is not None and foreground_instances.has("pred_masks") and len(foreground_instances) > 0:
            person_color = (0, 255, 0)  # Green for people
            ball_color = (0, 165, 255)  # Orange for balls
            
            masks = foreground_instances.pred_masks.numpy()
            classes = foreground_instances.pred_classes.numpy()
            
            for i, (mask, cls) in enumerate(zip(masks, classes)):
                # Convert mask to contour
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contour with appropriate color
                color = person_color if cls == 0 else ball_color
                cv2.drawContours(result_frame, contours, -1, color, 2)
        
        # Add frame number to visualization
        if self.args.show_frame_num:
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result_frame

def main(args):
    """Main function to process video
    
    Args:
        args: Command-line arguments
        
    Returns:
        0 on success, 1 on failure
    """
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create video processor
    processor = VideoProcessor(args)
    
    # Process video
    result = processor.process_video()
    
    # Return success status
    return 0 if result else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Billboard detection and replacement with RAFT optical flow tracking")
    parser.add_argument("--billboard-model", default="model_final.pth", help="Path to billboard model weights (custom Detectron2 billboard model, see docs/assets.md)")
    parser.add_argument("--input-video", default="data/adVideo1.mp4", help="Path to input video (sample clip, see repo docs/assets.md)")
    parser.add_argument("--output-video", default="output/result.mp4", help="Path to output video")
    parser.add_argument("--replacement-image", default="data/replace.jpg", help="Path to replacement advertisement image")
    parser.add_argument("--confidence", type=float, default=0.5, help="Billboard confidence threshold (default: 0.5)")
    parser.add_argument("--coco-conf", type=float, default=0.5, help="COCO model confidence threshold (default: 0.5)")
    parser.add_argument("--max-age", type=int, default=5, help="Maximum frames to keep lost tracks (default: 5)")
    parser.add_argument("--mask-mode", action="store_true", help="Only show colored masks instead of replacement")
    parser.add_argument("--respect-foreground", action="store_true", help="Respect foreground objects when replacing")
    parser.add_argument("--show-foreground", action="store_true", help="Show foreground objects")
    parser.add_argument("--show-outlines", action="store_true", help="Show outlines of foreground objects")
    parser.add_argument("--show-flow", action="store_true", help="Show optical flow visualization")
    parser.add_argument("--show-ids", action="store_true", help="Show billboard IDs and confidence")
    parser.add_argument("--show-frame-num", action="store_true", help="Show frame number")
    parser.add_argument("--fast-mode", action="store_true", help="Use smaller RAFT model for faster processing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing (experimental)")
    
    args = parser.parse_args()
    sys.exit(main(args))