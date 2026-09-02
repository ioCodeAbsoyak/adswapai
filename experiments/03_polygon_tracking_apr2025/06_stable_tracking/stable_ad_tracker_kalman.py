"""AdSwapAI R&D, 2025-04-13: StableAdTracker - per-corner Kalman filter, IoU matching,
coasting when detections are missed, and running the detector only every N frames."""

import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict

VIDEO_PATH = "data/adVideo2.mp4"   # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"
MODEL_PATH = "best.pt"   # single-class YOLOv8s-seg "billboard" model, see docs/assets.md

# Draw debug overlays (green tracked polygons/IDs, blue raw detections) into the output video
DRAW_DEBUG = False

class StableAdTracker:
    """
    Stable advertisement tracker with Kalman filtering and temporal smoothing
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.5, smoothing_factor=0.1):
        self.max_age = max_age  # Maximum frames to keep without detection
        self.min_hits = min_hits  # Minimum hits to consider as confirmed
        self.iou_threshold = iou_threshold  # IOU threshold for matching
        self.smoothing_factor = smoothing_factor  # Lower = more stable but slower response

        self.trackers = []  # Active trackers
        self.next_id = 1    # ID counter for new trackers

        # Detection frequency control
        self.detection_interval = 10  # Run detection every N frames
        self.frame_count = 0

    def init_kalman(self):
        """Initialize a Kalman filter for quad point tracking"""
        # 16 state variables (8 points, each with x,y) and 8 measurement variables
        kalman = cv2.KalmanFilter(16, 8)

        # State transition matrix - mostly identity with velocity prediction
        kalman.transitionMatrix = np.eye(16, dtype=np.float32)

        # Add position delta predictions (simplified constant velocity model)
        for i in range(8):
            kalman.transitionMatrix[i, i+8] = 1.0

        # Measurement matrix - we only measure the positions, not velocities
        kalman.measurementMatrix = np.zeros((8, 16), dtype=np.float32)
        for i in range(8):
            kalman.measurementMatrix[i, i] = 1.0

        # Process noise covariance - critical for stability
        # **MODIFIED:** Reduced process noise assuming more stable objects
        kalman.processNoiseCov = np.eye(16, dtype=np.float32) * 0.0005

        # Measurement noise covariance - increase for smoother tracking
        # **MODIFIED:** Increased measurement noise to trust filter prediction more
        kalman.measurementNoiseCov = np.eye(8, dtype=np.float32) * 1.5

        # Error covariance
        kalman.errorCovPost = np.eye(16, dtype=np.float32) * 1.0

        return kalman

    def create_tracker(self, poly, track_id):
        """Create a new tracker with given polygon and ID"""
        # Order polygon points consistently
        poly = self.order_points(poly)

        # Initialize Kalman filter
        kalman = self.init_kalman()

        # Flatten polygon to state vector [x1,y1,x2,y2,x3,y3,x4,y4]
        flattened = poly.reshape(-1)

        # Initialize state with flattened polygon and zero velocities
        state = np.zeros(16, dtype=np.float32)
        state[:8] = flattened
        kalman.statePost = state.reshape((16, 1))

        # Create tracker object
        tracker = {
            'id': track_id,
            'kalman': kalman,
            'polygon': poly.copy(),  # Current smoothed polygon
            'age': 0,                # Age in frames
            'hits': 1,               # Number of detections
            'time_since_update': 0,  # Frames since last update
            'original_size': self.polygon_size(poly),  # Original size for consistency check
            'history': [poly.copy()],  # History of SMOOTHED positions
        }

        return tracker

    def order_points(self, pts):
        """Order polygon points: top-left, top-right, bottom-right, bottom-left"""
        if isinstance(pts, list):
            pts = np.array(pts, dtype=np.float32)

        # Ensure we have a 4-point polygon
        if pts.shape[0] != 4:
            # If not 4 points, generate minimum bounding rectangle
            try:
                rect = cv2.minAreaRect(pts)
                pts = cv2.boxPoints(rect).astype(np.float32)
            except Exception: # Handle cases where minAreaRect fails (e.g., colinear points)
                 # Return a default rectangle if points are invalid for minAreaRect
                 return np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


        # Safety check for empty arrays or invalid shapes
        if pts.size == 0 or pts.shape != (4, 2):
            # Return a default rectangle
            return np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        # Order the points consistently
        rect = np.zeros((4, 2), dtype=np.float32)

        # Find top-left (smallest sum) and bottom-right (largest sum)
        s = pts.sum(axis=1)
        if s.size > 0:
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            # Find top-right (smallest difference) and bottom-left (largest difference)
            diff = np.diff(pts, axis=1)
            # Check diff shape before accessing indices
            if diff.size > 0 and diff.shape[0] == pts.shape[0]:
                 rect[1] = pts[np.argmin(diff)]
                 rect[3] = pts[np.argmax(diff)]
            elif pts.shape[0] == 4: # Fallback if diff calculation fails but we have 4 points
                 # A simple fallback (might not be perfect for all rotations)
                 remaining_pts = [p for p in pts.tolist() if p != rect[0].tolist() and p != rect[2].tolist()]
                 if len(remaining_pts) == 2:
                     # Assign based on x-coordinate relative to top-left
                     if remaining_pts[0][0] > remaining_pts[1][0]:
                         rect[1] = np.array(remaining_pts[0])
                         rect[3] = np.array(remaining_pts[1])
                     else:
                         rect[1] = np.array(remaining_pts[1])
                         rect[3] = np.array(remaining_pts[0])
                 else: # If fallback fails, use default
                     return np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

            else:
                # Fallback if diff is empty or shape mismatch
                return np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        else:
            # Fallback if s is empty
            return np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        return rect

    def polygon_iou(self, poly1, poly2):
        """Calculate IoU between two polygons"""
        try:
            # Safety check for valid polygons
            if poly1.shape != (4, 2) or poly2.shape != (4, 2):
                return 0

            # Check for NaN or Inf values
            if not np.all(np.isfinite(poly1)) or not np.all(np.isfinite(poly2)):
                return 0

            # Convert to int32 for contourArea and intersectConvexConvex functions
            poly1_int = poly1.astype(np.int32)
            poly2_int = poly2.astype(np.int32)

            # Calculate areas
            area1 = cv2.contourArea(poly1_int)
            area2 = cv2.contourArea(poly2_int)

            # If either area is too small or non-convex (negative area), return 0
            if area1 <= 10 or area2 <= 10: # Increased threshold slightly
                return 0

            # Calculate intersection
            inter_area = 0
            # intersectConvexConvex requires convex polygons
            # Ensure polygons are reasonably convex before proceeding
            if cv2.isContourConvex(poly1_int) and cv2.isContourConvex(poly2_int):
                try:
                    # Use intersectConvexConvex if polygons are convex
                    inter_area, _ = cv2.intersectConvexConvex(poly1_int, poly2_int)
                except Exception as e:
                    # print(f"IoU intersection error: {e}") # Optional: for debugging
                    return 0 # Return 0 if intersection fails
            else:
                # Fallback for non-convex (though ideally detections should be convex)
                # This is computationally more expensive and might be less accurate
                # Consider if this fallback is necessary or if non-convex shapes indicate bad detections
                # For simplicity, we can return 0 if not convex, assuming good detections
                return 0


            # Calculate union
            union_area = area1 + area2 - inter_area
            if union_area <= 0:
                return 0

            return max(0.0, min(1.0, inter_area / union_area)) # Clamp IoU between 0 and 1

        except Exception as e:
            # print(f"IoU calculation error: {e}") # Optional: for debugging
            return 0

    def polygon_size(self, poly):
        """Calculate size (area) of a polygon"""
        try:
            # Ensure valid shape before calculating area
            if poly.shape == (4, 2):
                return cv2.contourArea(poly.astype(np.int32))
            else:
                return 0
        except Exception:
             return 0 # Return 0 if contourArea fails

    def is_valid_polygon(self, poly, frame_shape):
        """Check if polygon is valid and within frame bounds"""
        # Check if polygon has 4 points and correct shape
        if poly.shape != (4, 2):
            return False

        # Check for NaN or Inf values
        if not np.all(np.isfinite(poly)):
            return False

        # Check if polygon is within frame bounds with some margin
        h, w = frame_shape[:2]
        margin = 5

        # Check all points are within frame
        for x, y in poly:
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                return False

        # Check for reasonable area
        area = self.polygon_size(poly)
        if area < 100 or area > 0.7 * w * h: # Min area 100, max 70% of frame
            return False

        # Check for reasonable aspect ratio (not too stretched)
        # Calculate width and height from bounding rectangle for robustness
        try:
            rect = cv2.minAreaRect(poly.astype(np.float32))
            (width, height) = rect[1] # rect[1] contains (width, height)

            if width < 5 or height < 5:
                return False

            # Avoid division by zero
            if height == 0 or width == 0:
                return False

            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > 10:  # Extremely stretched polygons are likely errors
                return False
        except Exception:
            return False # Invalid if minAreaRect fails


        # Check for convexity
        if not cv2.isContourConvex(poly.astype(np.int32)):
             return False

        return True

    def update(self, detections, frame):
        """
        Update trackers with new detections

        Args:
            detections: List of polygons from detector
            frame: Current frame for size reference
        """
        self.frame_count += 1
        frame_shape = frame.shape

        # Predict new locations for all trackers
        for tracker in self.trackers:
            # Update tracking metrics
            tracker['age'] += 1
            tracker['time_since_update'] += 1

            try:
                # Predict using Kalman filter
                prediction = tracker['kalman'].predict()

                # Extract polygon from prediction
                pred_poly = prediction[:8].reshape(4, 2)

                # Verify prediction is reasonable before updating the tracker's main polygon
                if self.is_valid_polygon(pred_poly, frame_shape):
                    # Update the polygon based on prediction ONLY IF no detection is matched later
                    # We store the prediction temporarily, the actual update happens after matching
                    tracker['predicted_polygon'] = pred_poly
                else:
                    # If prediction is invalid, keep the last known good polygon
                    tracker['predicted_polygon'] = tracker['polygon']

            except Exception as e:
                print(f"Warning: Kalman prediction error for tracker {tracker['id']}: {e}")
                # If prediction fails, keep the last known good polygon
                tracker['predicted_polygon'] = tracker['polygon']


        # Remove dead trackers first
        active_trackers = []
        for tracker in self.trackers:
            if tracker['time_since_update'] <= self.max_age:
                active_trackers.append(tracker)
            # else: # Optional: print when a tracker is removed
            #     print(f"Removing tracker {tracker['id']} due to age.")
        self.trackers = active_trackers

        # If no detections this frame, update trackers with predictions
        if not detections or len(detections) == 0:
             for tracker in self.trackers:
                  # Update the main polygon with the predicted one
                  tracker['polygon'] = tracker.get('predicted_polygon', tracker['polygon'])
                  # Add the predicted (now current) polygon to history
                  tracker['history'].append(tracker['polygon'].copy())
                  # Keep history limited
                  if len(tracker['history']) > 30:
                       tracker['history'] = tracker['history'][-30:]
             # Return confirmed trackers based on their current state
             return [t for t in self.trackers if t['hits'] >= self.min_hits]


        # --- Matching Process ---
        # Prepare detections (order points and filter invalid ones)
        valid_detections = []
        for det in detections:
             ordered_poly = self.order_points(det)
             if self.is_valid_polygon(ordered_poly, frame_shape):
                 valid_detections.append(ordered_poly)
             # else: # Optional: print invalid detections
             #      print(f"Skipping invalid detection polygon.")

        if not valid_detections:
             # If all detections were invalid, behave as if no detections
             for tracker in self.trackers:
                  tracker['polygon'] = tracker.get('predicted_polygon', tracker['polygon'])
                  tracker['history'].append(tracker['polygon'].copy())
                  if len(tracker['history']) > 30: tracker['history'] = tracker['history'][-30:]
             return [t for t in self.trackers if t['hits'] >= self.min_hits]


        matched_indices = []
        unmatched_detections = list(range(len(valid_detections)))
        unmatched_trackers = list(range(len(self.trackers)))

        # If no trackers yet, all valid detections are unmatched
        if len(self.trackers) > 0:
            # Calculate IoU matrix between valid detections and tracker PREDICTIONS
            iou_matrix = np.zeros((len(valid_detections), len(self.trackers)))
            for d, det_poly in enumerate(valid_detections):
                for t, tracker in enumerate(self.trackers):
                    # Use the predicted polygon for matching
                    tracker_poly_for_match = tracker.get('predicted_polygon', tracker['polygon'])
                    iou_matrix[d, t] = self.polygon_iou(det_poly, tracker_poly_for_match)


            # Use greedy matching (or Hungarian algorithm for optimal assignment)
            # Greedy matching:
            while True:
                if not unmatched_detections or not unmatched_trackers or iou_matrix.size == 0:
                    break

                # Find best match among remaining
                max_iou = -1
                best_match = (-1, -1)
                temp_unmatched_detections = list(unmatched_detections) # Create copies to iterate over
                temp_unmatched_trackers = list(unmatched_trackers)

                for d in temp_unmatched_detections:
                    for t in temp_unmatched_trackers:
                         # Check if indices are still valid in the current matrix shape
                         if d < iou_matrix.shape[0] and t < iou_matrix.shape[1]:
                            if iou_matrix[d, t] > max_iou:
                                max_iou = iou_matrix[d, t]
                                best_match = (d, t)

                # If best match is below threshold, stop matching
                if max_iou < self.iou_threshold:
                    break

                # Mark as matched
                d, t = best_match
                matched_indices.append((d, t))
                unmatched_detections.remove(d)
                unmatched_trackers.remove(t)

                # Remove row and column from consideration for next iteration (set to -1)
                # It's often easier to just iterate through remaining indices like above
                # than modifying the matrix in place during greedy matching.

        # --- Update Matched Trackers ---
        for det_idx, trk_idx in matched_indices:
            try:
                tracker = self.trackers[trk_idx]
                det_poly = valid_detections[det_idx] # Use the validated & ordered detection

                # Prepare measurement for Kalman filter
                measurement = det_poly.reshape(-1).astype(np.float32)
                measurement = measurement.reshape((8, 1))

                # Update using Kalman filter
                tracker['kalman'].correct(measurement)

                # Extract updated state (Kalman's best estimate based on measurement)
                updated_state = tracker['kalman'].statePost
                kalman_poly = updated_state[:8].reshape(4, 2)

                # **MODIFIED:** Temporal Smoothing Logic
                smoothed_poly = kalman_poly # Default to Kalman output if no history
                adaptive_smoothing = self.smoothing_factor # Start with default smoothing

                # Optional: Add adaptive smoothing based on size change (can be tuned/removed)
                det_size = self.polygon_size(det_poly)
                tracker_size = self.polygon_size(tracker['polygon']) # Use last confirmed polygon size
                if det_size > 10 and tracker_size > 10: # Avoid division by zero/small numbers
                     size_ratio = max(det_size, tracker_size) / min(det_size, tracker_size)
                     if size_ratio > 1.5: # If size changed significantly, reduce smoothing factor
                          adaptive_smoothing = min(0.05, self.smoothing_factor) # More conservative update

                # Apply smoothing between Kalman's output and the PREVIOUS frame's smoothed polygon
                if len(tracker['history']) > 0:
                    last_confirmed_poly = tracker['history'][-1]
                    # Ensure last_confirmed_poly is valid before smoothing
                    if np.all(np.isfinite(last_confirmed_poly)) and last_confirmed_poly.shape == (4, 2):
                         smoothed_poly = adaptive_smoothing * kalman_poly + (1 - adaptive_smoothing) * last_confirmed_poly
                    # else: # Optional: handle invalid history case
                    #      smoothed_poly = kalman_poly # Fallback to Kalman output

                # Final check: ensure the smoothed polygon is valid
                if self.is_valid_polygon(smoothed_poly, frame_shape):
                     tracker['polygon'] = smoothed_poly
                else:
                     # If smoothed is invalid, fallback to Kalman's output (if valid)
                     if self.is_valid_polygon(kalman_poly, frame_shape):
                          tracker['polygon'] = kalman_poly
                     # else keep the previous polygon (last resort) - already handled by prediction stage


                # Update tracking metrics
                tracker['hits'] += 1
                tracker['time_since_update'] = 0
                # Add the NEWLY calculated smoothed polygon to history
                tracker['history'].append(tracker['polygon'].copy())

                # Keep history limited
                if len(tracker['history']) > 30:
                    tracker['history'] = tracker['history'][-30:]

            except Exception as e:
                print(f"Warning: Tracker update error for tracker {self.trackers[trk_idx]['id']}: {e}")
                # If update fails, mark tracker as potentially lost by not resetting time_since_update
                # tracker['time_since_update'] += 1 # Or keep it as is to rely on age removal
                continue

        # --- Update Unmatched Trackers ---
        # These trackers didn't get a detection, so update with their prediction
        for trk_idx in unmatched_trackers:
             tracker = self.trackers[trk_idx]
             # Use the prediction stored earlier
             tracker['polygon'] = tracker.get('predicted_polygon', tracker['polygon'])
             # Add the predicted (now current) polygon to history
             tracker['history'].append(tracker['polygon'].copy())
             # Keep history limited
             if len(tracker['history']) > 30:
                  tracker['history'] = tracker['history'][-30:]


        # --- Create New Trackers for Unmatched Detections ---
        for det_idx in unmatched_detections:
            try:
                det_poly = valid_detections[det_idx] # Already validated and ordered
                # Create the new tracker
                new_tracker = self.create_tracker(det_poly, self.next_id)
                self.trackers.append(new_tracker)
                self.next_id += 1
                # print(f"Created new tracker {new_tracker['id']}") # Optional: for debugging
            except Exception as e:
                print(f"Warning: New tracker creation error: {e}")
                continue

        # Return confirmed trackers (those with enough hits)
        return [t for t in self.trackers if t['hits'] >= self.min_hits]

    def should_detect(self):
        """Determine if we should run detection on this frame"""
        # Always detect for first few frames to establish tracking
        if self.frame_count < 5:
            return True

        # Detect on regular intervals
        if self.frame_count % self.detection_interval == 0:
            return True

        # Detect if we have few stable trackers
        stable_trackers = len([t for t in self.trackers if t['hits'] >= self.min_hits])
        # Adjust threshold based on expected number of ads (e.g., 1 if only one ad expected)
        if stable_trackers < 1:
            return True

        # Detect if a significant portion of trackers are 'young' (might indicate new entries)
        young_trackers = len([t for t in self.trackers if t['hits'] < self.min_hits])
        if len(self.trackers) > 0 and young_trackers / len(self.trackers) > 0.5:
             return True


        return False


def replace_advertisement(frame, polygon, replacement_img, alpha=1.0): # Default alpha to 1.0 for full replacement
    """
    Replace advertisement in frame with the replacement image using perspective warp.

    Args:
        frame: Original video frame
        polygon: 4-point polygon defining the advertisement area (ordered)
        replacement_img: Image to replace the advertisement with
        alpha: Blending factor (0-1). 1.0 means full replacement. (Currently unused, but kept for potential future blending)

    Returns:
        Frame with advertisement replaced, or original frame if replacement fails.
    """
    try:
        # Ensure polygon is valid float32 and ordered
        if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
             # print("Invalid polygon for replacement") # Debug
             return frame
        src_pts = polygon.astype(np.float32) # Ensure float32 for getPerspectiveTransform

        # Get dimensions of replacement image
        h_repl, w_repl = replacement_img.shape[:2]
        if h_repl == 0 or w_repl == 0:
             # print("Invalid replacement image dimensions") # Debug
             return frame

        # Define destination points (corners of replacement image - standard order)
        dst_pts = np.array([[0, 0], [w_repl - 1, 0], [w_repl - 1, h_repl - 1], [0, h_repl - 1]], dtype=np.float32)

        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(dst_pts, src_pts)
        if M is None:
             # print("Failed to get perspective transform") # Debug
             return frame


        # Warp the replacement image to fit the polygon in the frame's perspective
        h, w = frame.shape[:2]
        # Use BORDER_CONSTANT with black border to avoid edge artifacts if polygon goes slightly out
        warped = cv2.warpPerspective(replacement_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Create a mask for the polygon area in the frame
        mask = np.zeros((h, w), dtype=np.uint8)
        # Use fillConvexPoly for potentially slightly non-convex results from smoothing/kalman
        cv2.fillConvexPoly(mask, src_pts.astype(np.int32), 255)

        # Create an inverse mask to keep the original frame background
        mask_inv = cv2.bitwise_not(mask)

        # Black-out the area of the advertisement in the original frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Take only region of advertisement from warped image
        ad_fg = cv2.bitwise_and(warped, warped, mask=mask)

        # Put advertisement region and background together
        frame_out = cv2.add(frame_bg, ad_fg)

        # --- Optional Blending (if alpha < 1.0) ---
        if alpha < 1.0:
            original_ad_area = cv2.bitwise_and(frame, frame, mask=mask)
            blended_ad = cv2.addWeighted(ad_fg, alpha, original_ad_area, 1 - alpha, 0)
            frame_out = cv2.add(frame_bg, blended_ad)
        else: # Use the direct replacement if alpha is 1.0
            frame_out = cv2.add(frame_bg, ad_fg)


        return frame_out

    except Exception as e:
        print(f"Error during advertisement replacement: {e}")
        # Return the original frame if any error occurs during replacement
        return frame


def draw_polygon(frame, polygon, color=(0, 255, 0), thickness=2, label=None):
    """Draw polygon on frame with optional label"""
    try:
        # Ensure integer points for drawing
        pts = polygon.astype(np.int32)
        # Ensure it's a closed polygon for polylines
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

        if label:
            # Find the topmost point for label placement
            top_point_idx = np.argmin(pts[:, 1])
            top_point = pts[top_point_idx]
            # Put text slightly above the top point
            cv2.putText(frame, str(label), (top_point[0], max(0, top_point[1] - 10)), # Ensure y doesn't go negative
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness) # Use same thickness for text visibility
    except Exception as e:
         print(f"Error drawing polygon: {e}") # Avoid crashing if drawing fails


def main():
    # Load YOLO model (ensure the path is correct)
    print("Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH) # Update path if needed
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Create tracker with potentially adjusted smoothing factor
    tracker = StableAdTracker(
        max_age=30,          # Keep tracking for 30 frames without detection
        min_hits=3,          # Need 3 hits to confirm a tracker
        iou_threshold=0.4,   # Slightly lower IoU threshold might help matching if detections jitter
        smoothing_factor=0.1 # Start with 0.1, adjust based on results (lower=smoother, higher=more responsive)
    )

    # Load replacement image (ensure the path is correct)
    print("Loading replacement image...")
    replacement_img_path = REPLACEMENT_PATH
    replacement_img = cv2.imread(replacement_img_path)
    if replacement_img is None:
        print(f"Error: Could not load replacement image from {replacement_img_path}")
        return
    # Optional: Resize replacement image if needed, but perspective warp handles scaling
    # replacement_img = cv2.resize(replacement_img, (desired_width, desired_height))


    # Open video (ensure the path is correct)
    video_path = VIDEO_PATH
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Handle cases where FPS might be 0 or invalid
    fps = fps if fps > 0 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"output_stable_ads_{timestamp}.mp4")
    try:
        # Use 'mp4v' or 'avc1' for H.264 encoding if available, 'XVID' is another option
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
             print("Error: Could not open video writer. Trying XVID...")
             fourcc = cv2.VideoWriter_fourcc(*'XVID') # Fallback codec
             output_path = os.path.join(OUTPUT_DIR, f"output_stable_ads_{timestamp}.avi") # Change extension for AVI
             out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
             if not out.isOpened():
                  print("Error: Failed to open video writer with both mp4v and XVID.")
                  cap.release()
                  return
    except Exception as e:
         print(f"Error initializing VideoWriter: {e}")
         cap.release()
         return


    print(f"Writing output to: {output_path}")
    print(f"Video properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")

    # Statistics for debugging
    frames_processed = 0
    detections_run = 0
    total_ads_tracked = 0

    # Start time for FPS calculation
    start_time = time.time()

    # Process video frames
    print("Starting processing...")
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            frame_to_process = frame.copy() # Work on a copy

            detected_polygons = []
            # Decide whether to run detection on this frame
            if tracker.should_detect():
                detections_run += 1
                # Run YOLO detection
                try:
                    # Confidence threshold can be set here
                    results = model.predict(frame_to_process, conf=0.5, augment=False, verbose=False)[0]
                except Exception as e:
                    print(f"Error during YOLO prediction: {e}")
                    results = None # Ensure results is None if prediction fails


                # Process detections - extract polygons preferentially from masks
                if results and results.masks is not None and len(results.masks.xy) > 0:
                    # print(f"Frame {frames_processed + 1}: Using masks for detection.") # Debug
                    for mask_xy in results.masks.xy:
                        if len(mask_xy) >= 3: # Need at least 3 points for minAreaRect/approxPolyDP
                            polygon = np.array(mask_xy, dtype=np.float32)

                            # Option 1: Use approxPolyDP to simplify to 4 points (can be unstable)
                            #epsilon = 0.02 * cv2.arcLength(polygon, True)
                            #approx = cv2.approxPolyDP(polygon, epsilon, True)
                            #if len(approx) == 4:
                            #    detected_polygons.append(approx.reshape(4, 2))
                            #else: # Fallback to minAreaRect if approxPolyDP doesn't give 4 points
                            #    try:
                            #        rect = cv2.minAreaRect(polygon)
                            #        box = cv2.boxPoints(rect).astype(np.float32)
                            #        detected_polygons.append(box)
                            #    except Exception:
                            #        continue # Skip if minAreaRect fails

                            # Option 2: Directly use minAreaRect (often more stable than approxPolyDP for noisy masks)
                            try:
                                rect = cv2.minAreaRect(polygon)
                                box = cv2.boxPoints(rect).astype(np.float32)
                                # Basic filtering for the detected box itself
                                if cv2.contourArea(box.astype(np.int32)) > 50: # Filter very small boxes
                                    detected_polygons.append(box)
                            except Exception:
                                continue # Skip if minAreaRect fails

                # Fallback to bounding boxes if no masks or masks failed
                elif results and results.boxes is not None and len(results.boxes) > 0 and not detected_polygons:
                    # print(f"Frame {frames_processed + 1}: Using bounding boxes for detection.") # Debug
                    for box in results.boxes:
                        # Get box coordinates
                        xyxy = box.xyxy.cpu().numpy().flatten()
                        x1, y1, x2, y2 = map(int, xyxy)

                        # Basic filtering for boxes
                        box_w = x2 - x1
                        box_h = y2 - y1
                        if box_w < 20 or box_h < 20: continue # Too small
                        if box_w > 0.8 * frame_width or box_h > 0.8 * frame_height: continue # Too large

                        # Create polygon from bounding box
                        box_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                        detected_polygons.append(box_poly)
                # else: # Debug
                    # print(f"Frame {frames_processed + 1}: No valid detections found.")


                # Update tracker with new detections (or empty list if none found)
                tracked_ads = tracker.update(detected_polygons, frame_to_process)
            else:
                # Just update tracker predict step without new detections
                # print(f"Frame {frames_processed + 1}: Skipping detection, updating trackers.") # Debug
                tracked_ads = tracker.update([], frame_to_process)


            # --- Visualization and Replacement ---
            output_frame = frame.copy() # Draw on a fresh copy of the original frame

            # Replace advertisements using the SMOOTHED polygons from the tracker
            current_tracked_count = 0
            if tracked_ads: # Check if list is not empty
                current_tracked_count = len(tracked_ads)
                for ad in tracked_ads:
                    # Ensure the polygon is valid before attempting replacement
                    if tracker.is_valid_polygon(ad['polygon'], output_frame.shape):
                        output_frame = replace_advertisement(output_frame, ad['polygon'], replacement_img)
                        if DRAW_DEBUG:
                            # Draw the smoothed polygon for debugging
                            draw_polygon(output_frame, ad['polygon'], color=(0, 255, 0), thickness=2, label=f"ID:{ad['id']}")
                    elif DRAW_DEBUG: # Draw invalid polygons differently
                        draw_polygon(output_frame, ad['polygon'], color=(0, 0, 255), thickness=1, label=f"ID:{ad['id']} (Invalid)")


            if DRAW_DEBUG:
                # Draw raw detections for comparison/debugging
                for poly in detected_polygons:
                    draw_polygon(output_frame, poly, color=(255, 0, 0), thickness=1) # Blue for raw detections


            # Write the processed frame to output video
            out.write(output_frame)

            frames_processed += 1
            total_ads_tracked = max(total_ads_tracked, current_tracked_count)

            # Print progress periodically
            if frames_processed % 30 == 0:
                elapsed_time = time.time() - start_time
                current_fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
                progress = (cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames * 100) if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% ({frames_processed}/{total_frames if total_frames > 0 else '?'}) | "
                      f"FPS: {current_fps:.2f} | Tracking {current_tracked_count} ads | "
                      f"Detections Ran: {detections_run}")

            # Optional: Display the frame (can slow down processing significantly)
            # cv2.imshow("Processed Frame", output_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

        except Exception as e:
            print(f"\n--- Error during frame processing loop (frame {frames_processed}): {e} ---")
            # Optionally decide whether to break or try to continue
            # break
            continue # Try to process the next frame


    # Cleanup
    elapsed_time = time.time() - start_time
    avg_fps = frames_processed / elapsed_time if elapsed_time > 0 else 0

    cap.release()
    out.release()
    cv2.destroyAllWindows() # Close any display windows if used

    print(f"\nProcessing complete!")
    print(f"Output file: {output_path}")
    print(f"Processed {frames_processed} frames in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Detection runs: {detections_run} ({detections_run/frames_processed*100:.1f}% of frames)" if frames_processed > 0 else "Detection runs: 0")
    print(f"Maximum ads tracked simultaneously: {total_ads_tracked}")


if __name__ == "__main__":
    main()