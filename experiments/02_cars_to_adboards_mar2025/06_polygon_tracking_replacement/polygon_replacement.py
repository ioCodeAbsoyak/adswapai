"""AdSwapAI R&D, 2025-03-17: first ad replacement, perspective-warp an image into tracked polygons minus person masks."""

import sys
import os
import time
import argparse
import traceback

# Print system information for debugging
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Add better import error handling
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"ERROR: Could not import OpenCV: {e}")
    print("Please install OpenCV: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"ERROR: Could not import NumPy: {e}")
    print("Please install NumPy: pip install numpy")
    sys.exit(1)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"WARNING: Could not import PyTorch: {e}")
    print("Some features will be disabled. To enable them, install PyTorch.")
    # Create a dummy torch module to avoid errors
    class DummyTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False
    torch = DummyTorch()

class VideoAdTracker:
    """
    Interactive video advertisement tracker with polygon tracking.
    
    Features:
    - Play video immediately without selection
    - Select multiple polygonal regions with 'r' key
    - Tracks exact polygons, not bounding boxes
    - Replace ads with custom images that match the perspective
    - Removes people from the ad area
    """
    
    def __init__(self, video_path, replacement_image_path=None, output_dir=None, use_gpu=True):
        """Initialize with video path and optional replacement image"""
        print(f"Initializing with video: {video_path}")
        
        # Fix Windows path if needed
        self.video_path = video_path.replace('\\', '/')
        print(f"Using video path: {self.video_path}")
        
        # Check if file exists
        if not os.path.exists(self.video_path):
            raise ValueError(f"Video file does not exist: {self.video_path}")
        
        # Load replacement image if provided
        self.replacement_image = None
        if replacement_image_path:
            replacement_image_path = replacement_image_path.replace('\\', '/')
            print(f"Loading replacement image: {replacement_image_path}")
            if os.path.exists(replacement_image_path):
                self.replacement_image = cv2.imread(replacement_image_path)
                if self.replacement_image is None:
                    print(f"Warning: Could not load replacement image {replacement_image_path}")
                else:
                    print(f"Replacement image loaded: {self.replacement_image.shape}")
            else:
                print(f"Warning: Replacement image file not found: {replacement_image_path}")
        
        print("Opening video capture...")
        self.cap = cv2.VideoCapture(self.video_path)
        
        # Check with timeout if video opened successfully
        start_time = time.time()
        timeout = 5  # 5 seconds timeout
        while not self.cap.isOpened():
            if time.time() - start_time > timeout:
                raise ValueError(f"Could not open video file after {timeout} seconds: {self.video_path}")
            print("Waiting for video to open...")
            time.sleep(0.5)
            self.cap = cv2.VideoCapture(self.video_path)
            
        print("Video capture opened successfully")
            
        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Output settings
        if output_dir is None:
            self.output_dir = os.path.dirname(self.video_path)
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        # GPU settings
        self.use_gpu = False
        try:
            self.use_gpu = use_gpu and torch.cuda.is_available()
            if self.use_gpu:
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                
                # Configure CUDA settings for OpenCV if possible
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    print("OpenCV CUDA support available")
                    self.has_cv_cuda = True
                else:
                    self.has_cv_cuda = False
                    print("OpenCV CUDA support not available")
            else:
                self.has_cv_cuda = False
                print("Using CPU")
        except Exception as e:
            print(f"Warning: Error checking CUDA: {e}")
            self.use_gpu = False
            self.has_cv_cuda = False
            print("Defaulting to CPU mode due to error")
        
        # Window settings
        self.window_name = "Ad Tracker (r: reset polygons, n: new polygon, p: play/pause, q: quit)"
        
        # Multi-polygon tracking variables
        self.selecting = False                # Currently in selection mode
        self.current_polygon_points = []      # Current polygon points being drawn
        self.polygons = []                    # List of completed polygons (points)
        self.polygon_selected = False         # Has at least one polygon been completed
        
        # Original polygon info (multiple polygons)
        self.original_frames = []             # Original frames for each polygon
        self.original_polygons = []           # Original polygon points
        self.original_keypoints_list = []     # List of keypoints for each polygon
        self.original_descriptors_list = []   # List of descriptors for each polygon
        
        # Current tracking info (multiple polygons)
        self.current_polygons = []            # Current polygon points (tracked)
        self.tracking_quality_list = []       # Quality measure for each polygon (0-1)
        self.mask = None                      # Combined mask from all polygons
        self.person_mask = None               # Current person mask
        
        # For image replacement
        self.warped_images = []               # List of warped replacement images for each polygon
        
        # Feature matching for tracking
        print("Initializing feature detector...")
        try:
            self.detector = cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            print("Feature detector initialized")
        except Exception as e:
            print(f"Error initializing feature detector: {e}")
            raise
        
        # Person detection model
        print("Initializing person detector...")
        self.person_detector = self.initialize_person_detector()
        
        # Current frame and state
        self.frame = None
        self.current_frame_num = 0
        self.is_playing = False  # Start paused until user presses play
        self.was_playing = False # Remember play state before selection mode
        
        # For saving output
        self.video_writer = None
        
        # For FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.frame_counter = 0
        self.fps_display = 0
        self.fps_update_interval = 10  # Update FPS every 10 frames
        
        # Tracking confidence threshold - hide ad below this quality
        self.tracking_quality_threshold = 0.3
        
        # Person detection confidence threshold - lowered for better detection
        self.person_detection_confidence = 0.2  # Changed from 0.5 to 0.2 for better person detection
        
        print("Initialization complete")
    
    def initialize_person_detector(self):
        """Initialize person detector model with instance segmentation if possible"""
        print("Initializing person detector...")
        
        # Try to load YOLOv8 segmentation model (preferred for precise masks)
        try:
            print("Trying to import ultralytics for YOLOv8...")
            from ultralytics import YOLO
            print("Successfully imported ultralytics")
            
            print("Loading YOLOv8 model...")
            model = YOLO('yolov8n-seg.pt')  # Load the YOLOv8 segmentation model
            # Lower confidence threshold
            model.conf = 0.2  # Lower confidence for better detection
            model.iou = 0.5   # IOU threshold
            print("Using YOLOv8 segmentation for precise person masking")
            return {"model": model, "type": "yolov8-seg"}
        except Exception as e:
            print(f"Could not load YOLOv8 segmentation: {e}")
            
            # Try to load YOLOv5 model (less precise but still good)
            try:
                print("Trying to load YOLOv5 via torch hub...")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                
                # Set device
                if self.use_gpu:
                    model.to('cuda')
                    
                # Set inference size
                model.conf = self.person_detection_confidence  # Lower confidence threshold
                model.classes = [0]  # Person class only
                
                print(f"Using YOLOv5 for person detection with confidence threshold {self.person_detection_confidence}")
                return {"model": model, "type": "yolov5"}
            except Exception as e:
                print(f"Could not load YOLOv5: {e}")
        
        # Fallback to HOG person detector
        print("Using HOG person detector (limited precision)")
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return {"hog": hog, "type": "hog"}
        except Exception as e:
            print(f"Error initializing HOG detector: {e}")
            print("Using dummy person detector - no person detection will be performed")
            return {"type": "dummy"}
    
    def detect_persons_precise(self, frame):
        """Detect people only inside polygon areas to optimize performance"""
        h, w = frame.shape[:2]
        person_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Only continue if we have polygons with good tracking quality
        if not self.current_polygons or not self.tracking_quality_list:
            return person_mask
        
        # Create ROI mask of only the polygon areas we care about
        polygon_mask = np.zeros((h, w), dtype=np.uint8)
        for i, poly in enumerate(self.current_polygons):
            if i < len(self.tracking_quality_list) and self.tracking_quality_list[i] > self.tracking_quality_threshold:
                cv2.fillPoly(polygon_mask, [poly], 255)
        
        # If no valid polygons, return empty mask
        if np.sum(polygon_mask) == 0:
            return person_mask
            
        # Get bounding rectangle for all polygons combined to reduce search area
        contours, _ = cv2.findContours(polygon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each polygon area separately
        for contour in contours:
            # Get bounding box for this polygon region
            x, y, w_roi, h_roi = cv2.boundingRect(contour)
            
            # Expand region slightly to catch people partially in the area
            padding = 20  # pixels
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + w_roi + padding)
            y2 = min(h, y + h_roi + padding)
            
            # Skip if region is too small
            if w_roi < 10 or h_roi < 10:
                continue
                
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            roi_mask = polygon_mask[y1:y2, x1:x2]
            
            # Only proceed if ROI has content
            if roi.size == 0 or np.sum(roi_mask) == 0:
                continue
            
            detector_type = self.person_detector["type"]
            
            try:
                if detector_type == "yolov8-seg":
                    # Use YOLOv8 segmentation only on the ROI
                    model = self.person_detector["model"]
                    results = model(roi, classes=0)  # Only detect people (class 0)
                    
                    if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                        masks = results[0].masks.data
                        for mask in masks:
                            # Convert mask to correct size and format
                            mask_cv = (mask.cpu().numpy() * 255).astype(np.uint8)
                            mask_resized = cv2.resize(mask_cv, (x2-x1, y2-y1))
                            # Add to the full mask at the correct position
                            person_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                                person_mask[y1:y2, x1:x2], 
                                cv2.bitwise_and(mask_resized, roi_mask)
                            )
                
                elif detector_type == "yolov5":
                    # Use YOLOv5 on ROI
                    model = self.person_detector["model"]
                    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    results = model(rgb_roi, size=640)
                    
                    predictions = results.pred[0]
                    
                    for *box, conf, cls in predictions.cpu().numpy():
                        if conf < self.person_detection_confidence:
                            continue
                        
                        # Convert box coordinates to ROI coordinates
                        rx1, ry1, rx2, ry2 = map(int, box)
                        
                        # Ensure box is within ROI boundaries
                        rx1, ry1 = max(0, rx1), max(0, ry1)
                        rx2, ry2 = min(roi.shape[1]-1, rx2), min(roi.shape[0]-1, ry2)
                        
                        if rx2 <= rx1 or ry2 <= ry1:
                            continue
                        
                        # Create a mask for this person in the ROI
                        person_roi_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                        cv2.rectangle(person_roi_mask, (rx1, ry1), (rx2, ry2), 255, -1)
                        
                        # Only include the person if they overlap with the polygon
                        overlap = cv2.bitwise_and(person_roi_mask, roi_mask)
                        if np.sum(overlap) > 0:
                            # Add to the full mask at the correct position
                            person_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                                person_mask[y1:y2, x1:x2], 
                                overlap
                            )
                
                elif detector_type == "hog":
                    # Use HOG detector on ROI only if it has sufficient size
                    if w_roi < 64 or h_roi < 128:  # HOG needs minimum size
                        continue
                        
                    hog = self.person_detector["hog"]
                    boxes, weights = hog.detectMultiScale(roi, winStride=(8, 8), padding=(4, 4), scale=1.05)
                    
                    for (rx, ry, rw, rh) in boxes:
                        # Create mask for this detection in the ROI
                        person_roi_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                        cv2.rectangle(person_roi_mask, (rx, ry), (rx+rw, ry+rh), 255, -1)
                        
                        # Only include the person if they overlap with the polygon
                        overlap = cv2.bitwise_and(person_roi_mask, roi_mask)
                        if np.sum(overlap) > 0:
                            # Add to the full mask at the correct position
                            person_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                                person_mask[y1:y2, x1:x2], 
                                overlap
                            )
            
            except Exception as e:
                print(f"Error in person detection for ROI: {e}")
        
        # Apply morphological operations to clean up mask
        try:
            kernel = np.ones((5, 5), np.uint8)
            person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        except Exception as e:
            print(f"Error in morphological operations: {e}")
        
        # Store person mask for debugging/visualization
        self.person_mask = person_mask
    
        return person_mask
    
    def start_selection(self):
        """Start/reset polygon selection mode"""
        # Remember play state
        self.was_playing = self.is_playing
        
        # Pause video during selection
        self.is_playing = False
        
        # Reset all polygon data
        self.selecting = True
        self.current_polygon_points = []
        self.polygons = []
        self.polygon_selected = False
        
        self.original_frames = []
        self.original_polygons = []
        self.original_keypoints_list = []
        self.original_descriptors_list = []
        
        self.current_polygons = []
        self.tracking_quality_list = []
        self.warped_images = []
        self.mask = None
            
        print("Ad selection mode started. Click to add points, right-click to complete polygon.")
        print("Press 'n' to start a new polygon after completing one.")
    
    def start_new_polygon(self):
        """Start drawing a new polygon while keeping existing ones"""
        if not self.selecting:
            # Remember play state
            self.was_playing = self.is_playing
            
            # Pause video during selection
            self.is_playing = False
            
            # Start selection mode for a new polygon
            self.selecting = True
            self.current_polygon_points = []
            
            print("Drawing new polygon. Click to add points, right-click to complete.")
    
    def complete_current_polygon(self):
        """Complete the current polygon and set up tracking for it"""
        if self.selecting and len(self.current_polygon_points) >= 3:
            # Add current polygon to the list of polygons
            self.polygons.append(self.current_polygon_points)
            
            # Set up tracking for this polygon
            self.setup_tracking_for_polygon(self.frame, len(self.polygons) - 1)
            
            # Reset current polygon points for potential new polygon
            self.current_polygon_points = []
            
            # Mark that we have at least one polygon
            self.polygon_selected = True
            
            # Exit selection mode
            self.selecting = False
            
            # Restore play state
            self.is_playing = self.was_playing
            
            print(f"Polygon {len(self.polygons)} completed and tracking initialized")
    
    def setup_tracking_for_polygon(self, frame, polygon_index):
        """Initialize tracking for a specific polygon"""
        # Get the polygon points
        poly_points = self.polygons[polygon_index]
        
        # Store original data
        self.original_frames.append(frame.copy())
        
        # Convert points to numpy array
        np_poly = np.array(poly_points, dtype=np.int32)
        self.original_polygons.append(np_poly)
        self.current_polygons.append(np_poly.copy())
        
        # Initialize warped image for this polygon
        self.warped_images.append(None)
        
        # Create a mask from the polygon
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [np_poly], 255)
        
        # Find region of interest (bounding rectangle)
        x, y, w, h = cv2.boundingRect(np_poly)
        
        # Extract ROI for feature detection
        roi = frame[y:y+h, x:x+w].copy()
        
        # Create a mask for the ROI
        roi_local_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_poly = np_poly - np.array([x, y])
        cv2.fillPoly(roi_local_mask, [shifted_poly], 255)
        
        # Detect keypoints in the ROI (masked to polygon)
        keypoints, descriptors = self.detector.detectAndCompute(roi, roi_local_mask)
        
        if keypoints and descriptors is not None and len(keypoints) > 0:
            # Adjust keypoint coordinates to full frame
            for kp in keypoints:
                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                
            self.original_keypoints_list.append(keypoints)
            self.original_descriptors_list.append(descriptors)
            self.tracking_quality_list.append(1.0)  # Initial quality is perfect
            print(f"Tracking initialized for polygon {polygon_index} with {len(keypoints)} keypoints")
            
            # Prepare the warped replacement image for this polygon
            self.update_warped_image(polygon_index)
            
            # Update combined mask
            self.update_mask(frame)
            
            return True
        else:
            print(f"Warning: No keypoints found in polygon {polygon_index}, tracking may not work")
            self.original_keypoints_list.append([])
            self.original_descriptors_list.append(None)
            self.tracking_quality_list.append(0.0)
            
            return False
    
    def update_warped_image(self, polygon_index):
        """Update the warped replacement image for a specific polygon"""
        if self.replacement_image is None:
            # No replacement image provided
            self.warped_images[polygon_index] = None
            return
        
        try:
            # Get the current polygon
            polygon = self.current_polygons[polygon_index]
            
            if polygon is None or len(polygon) < 4:
                self.warped_images[polygon_index] = None
                return
            
            # Get polygon bounding rectangle
            x, y, w, h = cv2.boundingRect(polygon)
            
            # Resize replacement image to match the polygon's size while maintaining aspect ratio
            img_h, img_w = self.replacement_image.shape[:2]
            img_aspect = img_w / img_h
            
            # Determine target size to fit within the polygon
            if w / h > img_aspect:
                target_w = w
                target_h = int(w / img_aspect)
            else:
                target_h = h
                target_w = int(h * img_aspect)
            
            # Resize the replacement image
            resized_img = cv2.resize(self.replacement_image, (target_w, target_h))
            
            # Define destination points (polygon vertices)
            # For best results with perspective transform, use exactly 4 points
            dst_points = np.zeros((4, 2), dtype=np.float32)
            
            # If polygon has more than 4 points, use the corners of the bounding box
            if len(polygon) > 4:
                # Define corners of the bounding rectangle
                dst_points[0] = [x, y]                  # Top-left
                dst_points[1] = [x + w, y]              # Top-right
                dst_points[2] = [x + w, y + h]          # Bottom-right
                dst_points[3] = [x, y + h]              # Bottom-left
            else:
                # Use the polygon vertices directly
                for i in range(min(4, len(polygon))):
                    dst_points[i] = polygon[i]
            
            # Define source points (corners of the replacement image)
            src_points = np.array([
                [0, 0],                      # Top-left
                [target_w - 1, 0],           # Top-right
                [target_w - 1, target_h - 1], # Bottom-right
                [0, target_h - 1]            # Bottom-left
            ], dtype=np.float32)
            
            # Calculate perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective transform to the replacement image
            warped_img = cv2.warpPerspective(
                resized_img, 
                M, 
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            # Create mask for the polygon
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            
            # Only keep warped image pixels inside the polygon
            warped_mask = cv2.bitwise_and(warped_img, warped_img, mask=mask)
            
            # Store the warped image
            self.warped_images[polygon_index] = warped_mask
            
        except Exception as e:
            print(f"Error updating warped image for polygon {polygon_index}: {e}")
            self.warped_images[polygon_index] = None
    
    def update_mask(self, frame):
        """Update the combined mask based on all tracked polygons, removing people"""
        # Only proceed if we have polygons to track
        if not self.current_polygons:
            self.mask = None
            return
            
        # Create combined polygon mask
        combined_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Add each polygon with good tracking quality to the mask
        for i, poly in enumerate(self.current_polygons):
            if i < len(self.tracking_quality_list) and self.tracking_quality_list[i] > self.tracking_quality_threshold:
                cv2.fillPoly(combined_mask, [poly], 255)
        
        # Detect people with precise segmentation
        person_mask = self.detect_persons_precise(frame)
        
        # Subtract person mask from polygon mask
        # This ensures people in the ad region aren't colored
        ad_mask = cv2.subtract(combined_mask, person_mask)
        
        # Store the final mask
        self.mask = ad_mask
    
    def update_tracking(self, frame):
        """Update tracking for all polygons"""
        # If no polygons to track, exit
        if not self.original_polygons:
            return False
        
        # Track each polygon
        updated_any = False
        for i in range(len(self.original_polygons)):
            if self.update_tracking_for_polygon(frame, i):
                # Update the warped image for this polygon
                self.update_warped_image(i)
                updated_any = True
        
        # Update the combined mask
        if updated_any:
            self.update_mask(frame)
            
        return updated_any
    
    def update_tracking_for_polygon(self, frame, polygon_index):
        """Update tracking for a specific polygon"""
        # Check if we have valid data for this polygon
        if (polygon_index >= len(self.original_frames) or
            polygon_index >= len(self.original_descriptors_list) or
            polygon_index >= len(self.original_keypoints_list) or
            polygon_index >= len(self.original_polygons) or
            self.original_descriptors_list[polygon_index] is None or
            len(self.original_keypoints_list[polygon_index]) == 0):
            return False
        
        # Get polygon data
        original_keypoints = self.original_keypoints_list[polygon_index]
        original_descriptors = self.original_descriptors_list[polygon_index]
        original_poly = self.original_polygons[polygon_index]
        
        # Detect keypoints in current frame
        frame_keypoints, frame_descriptors = self.detector.detectAndCompute(frame, None)
        
        if not frame_keypoints or frame_descriptors is None or len(frame_keypoints) == 0:
            self.tracking_quality_list[polygon_index] = 0
            return False
        
        # Match keypoints
        matches = []
        try:
            matches = self.matcher.knnMatch(original_descriptors, frame_descriptors, k=2)
        except cv2.error:
            # If knnMatch fails, try regular match
            matches = self.matcher.match(original_descriptors, frame_descriptors)
            matches = [[m, m] for m in matches]  # Convert to similar format
        
        # Apply ratio test for good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            else:
                good_matches.append(match_pair[0])
        
        # Check if we have enough matches
        min_matches = 8  # Lower threshold for better tracking with multiple polygons
        if len(good_matches) < min_matches:
            self.tracking_quality_list[polygon_index] = len(good_matches) / min_matches
            print(f"Too few matches for polygon {polygon_index}: {len(good_matches)}/{min_matches}")
            return False
        
        # Get matched keypoints
        src_pts = np.float32([original_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches])
        
        # Calculate homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Calculate tracking quality based on inliers
        inliers = np.sum(mask)
        self.tracking_quality_list[polygon_index] = inliers / len(good_matches) if len(good_matches) > 0 else 0
        
        # If tracking quality is good, update polygon
        if H is not None and self.tracking_quality_list[polygon_index] > self.tracking_quality_threshold:
            # Convert polygon to format for transformation
            poly_points = original_poly.reshape(-1, 1, 2).astype(np.float32)
            
            # Transform the polygon
            transformed_poly = cv2.perspectiveTransform(poly_points, H)
            
            # Update current polygon
            self.current_polygons[polygon_index] = transformed_poly.reshape(-1, 2).astype(np.int32)
            
            return True
        else:
            # Low quality tracking - don't update this polygon
            print(f"Low tracking quality for polygon {polygon_index}: {self.tracking_quality_list[polygon_index]:.2f}")
            return False
    
    def draw_selection(self, frame):
        """Draw the current selection and completed polygons on the frame"""
        # Draw completed polygons
        for i, poly in enumerate(self.polygons):
            if i < len(self.tracking_quality_list):
                quality = self.tracking_quality_list[i]
                # Use color based on tracking quality - green for good, yellow for marginal, red for poor
                if quality > self.tracking_quality_threshold:
                    color = (0, 255, 0)  # Green
                elif quality > 0:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw polygon outline
                #np_poly = np.array(poly, dtype=np.int32)
                #cv2.polylines(frame, [np_poly], True, color, 1)
                
                # Show polygon number
                #centroid = np.mean(np_poly, axis=0).astype(int)
                #cv2.putText(frame, f"{i+1}", tuple(centroid), 
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw current polygon being drawn
        if self.selecting and len(self.current_polygon_points) > 0:
            # Draw points
            for point in self.current_polygon_points:
                cv2.circle(frame, point, 5, (0, 255, 255), -1)
            
            # Connect points with lines
            for i in range(len(self.current_polygon_points)):
                if i < len(self.current_polygon_points) - 1:
                    cv2.line(frame, self.current_polygon_points[i], self.current_polygon_points[i + 1], (0, 255, 255), 2)
            
            # Close the polygon if it has at least 3 points
            if len(self.current_polygon_points) >= 3:
                cv2.line(frame, self.current_polygon_points[-1], self.current_polygon_points[0], (0, 255, 255), 2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        # Only process events if in selection mode
        if not self.selecting:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to selection
            self.current_polygon_points.append((x, y))
            print(f"Added point at ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete selection with right click if we have at least 3 points
            if len(self.current_polygon_points) >= 3:
                self.complete_current_polygon()
    
    def calculate_fps(self):
        """Calculate and update FPS display value"""
        self.new_frame_time = time.time()
        self.frame_counter += 1
        
        # Update FPS every few frames for stable display
        if self.frame_counter >= self.fps_update_interval:
            time_diff = self.new_frame_time - self.prev_frame_time
            if time_diff > 0:
                self.fps_display = self.fps_update_interval / time_diff
                self.prev_frame_time = self.new_frame_time
                self.frame_counter = 0
    
    def display_frame(self, frame):
        """Display the current frame with replacement images and FPS"""
        try:
            display_frame = frame.copy()
            
            # Draw selection if in selection mode or if we have polygons
            self.draw_selection(display_frame)
            
            if self.selecting:
                cv2.putText(display_frame, "Selection Mode: Left-click to add points, Right-click to complete", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, "Press 'n' to start a new polygon, 'r' to reset all", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # If we have tracked polygons with good quality, overlay the replacement images
            if self.mask is not None:
                # For each polygon with good tracking quality
                for i, poly in enumerate(self.current_polygons):
                    if (i < len(self.tracking_quality_list) and 
                        self.tracking_quality_list[i] > self.tracking_quality_threshold):
                        
                        # Create mask for this polygon (excluding people)
                        poly_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                        cv2.fillPoly(poly_mask, [poly], 255)
                        
                        # Subtract person mask
                        if self.person_mask is not None:
                            poly_mask = cv2.subtract(poly_mask, self.person_mask)
                        
                        # If we have a replacement image for this polygon, use it
                        if i < len(self.warped_images) and self.warped_images[i] is not None:
                            # Apply the warped replacement image
                            mask_3ch = cv2.merge([poly_mask, poly_mask, poly_mask])
                            np.copyto(display_frame, self.warped_images[i], where=mask_3ch.astype(bool))
                        else:
                            # Fallback to red color overlay if no replacement image
                            red_overlay = display_frame.copy()
                            red_overlay[poly_mask > 0] = [0, 0, 255]  # BGR format
                            
                            # Blend with original frame
                            alpha = 0.7
                            cv2.addWeighted(red_overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                
                # Display tracking quality for each polygon
                y_pos = 60
                for i, quality in enumerate(self.tracking_quality_list):
                    status = "Good" if quality > self.tracking_quality_threshold else "Poor"
                    cv2.putText(display_frame, f"Polygon {i+1} tracking: {quality:.2f} ({status})", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_pos += 30
            
            # Display key controls
            if not self.selecting:
                cv2.putText(display_frame, "Press 'r' to reset polygons, 'n' for new polygon, 'p' to play/pause", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame number and playing status
            status = "Playing" if self.is_playing else "Paused"
            cv2.putText(display_frame, f"Frame: {self.current_frame_num}/{self.total_frames} ({status})", 
                       (10, self.height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {self.fps_display:.1f}", 
                       (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show the frame
            cv2.imshow(self.window_name, display_frame)
            
            # Write to output video if initialized
            if self.video_writer is not None:
                self.video_writer.write(display_frame)
        except Exception as e:
            print(f"Error in display_frame: {e}")
            traceback.print_exc()
    
    def setup_video_writer(self):
        """Initialize video writer for output"""
        try:
            # Create output filename
            base_name = os.path.basename(self.video_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(self.output_dir, f"{name}_replaced{ext}")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            
            print(f"Output will be saved to: {output_path}")
        except Exception as e:
            print(f"Error setting up video writer: {e}")
            self.video_writer = None
    
    def run(self):
        """Main processing loop"""
        try:
            # Create window
            print("Creating display window...")
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
            print("Reading first frame...")
            # Read first frame
            ret, self.frame = self.cap.read()
            if not ret:
                print("Failed to read video")
                return
                
            print("First frame read successfully")
                
            # Initialize FPS timer
            self.prev_frame_time = time.time()
            
            # Initial display
            print("Displaying first frame...")
            self.display_frame(self.frame)
            
            # Set up video writer
            print("Setting up video writer...")
            self.setup_video_writer()
            
            print("\nControls:")
            print("  - 'r': Reset all polygons")
            print("  - 'n': Start drawing a new polygon")
            print("  - Left click: Add point to polygon")
            print("  - Right click: Complete current polygon")
            print("  - 'p': Play/Pause video")
            print("  - 'q': Quit and save")
            
            print("Entering main loop...")
            
            # Main loop
            while True:
                # Calculate FPS
                self.calculate_fps()
                
                # If playing, read next frame (regardless of selection)
                if self.is_playing and not self.selecting:
                    ret, self.frame = self.cap.read()
                    
                    if not ret:
                        print("End of video reached")
                        break
                        
                    self.current_frame_num += 1
                    
                    # Update tracking if we have polygons
                    if self.polygon_selected:
                        self.update_tracking(self.frame)
                
                # Display current frame
                self.display_frame(self.frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('p'):  # Play/Pause
                    # Allow play/pause when not in selection mode
                    if not self.selecting:
                        self.is_playing = not self.is_playing
                        status = "Playing" if self.is_playing else "Paused"
                        print(f"Video {status}")
                
                elif key == ord('r'):  # Reset all polygons
                    self.start_selection()
                    print("All polygons reset")
                
                elif key == ord('n'):  # Start new polygon
                    self.start_new_polygon()
                
                elif key == ord('q'):  # Quit
                    print("Quitting...")
                    break
            
            # Clean up
            print("Cleaning up...")
            if self.video_writer is not None:
                self.video_writer.release()
            self.cap.release()
            cv2.destroyAllWindows()
            print("Cleanup complete")
            
        except Exception as e:
            print(f"Error in run: {e}")
            traceback.print_exc()
            
            # Clean up in case of error
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()

# Main function
def main():
    try:
        parser = argparse.ArgumentParser(description="Track and replace advertisements in videos")
        parser.add_argument("--video", "-v", default="adVideo.mp4", help="Path to input video")
        parser.add_argument("--replacement", "-r", default=None, help="Path to replacement image")
        parser.add_argument("--output", "-o", default=None, help="Output directory")
        parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
        
        args = parser.parse_args()
        print(f"Arguments: {args}")
        
        # Create and run tracker
        print("Creating VideoAdTracker instance...")
        tracker = VideoAdTracker(args.video, args.replacement, args.output, not args.cpu)
        
        print("Starting tracker...")
        tracker.run()
        
        print("Tracker finished successfully")
    except Exception as e:
        print(f"ERROR in main: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    print("Starting video_adtracker.py")
    sys.exit(main())