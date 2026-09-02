"""AdSwapAI R&D, 2025-03-16: track a user-drawn polygon through video with SIFT + RANSAC homography; YOLOv8-seg person masks excluded."""

import cv2
import numpy as np
import os
import time
import argparse
import torch

class VideoAdTracker:
    """
    Interactive video advertisement tracker with polygon tracking.
    
    Features:
    - Play video immediately without selection
    - Select polygonal regions at any time with 'r' key
    - Tracks exact polygons, not bounding boxes
    - Only shows ads when they're reliably tracked
    - Removes people from the ad area
    """
    
    def __init__(self, video_path, output_dir=None, use_gpu=True):
        """Initialize with video path"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Output settings
        if output_dir is None:
            self.output_dir = os.path.dirname(video_path)
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # GPU settings
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Configure CUDA settings for OpenCV if possible
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("OpenCV CUDA support available")
                self.has_cv_cuda = True
            else:
                self.has_cv_cuda = False
                print("OpenCV CUDA support not available")
        else:
            self.has_cv_cuda = False
            print("Using CPU")
        
        # Window settings
        self.window_name = "Ad Tracker (r: select ad, p: play/pause, q: quit)"
        
        # Tracking variables
        self.selecting = False  # Currently in selection mode
        self.roi_points = []    # Polygon points
        self.roi_selected = False  # Has a ROI been selected
        
        # Feature matching for tracking
        self.detector = cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Original ROI info
        self.original_roi_points = None  # Initial polygon points
        self.original_frame = None       # Initial frame
        self.original_keypoints = None   # Keypoints in original frame
        self.original_descriptors = None # Descriptors in original frame
        
        # Current tracking info
        self.current_roi_points = None  # Current polygon points
        self.tracking_quality = 1.0     # Quality measure (0-1)
        self.mask = None                # Current mask
        
        # Person detection model
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
    
    def initialize_person_detector(self):
        """Initialize person detector model with instance segmentation if possible"""
        print("Initializing person detector...")
        
        # Try to load YOLOv8 segmentation model (preferred for precise masks)
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n-seg.pt')  # Load the YOLOv8 segmentation model
            print("Using YOLOv8 segmentation for precise person masking")
            return {"model": model, "type": "yolov8-seg"}
        except Exception as e:
            print(f"Could not load YOLOv8 segmentation: {e}")
            
            # Try to load YOLOv5 model (less precise but still good)
            try:
                import torch
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                
                # Set device
                if self.use_gpu:
                    model.to('cuda')
                    
                # Set inference size
                model.conf = 0.5  # Confidence threshold
                model.classes = [0]  # Person class only
                
                print("Using YOLOv5 for person detection")
                return {"model": model, "type": "yolov5"}
            except Exception as e:
                print(f"Could not load YOLOv5: {e}")
        
        # Fallback to HOG person detector
        print("Using HOG person detector (limited precision)")
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return {"hog": hog, "type": "hog"}
    
    def detect_persons_precise(self, frame):
        """Detect people in the frame and return a precise mask without drawing bounding boxes"""
        h, w = frame.shape[:2]
        person_mask = np.zeros((h, w), dtype=np.uint8)
        
        detector_type = self.person_detector["type"]
        
        if detector_type == "yolov8-seg":
            # Use YOLOv8 segmentation for precise masks
            model = self.person_detector["model"]
            
            # Create a copy for detection without modifying original
            detect_frame = frame.copy()
            
            # Run prediction with segmentation
            results = model(detect_frame, classes=0)  # Only detect people (class 0)
            
            # Process segmentation masks
            if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.data
                for mask in masks:
                    # Convert mask to correct size and format
                    mask_cv = (mask.cpu().numpy() * 255).astype(np.uint8)
                    mask_resized = cv2.resize(mask_cv, (w, h))
                    person_mask = cv2.bitwise_or(person_mask, mask_resized)
        
        elif detector_type == "yolov5":
            # Use YOLOv5 for detection and GrabCut for segmentation
            model = self.person_detector["model"]
            
            # Copy frame for detection
            detect_frame = frame.copy()
            
            # Convert to RGB for YOLOv5
            rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            results = model(rgb_frame, size=640)
            
            # Get detections
            predictions = results.pred[0]
            
            # GrabCut for better segmentation within bounding boxes
            for *box, conf, cls in predictions.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                
                # Make sure the box is within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Use GrabCut to get a precise segmentation within the box
                try:
                    # Prepare mask
                    box_mask = np.zeros((h, w), np.uint8)
                    box_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD  # Probable foreground
                    
                    # Create inner rectangle (likely foreground)
                    inner_margin = int((x2-x1) * 0.2)
                    inner_x1 = x1 + inner_margin
                    inner_x2 = x2 - inner_margin
                    inner_y1 = y1 + inner_margin
                    inner_y2 = y2 - inner_margin
                    
                    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
                        box_mask[inner_y1:inner_y2, inner_x1:inner_x2] = cv2.GC_FGD  # Definite foreground
                    
                    # Mark border as background
                    box_mask[0:y1, :] = box_mask[y2:h, :] = box_mask[:, 0:x1] = box_mask[:, x2:w] = cv2.GC_BGD
                    
                    # Prepare GrabCut arrays
                    bgd_model = np.zeros((1, 65), np.float64)
                    fgd_model = np.zeros((1, 65), np.float64)
                    
                    # Extract ROI
                    rect = (x1, y1, x2-x1, y2-y1)
                    
                    # Apply GrabCut
                    cv2.grabCut(detect_frame, box_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                    
                    # Create mask where foreground (GC_FGD) or probable foreground (GC_PR_FGD)
                    person_segment = np.where((box_mask == cv2.GC_FGD) | (box_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
                    
                    # Add to person mask
                    person_mask = cv2.bitwise_or(person_mask, person_segment)
                    
                except cv2.error as e:
                    print(f"GrabCut error: {e}")
                    # Fallback to simple rectangle
                    cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, -1)
        
        else:
            # Use HOG detector
            hog = self.person_detector["hog"]
            
            # Detect people
            boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
            
            # Process each detection with GrabCut
            for (x, y, w, h) in boxes:
                try:
                    # Apply GrabCut for more precise segmentation
                    mask = np.zeros(frame.shape[:2], np.uint8)
                    rect = (x, y, w, h)
                    bgd_model = np.zeros((1, 65), np.float64)
                    fgd_model = np.zeros((1, 65), np.float64)
                    
                    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    
                    # Add to person mask
                    person_mask = cv2.bitwise_or(person_mask, mask2 * 255)
                except:
                    # Fallback to simple rectangle
                    cv2.rectangle(person_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        
        return person_mask
    
    def start_selection(self):
        """Start ROI selection mode"""
        # Remember play state
        self.was_playing = self.is_playing
        
        # Pause video during selection
        self.is_playing = False
        
        # Start selection mode
        self.selecting = True
        self.roi_points = []
        self.roi_selected = False
        self.current_roi_points = None
        self.original_roi_points = None
        self.original_frame = None
        self.original_keypoints = None
        self.original_descriptors = None
        self.mask = None
            
        print("Ad selection mode started. Click to add points, right-click to complete.")
    
    def setup_tracking(self, frame):
        """Initialize tracking with the current frame and selected ROI points"""
        # Store original data
        self.original_frame = frame.copy()
        self.original_roi_points = np.array(self.roi_points, dtype=np.int32)
        self.current_roi_points = self.original_roi_points.copy()
        
        # Create a mask from the polygon
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [self.original_roi_points], 255)
        
        # Find region of interest (bounding rectangle)
        x, y, w, h = cv2.boundingRect(self.original_roi_points)
        
        # Extract ROI for feature detection
        roi = frame[y:y+h, x:x+w].copy()
        
        # Create a mask for the ROI
        roi_local_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_poly = self.original_roi_points - np.array([x, y])
        cv2.fillPoly(roi_local_mask, [shifted_poly], 255)
        
        # Detect keypoints in the ROI (masked to polygon)
        keypoints, descriptors = self.detector.detectAndCompute(roi, roi_local_mask)
        
        if keypoints and descriptors is not None and len(keypoints) > 0:
            # Adjust keypoint coordinates to full frame
            for kp in keypoints:
                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                
            self.original_keypoints = keypoints
            self.original_descriptors = descriptors
            print(f"Tracking initialized with {len(keypoints)} keypoints in polygon")
            
            # Create initial mask
            self.update_mask(frame)
            
            # Restore play state
            self.is_playing = self.was_playing
            
            return True
        else:
            print("Warning: No keypoints found in ROI, tracking may not work")
            self.original_keypoints = []
            self.original_descriptors = None
            
            # Restore play state
            self.is_playing = self.was_playing
            
            return False
    
    def update_mask(self, frame):
        """Update the mask based on the tracked polygon, removing people"""
        # Only proceed if we have a polygon to track
        if self.current_roi_points is None or len(self.current_roi_points) < 3:
            self.mask = None
            return
            
        # Create polygon mask
        polygon_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [self.current_roi_points], 255)
        
        # Detect people with precise segmentation
        person_mask = self.detect_persons_precise(frame)
        
        # Subtract person mask only within the ad region
        # This ensures we only remove people from the ad region, not everywhere
        intersection = cv2.bitwise_and(polygon_mask, person_mask)
        self.mask = cv2.subtract(polygon_mask, intersection)
    
    def update_tracking(self, frame):
        """Update tracking using feature matching to transform the polygon"""
        # If no tracking initialized, exit
        if (self.original_frame is None or self.original_descriptors is None or 
            self.original_keypoints is None or len(self.original_keypoints) == 0 or
            self.original_roi_points is None or len(self.original_roi_points) < 3):
            return False
        
        # Detect keypoints in current frame
        frame_keypoints, frame_descriptors = self.detector.detectAndCompute(frame, None)
        
        if not frame_keypoints or frame_descriptors is None or len(frame_keypoints) == 0:
            self.tracking_quality = 0
            return False
        
        # Match keypoints
        matches = []
        try:
            matches = self.matcher.knnMatch(self.original_descriptors, frame_descriptors, k=2)
        except cv2.error:
            # If knnMatch fails, try regular match
            matches = self.matcher.match(self.original_descriptors, frame_descriptors)
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
        min_matches = 10
        if len(good_matches) < min_matches:
            self.tracking_quality = len(good_matches) / min_matches
            print(f"Too few matches: {len(good_matches)}/{min_matches}")
            return False
        
        # Get matched keypoints
        src_pts = np.float32([self.original_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches])
        
        # Calculate homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Calculate tracking quality based on inliers
        inliers = np.sum(mask)
        self.tracking_quality = inliers / len(good_matches) if len(good_matches) > 0 else 0
        
        # If tracking quality is good, update polygon
        if H is not None and self.tracking_quality > self.tracking_quality_threshold:
            # Convert polygon to format for transformation
            poly_points = self.original_roi_points.reshape(-1, 1, 2).astype(np.float32)
            
            # Transform the polygon
            transformed_poly = cv2.perspectiveTransform(poly_points, H)
            
            # Update current polygon - maintain the original polygon shape
            self.current_roi_points = transformed_poly.reshape(-1, 2).astype(np.int32)
            
            # Update mask
            self.update_mask(frame)
            return True
        else:
            # Low quality tracking - hide the ad
            print(f"Low tracking quality: {self.tracking_quality:.2f}")
            self.mask = None
            return False
    
    def draw_selection(self, frame):
        """Draw the current selection on the frame"""
        if self.selecting and len(self.roi_points) > 0:
            # Draw points
            for point in self.roi_points:
                cv2.circle(frame, point, 5, (0, 255, 255), -1)
            
            # Connect points with lines
            for i in range(len(self.roi_points)):
                if i < len(self.roi_points) - 1:
                    cv2.line(frame, self.roi_points[i], self.roi_points[i + 1], (0, 255, 255), 2)
            
            # Close the polygon if it has at least 3 points
            if len(self.roi_points) >= 3:
                cv2.line(frame, self.roi_points[-1], self.roi_points[0], (0, 255, 255), 2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        # Only process events if in selection mode
        if not self.selecting:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to selection
            self.roi_points.append((x, y))
            print(f"Added point at ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete selection with right click if we have at least 3 points
            if len(self.roi_points) >= 3:
                self.selecting = False
                self.roi_selected = True
                print("Selection completed")
                
                # Setup tracking with current frame and points
                self.setup_tracking(self.frame)
    
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
        """Display the current frame with mask overlay and FPS"""
        display_frame = frame.copy()
        
        # Draw selection if in selection mode
        if self.selecting:
            self.draw_selection(display_frame)
            cv2.putText(display_frame, "Selection Mode: Left-click to add points, Right-click to complete", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # If mask exists and tracking quality is good, apply the color overlay without "AD" text
        if self.mask is not None and self.tracking_quality > self.tracking_quality_threshold:
            overlay = display_frame.copy()
            # Apply red color to mask area
            overlay[self.mask > 0] = [0, 0, 255]  # BGR format
            
            # Blend with original frame
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
            
            # Draw tracked polygon outline 
            if self.current_roi_points is not None and len(self.current_roi_points) >= 3:
                cv2.polylines(display_frame, [self.current_roi_points], True, (0, 255, 0), 2)
                
                # Display tracking quality (not on the polygon)
                cv2.putText(display_frame, f"Tracking Quality: {self.tracking_quality:.2f}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display key controls
        if not self.selecting:
            cv2.putText(display_frame, "Press 'r' to select ad region, 'p' to play/pause", 
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
    
    def setup_video_writer(self):
        """Initialize video writer for output"""
        # Create output filename
        base_name = os.path.basename(self.video_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(self.output_dir, f"{name}_masked{ext}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        print(f"Output will be saved to: {output_path}")
    
    def run(self):
        """Main processing loop"""
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Read first frame
        ret, self.frame = self.cap.read()
        if not ret:
            print("Failed to read video")
            return
            
        # Initialize FPS timer
        self.prev_frame_time = time.time()
        
        # Initial display
        self.display_frame(self.frame)
        
        # Set up video writer
        self.setup_video_writer()
        
        print("\nControls:")
        print("  - 'r': Start/reset ad selection mode")
        print("  - Left click: Add point to selection")
        print("  - Right click: Complete selection")
        print("  - 'p': Play/Pause video")
        print("  - 'q': Quit and save")
        
        # Main loop
        while True:
            # Calculate FPS
            self.calculate_fps()
            
            # If playing, read next frame (regardless of selection)
            if self.is_playing:
                ret, self.frame = self.cap.read()
                
                if not ret:
                    print("End of video reached")
                    break
                    
                self.current_frame_num += 1
                
                # Update tracking if we have a selection
                if self.roi_selected and not self.selecting:
                    self.update_tracking(self.frame)
            
            # Display current frame
            self.display_frame(self.frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p'):  # Play/Pause
                # Allow play/pause at any time, regardless of selection state
                self.is_playing = not self.is_playing
                status = "Playing" if self.is_playing else "Paused"
                print(f"Video {status}")
            
            elif key == ord('r'):  # Start/reset selection
                self.start_selection()
            
            elif key == ord('q'):  # Quit
                break
        
        # Clean up
        if self.video_writer is not None:
            self.video_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Track and mask advertisements in videos")
    parser.add_argument("--video", "-v", default="adVideo.mp4", help="Path to input video")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    # Create and run tracker
    tracker = VideoAdTracker(args.video, args.output, not args.cpu)
    tracker.run()

if __name__ == "__main__":
    main()