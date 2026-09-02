"""AdSwapAI R&D, 2025-03-01: PyQt5 GUI with torchvision Faster R-CNN detection and Kalman/appearance (DeepSORT-style) tracking."""

import cv2
import time
import sys
import os
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
import torch.nn.functional as torch_F


# Check CUDA availability and test compatibility
CUDA_AVAILABLE = False
DEVICE = torch.device('cpu')

# First check if CUDA is available at all
if torch.cuda.is_available():
    try:
        # Try a small test operation to verify CUDA is working properly
        test_tensor = torch.zeros(1).cuda()
        test_result = test_tensor + 1
        test_result = test_result.cpu()  # Try to move back to CPU
        
        # If we got here, CUDA seems to be working
        CUDA_AVAILABLE = True
        DEVICE = torch.device('cuda')
        print(f"CUDA is available and working: {torch.cuda.get_device_name(0)}")
        
        # Print some diagnostic information
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        print(f"CUDA is available but encountered an error: {e}")
        print("Falling back to CPU mode")
        CUDA_AVAILABLE = False
        DEVICE = torch.device('cpu')
else:
    print("CUDA is not available, using CPU")


class VideoDisplayWidget(QWidget):
    """Widget for displaying video frames and capturing user selections"""
    
    # Signal emitted when user makes a selection
    selection_made = pyqtSignal(QRect)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        
        # Variables for selection
        self.start_point = None
        self.end_point = None
        self.is_selecting = False
        
        # Current frame being displayed
        self.current_pixmap = None
        
    def set_frame(self, cv_img):
        """Set the current frame to display"""
        if cv_img is None:
            return
            
        # Convert OpenCV image (BGR) to Qt image (RGB)
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        
        # Convert BGR to RGB
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to pixmap and store
        self.current_pixmap = QPixmap.fromImage(qt_img)
        self.update()
        
    def paintEvent(self, event):
        """Paint the current frame and selection rectangle"""
        painter = QPainter(self)
        
        # Display the current frame
        if self.current_pixmap:
            # Calculate size to maintain aspect ratio
            scaled_pixmap = self.current_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Calculate position to center the image
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            
            # Draw the image
            painter.drawPixmap(x, y, scaled_pixmap)
            
            # If selecting, draw the selection rectangle
            if self.is_selecting and self.start_point is not None and self.end_point is not None:
                # Get the rect in the widget coordinates
                rect = QRect(self.start_point, self.end_point).normalized()
                
                # Draw the selection rectangle
                painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
                painter.drawRect(rect)
                
        painter.end()
    
    def mousePressEvent(self, event):
        """Handle mouse press to start selection"""
        if event.button() == Qt.LeftButton and self.current_pixmap:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.is_selecting = True
            self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement to update selection"""
        if self.is_selecting:
            self.end_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to finalize selection"""
        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.end_point = event.pos()
            
            # Create normalized rectangle
            selection = QRect(self.start_point, self.end_point).normalized()
            
            # Emit signal with the selection
            if selection.width() > 10 and selection.height() > 10:
                scaled_selection = self.scale_selection_to_image(selection)
                self.selection_made.emit(scaled_selection)
            
            self.update()
    
    def scale_selection_to_image(self, widget_selection):
        """Scale selection from widget coordinates to image coordinates"""
        if not self.current_pixmap:
            return widget_selection
        
        # Get widget size and pixmap size
        widget_width = self.width()
        widget_height = self.height()
        pixmap_width = self.current_pixmap.width()
        pixmap_height = self.current_pixmap.height()
        
        # Calculate scaled pixmap size (as displayed)
        scale_factor = min(widget_width / pixmap_width, widget_height / pixmap_height)
        scaled_width = int(pixmap_width * scale_factor)
        scaled_height = int(pixmap_height * scale_factor)
        
        # Calculate position offset (centered display)
        x_offset = (widget_width - scaled_width) // 2
        y_offset = (widget_height - scaled_height) // 2
        
        # Convert widget coordinates to image coordinates
        x = int((widget_selection.x() - x_offset) / scale_factor)
        y = int((widget_selection.y() - y_offset) / scale_factor)
        width = int(widget_selection.width() / scale_factor)
        height = int(widget_selection.height() / scale_factor)
        
        # Clamp to image boundaries
        x = max(0, min(x, pixmap_width - 1))
        y = max(0, min(y, pixmap_height - 1))
        width = min(width, pixmap_width - x)
        height = min(height, pixmap_height - y)
        
        return QRect(x, y, width, height)


class DeepFeatureExtractor:
    """Feature extractor for DeepSORT algorithm"""

    def __init__(self, model_type='resnet18', use_cuda=True):
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        try:
            # Try to load with new weights parameter style
            from torchvision.models import ResNet18_Weights
            self.model = torch.hub.load('pytorch/vision', model_type, weights=ResNet18_Weights.DEFAULT)
        except Exception as e:
            print(f"Error loading model with weights parameter: {e}")
            # Fallback to older method
            self.model = torch.hub.load('pytorch/vision', model_type, pretrained=True)
        
        # Remove the last fully connected layer to get feature vectors
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
        # Always set to eval mode first
        self.model.eval()
        
        # Normalization parameters for pre-trained model
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Try to move to CUDA if available, with fallback
        if self.use_cuda:
            try:
                self.model = self.model.to(DEVICE)
                self.mean = self.mean.to(DEVICE)
                self.std = self.std.to(DEVICE)
                
                # Test with a small tensor
                test_input = torch.zeros(1, 3, 224, 224).to(DEVICE)
                with torch.no_grad():
                    _ = self.model(test_input)
                print("Successfully moved feature extractor to GPU")
            except Exception as e:
                print(f"Failed to use GPU for feature extractor: {e}")
                self.use_cuda = False
                self.model = self.model.cpu()
                self.mean = self.mean.cpu()
                self.std = self.std.cpu()
    
    def extract_features(self, image_patches):
        """
        Extract feature vectors from image patches
        
        Args:
            image_patches: List of image patches (cropped from main image)
            
        Returns:
            Tensor of feature vectors
        """
        if not image_patches:
            return None
            
        try:
            # Process each patch
            processed_patches = []
            for patch in image_patches:
                try:
                    # Resize to 224x224 (standard input size for ResNet)
                    resized = cv2.resize(patch, (224, 224))
                    
                    # Convert BGR to RGB and normalize to [0, 1]
                    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
                    
                    # Convert to PyTorch tensor and add batch dimension
                    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
                    
                    # Normalize with ImageNet mean and std
                    img_tensor = (img_tensor - self.mean) / self.std
                    
                    processed_patches.append(img_tensor)
                except Exception as e:
                    print(f"Error processing image patch: {e}")
                    # Create a blank tensor for this patch to maintain batch consistency
                    dummy_tensor = torch.zeros(3, 224, 224)
                    dummy_tensor = (dummy_tensor - self.mean) / self.std
                    processed_patches.append(dummy_tensor)
            
            # Stack into batch
            batch = torch.stack(processed_patches)
            
            # Move to appropriate device with error handling
            device_to_use = DEVICE if self.use_cuda else torch.device('cpu')
            
            try:
                if self.use_cuda:
                    batch = batch.to(device_to_use)
                
                # Extract features
                with torch.no_grad():
                    features = self.model(batch)
                    
                # Reshape features to 2D and normalize
                features = features.view(features.size(0), -1)
                features = torch_F.normalize(features, p=2, dim=1)
                
                return features
                
            except RuntimeError as e:
                if "CUDA" in str(e) and self.use_cuda:
                    print(f"CUDA error during feature extraction: {e}")
                    print("Falling back to CPU for feature extraction...")
                    
                    # Switch to CPU
                    self.use_cuda = False
                    self.model = self.model.cpu()
                    self.mean = self.mean.cpu()
                    self.std = self.std.cpu()
                    
                    # Try again on CPU
                    return self.extract_features(image_patches)
                else:
                    raise e
                    
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return empty feature vector
            return torch.zeros(len(image_patches), 512)  # 512 is typical ResNet feature dim


class ObjectDetector:
    """Object detector using Faster R-CNN"""
    
    def __init__(self, confidence_threshold=0.5, use_cuda=True):
        self.confidence_threshold = confidence_threshold
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        # COCO class labels (we'll filter for cars, trucks, etc.)
        self.coco_labels = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 
            5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck'
        }
        
        # Vehicle class IDs (that we want to detect)
        self.vehicle_classes = {2, 3, 4, 6, 7, 8}  # bicycle, car, motorcycle, bus, train, truck
        
        try:
            # Load pre-trained model using the newer weights parameter
            from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
            self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        except Exception as e:
            print(f"Error loading model with weights parameter: {e}")
            # Fallback to older method
            self.model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        
        # Always ensure model is in evaluation mode
        self.model.eval()
        
        # Only move to GPU if CUDA is truly available and stable
        if self.use_cuda:
            try:
                self.model = self.model.to(DEVICE)
                # Test with a small tensor
                test_input = torch.zeros(1, 3, 224, 224).to(DEVICE)
                with torch.no_grad():
                    _ = self.model.backbone(test_input)
                print("Successfully moved model to GPU")
            except Exception as e:
                print(f"Failed to use GPU for model: {e}")
                self.use_cuda = False
                self.model = self.model.cpu()
        
    def detect(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detection boxes (x1, y1, x2, y2, score, class_id)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PyTorch tensor
        img_tensor = F.to_tensor(rgb_frame)
        
        # Safe device placement with error handling
        device_to_use = DEVICE if self.use_cuda else torch.device('cpu')
        
        try:
            if self.use_cuda:
                img_tensor = img_tensor.to(device_to_use)
                
            # Run detection with error handling
            with torch.no_grad():
                output = self.model([img_tensor])
                
            # Extract detections
            boxes = output[0]['boxes'].cpu().numpy()
            scores = output[0]['scores'].cpu().numpy()
            labels = output[0]['labels'].cpu().numpy()
            
            # Filter by confidence threshold and vehicle classes
            detections = []
            for box, score, label in zip(boxes, scores, labels):
                if score >= self.confidence_threshold and label in self.vehicle_classes:
                    x1, y1, x2, y2 = box
                    class_name = self.coco_labels.get(label, 'unknown')
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),  # Convert to (x, y, w, h)
                        'score': float(score),
                        'class_id': int(label),
                        'class_name': class_name
                    })
                    
            return detections
            
        except RuntimeError as e:
            if "CUDA" in str(e) and self.use_cuda:
                print(f"CUDA error during detection: {e}")
                print("Falling back to CPU for this detection...")
                
                # Switch to CPU for this detection
                self.use_cuda = False
                self.model = self.model.cpu()
                
                # Try again on CPU
                return self.detect(frame)
            else:
                # For other errors, return empty detections but log the error
                print(f"Error during object detection: {e}")
                return []
        except Exception as e:
            print(f"Unexpected error in detection: {e}")
            return []


class KalmanTracker:
    """Simple Kalman filter-based tracker for a single object"""
    
    def __init__(self, bbox, track_id):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        
        # Initialize Kalman filter
        self.kalman = cv2.KalmanFilter(8, 4)  # 8 state variables, 4 measurement variables
        
        # State: [x, y, w, h, vx, vy, vw, vh]
        # Measurement: [x, y, w, h]
        
        # State transition matrix
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]   # vh = vh
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0],  # w
            [0, 0, 0, 1, 0, 0, 0, 0]   # h
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32) * 1.0
        
        # Initialize state
        x, y, w, h = bbox
        self.kalman.statePost = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        
        # Tracking metrics
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.max_age = 30  # Maximum frames to keep without matching
        self.min_hits = 3   # Minimum hits to consider tracker confirmed
        
        # Store feature vector for matching
        self.features = None
    
    def predict(self):
        """Predict next state"""
        prediction = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        
        # Extract bbox from prediction
        x = max(0, prediction[0][0])
        y = max(0, prediction[1][0])
        w = max(1, prediction[2][0])
        h = max(1, prediction[3][0])
        
        self.bbox = (int(x), int(y), int(w), int(h))
        return self.bbox
    
    def update(self, bbox, features=None):
        """Update tracker with new detection"""
        x, y, w, h = bbox
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)
        
        # Update Kalman filter
        self.kalman.correct(measurement)
        
        # Extract bbox from corrected state
        state = self.kalman.statePost
        x = max(0, state[0][0])
        y = max(0, state[1][0])
        w = max(1, state[2][0])
        h = max(1, state[3][0])
        
        self.bbox = (int(x), int(y), int(w), int(h))
        
        # Update tracking metrics
        self.time_since_update = 0
        self.hits += 1
        
        # Update features if provided
        if features is not None:
            if self.features is None:
                self.features = features
            else:
                # Average with previous features for stability
                self.features = 0.7 * self.features + 0.3 * features
                # Re-normalize
                self.features = self.features / torch.norm(self.features)
                
        return self.bbox
    
    def is_confirmed(self):
        """Check if tracker is confirmed (reliable)"""
        return self.hits >= self.min_hits
    
    def is_deleted(self):
        """Check if tracker should be deleted"""
        return self.time_since_update > self.max_age
    
    def get_state(self):
        """Get current state"""
        return self.bbox


class DeepSort:
    """Multi-object tracker using Kalman filtering and deep association metrics"""
    
    def __init__(self, max_cosine_distance=0.5, nn_budget=100, use_cuda=True):
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        # Initialize trackers list
        self.trackers = []
        
        # Next track ID
        self.next_id = 1
        
        # Initialize feature extractor
        self.feature_extractor = DeepFeatureExtractor(use_cuda=self.use_cuda)
    
    def update(self, frame, detections):
        """
        Update tracks with new detections
        
        Args:
            frame: Current video frame
            detections: List of detected objects with bbox
            
        Returns:
            List of active tracks
        """
        # Predict new locations of existing trackers
        predicted_trackers = []
        for tracker in self.trackers:
            tracker.predict()
            if not tracker.is_deleted():
                predicted_trackers.append(tracker)
        
        self.trackers = predicted_trackers
        
        # If no detections, return current trackers
        if not detections:
            return [t for t in self.trackers if t.is_confirmed()]
        
        # Extract image patches for detected objects
        bboxes = [det['bbox'] for det in detections]
        patches = []
        for x, y, w, h in bboxes:
            patch = frame[y:y+h, x:x+w]
            if patch.size == 0:  # Skip empty patches
                patch = np.zeros((1, 1, 3), dtype=np.uint8)
            patches.append(patch)
        
        # Extract features from patches
        if patches:
            features = self.feature_extractor.extract_features(patches)
            if self.use_cuda:
                features = features.cpu()
            features = features.numpy()
        else:
            features = np.array([])
        
        # Match detections to trackers
        if self.trackers and features.size > 0:
            # Calculate cost matrix (feature distance)
            cost_matrix = np.zeros((len(detections), len(self.trackers)))
            
            for i, feat in enumerate(features):
                for j, tracker in enumerate(self.trackers):
                    if tracker.features is not None:
                        # Use cosine distance as cost
                        tracker_feat = tracker.features.cpu().numpy() if torch.is_tensor(tracker.features) else tracker.features
                        cost = 1.0 - np.dot(feat, tracker_feat)
                        cost_matrix[i, j] = cost
                    else:
                        cost_matrix[i, j] = self.max_cosine_distance + 0.1
            
            # Use Hungarian algorithm for assignment
            matched_indices = []
            unmatched_detections = list(range(len(detections)))
            unmatched_trackers = list(range(len(self.trackers)))
            
            # Simple greedy matching for demonstration
            # (In a real implementation, use Hungarian algorithm)
            for i in range(len(detections)):
                for j in range(len(self.trackers)):
                    if j in unmatched_trackers and i in unmatched_detections:
                        if cost_matrix[i, j] < self.max_cosine_distance:
                            matched_indices.append((i, j))
                            unmatched_detections.remove(i)
                            unmatched_trackers.remove(j)
                            break
        else:
            matched_indices = []
            unmatched_detections = list(range(len(detections)))
            unmatched_trackers = list(range(len(self.trackers)))
        
        # Update matched trackers
        for det_idx, trk_idx in matched_indices:
            det = detections[det_idx]
            tracker = self.trackers[trk_idx]
            
            # Convert feature tensor to PyTorch tensor if it's NumPy
            if isinstance(features[det_idx], np.ndarray):
                feat_tensor = torch.from_numpy(features[det_idx]).float()
                if self.use_cuda:
                    feat_tensor = feat_tensor.to(DEVICE)
            else:
                feat_tensor = features[det_idx]
                
            tracker.update(det['bbox'], feat_tensor)
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            
            # Convert feature tensor to PyTorch tensor if it's NumPy
            if features.size > 0:
                if isinstance(features[det_idx], np.ndarray):
                    feat_tensor = torch.from_numpy(features[det_idx]).float()
                    if self.use_cuda:
                        feat_tensor = feat_tensor.to(DEVICE)
                else:
                    feat_tensor = features[det_idx]
            else:
                feat_tensor = None
                
            new_tracker = KalmanTracker(det['bbox'], self.next_id)
            if feat_tensor is not None:
                new_tracker.features = feat_tensor
            self.next_id += 1
            self.trackers.append(new_tracker)
        
        # Return confirmed trackers
        return [t for t in self.trackers if t.is_confirmed()]


class AdvancedCarTrackingApp(QMainWindow):
    """Main application for advanced car tracking with Faster R-CNN and DeepSORT"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Advanced Car Tracking with Faster R-CNN & DeepSORT")
        self.setMinimumSize(800, 600)
        
        # Check CUDA availability
        self.use_cuda = CUDA_AVAILABLE
        
        # Video processing variables
        self.cap = None
        self.current_frame = None
        self.current_frame_num = 0
        
        # Video properties
        self.fps = 0
        self.frame_count = 0
        self.video_width = 0
        self.video_height = 0
        
        # Initialize models
        self.detector = None
        self.tracker = None
        self.tracks = []
        
        # Flags
        self.auto_detect = False
        self.show_detections = True
        
        # Timer for playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        # Timer for FPS calculation
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update FPS every second
        self.frames_processed = 0
        self.current_fps = 0
        
        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create top controls
        top_controls = QHBoxLayout()
        
        # Create buttons
        self.load_btn = QPushButton("Load Video")
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.detect_btn = QPushButton("Load Models")
        
        # Auto-detect checkbox
        self.auto_detect_check = QCheckBox("Auto-detect Cars")
        self.auto_detect_check.setChecked(self.auto_detect)
        self.auto_detect_check.stateChanged.connect(self.toggle_auto_detect)
        
        # Show detections checkbox
        self.show_detect_check = QCheckBox("Show Detections")
        self.show_detect_check.setChecked(self.show_detections)
        self.show_detect_check.stateChanged.connect(self.toggle_show_detections)
        
        # Disable buttons initially
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.auto_detect_check.setEnabled(False)
        
        # Connect signals
        self.load_btn.clicked.connect(self.load_video)
        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)
        self.detect_btn.clicked.connect(self.load_models)
        
        # Add buttons to layout
        top_controls.addWidget(self.load_btn)
        top_controls.addWidget(self.play_btn)
        top_controls.addWidget(self.pause_btn)
        top_controls.addWidget(self.detect_btn)
        top_controls.addWidget(self.auto_detect_check)
        top_controls.addWidget(self.show_detect_check)
        top_controls.addStretch()
        
        # Create GPU indicator
        self.gpu_label = QLabel("GPU: " + ("ENABLED" if self.use_cuda else "DISABLED"))
        self.gpu_label.setStyleSheet(
            "color: " + ("green" if self.use_cuda else "red") + "; font-weight: bold;"
        )
        top_controls.addWidget(self.gpu_label)
        
        # Create tracking info label
        self.track_label = QLabel("Tracking: 0 cars")
        top_controls.addWidget(self.track_label)
        
        # Create FPS label
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: red; font-weight: bold;")
        top_controls.addWidget(self.fps_label)
        
        # Add top controls to main layout
        main_layout.addLayout(top_controls)
        
        # Create video display widget
        self.video_widget = VideoDisplayWidget()
        self.video_widget.selection_made.connect(self.handle_selection)
        main_layout.addWidget(self.video_widget)
        
        # Create status bar
        status = "Ready. GPU: " + ("ENABLED" if self.use_cuda else "DISABLED")
        self.statusBar().showMessage(status)
    
    def load_models(self):
        """Load object detection and tracking models"""
        if self.detector is None:
            self.statusBar().showMessage("Loading models... This may take a moment.")
            QApplication.processEvents()
            
            # Initialize object detector
            self.detector = ObjectDetector(confidence_threshold=0.5, use_cuda=self.use_cuda)
            
            # Initialize tracker
            self.tracker = DeepSort(use_cuda=self.use_cuda)
            
            self.statusBar().showMessage("Models loaded successfully")
            self.auto_detect_check.setEnabled(True)
            
            # Enable auto-detect by default after models are loaded
            self.auto_detect = True
            self.auto_detect_check.setChecked(True)
        else:
            self.statusBar().showMessage("Models already loaded")
    
    def toggle_auto_detect(self, state):
        """Toggle automatic detection mode"""
        self.auto_detect = (state == Qt.Checked)
        
        if self.auto_detect and self.detector is None:
            self.load_models()
            
        self.statusBar().showMessage(
            f"Auto-detection {'enabled' if self.auto_detect else 'disabled'}"
        )
    
    def toggle_show_detections(self, state):
        """Toggle showing detection boxes"""
        self.show_detections = (state == Qt.Checked)
        
        # Update display if video is paused
        if self.current_frame is not None and not self.timer.isActive():
            self.next_frame()
    
    def load_video(self):
        """Open a file dialog to select a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        
        if file_path:
            # Open the video file
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                self.statusBar().showMessage(f"Error: Could not open video file")
                return
                
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Reset tracking variables
            self.current_frame_num = 0
            self.tracks = []
            
            # Read the first frame
            ret, self.current_frame = self.cap.read()
            if ret:
                # Display the first frame
                self.video_widget.set_frame(self.current_frame)
                
                # Update status bar
                self.statusBar().showMessage(
                    f"Video loaded: {self.video_width}x{self.video_height}, "
                    f"{self.fps:.2f} FPS, {self.frame_count} frames"
                )
                
                # Enable buttons
                self.play_btn.setEnabled(True)
            else:
                self.statusBar().showMessage("Error: Could not read video frames")
    
    def next_frame(self):
        """Process and display the next frame"""
        if self.cap is None or self.current_frame is None:
            return
            
        # Read the next frame
        ret, frame = self.cap.read()
        
        if not ret:
            # End of video
            self.pause_video()
            self.statusBar().showMessage("End of video reached")
            return
            
        # Update frame counter
        self.current_frame_num += 1
        self.current_frame = frame
        self.frames_processed += 1
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Run object detection and tracking if enabled
        if self.auto_detect and self.detector is not None and self.tracker is not None:
            # Detect vehicles
            detections = self.detector.detect(frame)
            
            # Update tracks
            self.tracks = self.tracker.update(frame, detections)
            
            # Update tracking label
            self.track_label.setText(f"Tracking: {len(self.tracks)} cars")
            
            # Draw detection results
            if self.show_detections:
                # Draw detection boxes
                for det in detections:
                    x, y, w, h = det['bbox']
                    score = det['score']
                    class_name = det['class_name']
                    
                    # Draw rectangle
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw label with score
                    label = f"{class_name}: {score:.2f}"
                    cv2.putText(
                        display_frame, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                
                # Draw tracking results
                for tracker in self.tracks:
                    x, y, w, h = tracker.get_state()
                    track_id = tracker.track_id
                    
                    # Draw rectangle
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Draw ID
                    cv2.putText(
                        display_frame, f"ID: {track_id}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
        
        # Display the frame
        self.video_widget.set_frame(display_frame)
        
        # Update status bar
        self.statusBar().showMessage(
            f"Frame: {self.current_frame_num}/{self.frame_count}, "
            f"FPS: {self.current_fps:.1f}"
        )
    
    def play_video(self):
        """Start video playback"""
        if self.cap is not None and not self.timer.isActive():
            # Set timer interval based on video FPS
            interval = int(1000 / self.fps) if self.fps > 0 else 33  # Default to ~30 FPS
            self.timer.start(interval)
            
            # Update buttons
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            
            self.statusBar().showMessage("Playing video...")
    
    def pause_video(self):
        """Pause video playback"""
        if self.timer.isActive():
            self.timer.stop()
            
            # Update buttons
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            
            self.statusBar().showMessage("Video paused")
    
    def update_fps(self):
        """Update FPS counter"""
        self.current_fps = self.frames_processed
        self.frames_processed = 0
        
        # Update FPS label
        self.fps_label.setText(f"FPS: {self.current_fps:.1f}")
        
        # Color code FPS label
        if self.current_fps >= 20:
            self.fps_label.setStyleSheet("color: green; font-weight: bold;")
        elif self.current_fps >= 10:
            self.fps_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.fps_label.setStyleSheet("color: red; font-weight: bold;")
    
    def handle_selection(self, rect):
        """Handle user selection on video frame"""
        if self.current_frame is None:
            return
            
        # Extract selected region
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        
        # Make sure the selection is within the frame
        x = max(0, min(x, self.video_width - 1))
        y = max(0, min(y, self.video_height - 1))
        w = min(w, self.video_width - x)
        h = min(h, self.video_height - y)
        
        self.statusBar().showMessage(f"Selected region: ({x}, {y}, {w}, {h})")
        
        # If we have a detector and tracker, we could initialize tracking here
        # For now, just highlight the selection
        if self.current_frame is not None:
            display_frame = self.current_frame.copy()
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            self.video_widget.set_frame(display_frame)
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop timers
        self.timer.stop()
        self.fps_timer.stop()
        
        # Release video capture
        if self.cap is not None:
            self.cap.release()
            
        # Accept the close event
        event.accept()


if __name__ == "__main__":
    try:
        # Enable CUDA error context
        if 'CUDA_LAUNCH_BLOCKING' not in os.environ:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Enable debugging information for CUDA
        if torch.cuda.is_available():
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
            
            # Force PyTorch to use the older cuDNN convolution algorithms which are more reliable
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # Check memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"Total GPU memory: {gpu_memory / 1e9:.2f} GB")
        
        # Create application
        app = QApplication(sys.argv)
        
        # Create and show main window
        main_window = AdvancedCarTrackingApp()
        main_window.show()
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to start in CPU-only mode if there was an error
        print("\nAttempting to start in CPU-only mode...")
        
        # Force CPU mode
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.cuda.is_available = lambda: False
        
        # Create application
        app = QApplication(sys.argv)
        
        # Create and show main window (CPU mode)
        main_window = AdvancedCarTrackingApp()
        main_window.show()
        
        # Run application
        sys.exit(app.exec_())