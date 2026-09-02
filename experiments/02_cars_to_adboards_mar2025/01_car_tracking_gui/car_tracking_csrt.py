"""AdSwapAI R&D, 2025-03-01: PyQt5 GUI to drag a box around a car and track it with OpenCV CSRT."""

import cv2
import time
import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor


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


class TrackedObject:
    """Class representing an object that is being tracked across video frames"""
    
    def __init__(self, initial_frame, bbox, object_id):
        """
        Initialize a tracked object
        
        Args:
            initial_frame: First frame where the object appears
            bbox: Initial bounding box (x, y, w, h)
            object_id: Unique identifier
        """
        self.id = object_id
        self.bbox = bbox  # (x, y, w, h)
        
        # Initialize tracker (using CSRT for better accuracy)
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(initial_frame, bbox)
        
        # Flag to indicate if tracking is active
        self.is_active = True
        
    def update(self, frame):
        """
        Update tracking on new frame
        
        Args:
            frame: New video frame
            
        Returns:
            bool: True if tracking was successful
        """
        if not self.is_active:
            return False
            
        success, bbox = self.tracker.update(frame)
        
        if success:
            # Convert to integers and update bbox
            self.bbox = tuple(map(int, bbox))
        else:
            self.is_active = False
            
        return success


class CarTrackingApp(QMainWindow):
    """Main application for car tracking"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Car Tracking Application")
        self.setMinimumSize(800, 600)
        
        # Video processing variables
        self.cap = None
        self.current_frame = None
        self.current_frame_num = 0
        self.tracked_objects = []
        self.next_object_id = 1
        
        # Video properties
        self.fps = 0
        self.frame_count = 0
        self.video_width = 0
        self.video_height = 0
        
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
        self.reset_btn = QPushButton("Reset Tracking")
        
        # Disable buttons initially
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        
        # Connect signals
        self.load_btn.clicked.connect(self.load_video)
        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)
        self.reset_btn.clicked.connect(self.reset_tracking)
        
        # Add buttons to layout
        top_controls.addWidget(self.load_btn)
        top_controls.addWidget(self.play_btn)
        top_controls.addWidget(self.pause_btn)
        top_controls.addWidget(self.reset_btn)
        top_controls.addStretch()
        
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
        self.statusBar().showMessage("Ready. Load a video file to start.")
        
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
            self.tracked_objects = []
            self.next_object_id = 1
            
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
                self.reset_btn.setEnabled(True)
            else:
                self.statusBar().showMessage("Error: Could not read video frames")
    
    def next_frame(self):
        """Process and display the next frame"""
        if self.cap is None or not self.cap.isOpened():
            return
            
        # Read the next frame
        ret, frame = self.cap.read()
        
        if not ret:
            # End of video
            self.pause_video()
            self.statusBar().showMessage("End of video reached")
            return
            
        self.current_frame_num += 1
        self.current_frame = frame.copy()
        
        # Update tracking for all objects
        processed_frame = self.current_frame.copy()
        
        for obj in self.tracked_objects:
            if obj.is_active:
                success = obj.update(self.current_frame)
                
                if success:
                    # Draw bounding box for visualization
                    x, y, w, h = obj.bbox
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Car {obj.id}", (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add FPS information
        cv2.putText(processed_frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the processed frame
        self.video_widget.set_frame(processed_frame)
        
        # Update status bar
        self.statusBar().showMessage(
            f"Frame: {self.current_frame_num}/{self.frame_count}, "
            f"Tracking {sum(1 for obj in self.tracked_objects if obj.is_active)} cars"
        )
        
        # Update FPS counter
        self.frames_processed += 1
    
    def play_video(self):
        """Start video playback"""
        if self.cap is None:
            return
            
        # Calculate interval based on video FPS
        interval = int(1000 / self.fps)
        self.timer.start(interval)
        
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
    
    def pause_video(self):
        """Pause video playback"""
        self.timer.stop()
        
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
    
    def update_fps(self):
        """Update FPS calculation"""
        self.current_fps = self.frames_processed
        self.frames_processed = 0
        
        self.fps_label.setText(f"FPS: {self.current_fps:.1f}")
    
    def reset_tracking(self):
        """Reset all tracking data"""
        if self.cap is None:
            return
            
        # Reset tracking variables
        self.tracked_objects = []
        self.next_object_id = 1
        
        # Pause video if playing
        if self.timer.isActive():
            self.pause_video()
        
        # Reset to current frame
        self.next_frame()
        
        self.statusBar().showMessage("Tracking data reset")
    
    def handle_selection(self, rect):
        """Handle user selection on the video frame"""
        if self.current_frame is None:
            return
            
        # Convert QRect to OpenCV bbox format (x, y, w, h)
        bbox = (rect.x(), rect.y(), rect.width(), rect.height())
        
        # Create a new tracked object
        tracked_obj = TrackedObject(
            self.current_frame.copy(), 
            bbox, 
            self.next_object_id
        )
        
        self.tracked_objects.append(tracked_obj)
        self.next_object_id += 1
        
        # Update the display
        self.next_frame()
        
        # Show message
        self.statusBar().showMessage(f"Added new car tracking (ID: {tracked_obj.id})")


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    window = CarTrackingApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()