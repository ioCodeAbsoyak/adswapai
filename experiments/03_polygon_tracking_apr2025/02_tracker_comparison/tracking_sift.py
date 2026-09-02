"""AdSwapAI R&D, 2025-04-02: VideoPolygonMapper - a user-drawn polygon tracked by SIFT
feature matching + RANSAC homography, with the replacement image perspective-warped onto it."""

import cv2
import numpy as np
import os
from datetime import datetime

VIDEO_PATH = "data/adVideo1.mp4"   # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"

class VideoPolygonMapper:
    def __init__(self, video_path, replacement_image_path):
        try:
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            print(f"Video properties: {self.width}x{self.height}, {self.fps} FPS")
            
            # Load replacement image with high quality
            self.replacement_image = cv2.imread(replacement_image_path)
            if self.replacement_image is None:
                raise ValueError(f"Could not load replacement image: {replacement_image_path}")
            print(f"Replacement image loaded: {self.replacement_image.shape}")
            
            # Convert to RGBA if not already
            if len(self.replacement_image.shape) == 3:
                self.replacement_image = cv2.cvtColor(self.replacement_image, cv2.COLOR_BGR2BGRA)
            
            # Oversampling factor for improved overlay quality
            self.oversampling = 3.0
            
            # Polygon selection variables
            self.points = []
            self.drawing = False
            self.polygon_completed = False
            
            # Tracking variables
            self.original_frame = None
            self.original_polygon = None
            self.current_polygon = None
            self.warped_image = None
            self.tracking_quality = 1.0  # Track quality (0-1)
            
            # Feature detection using SIFT
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            
            # Output video writer
            self.output_writer = None
            
            # Window name
            self.window_name = "Polygon Selector (Left click: add point, Right click: complete, N: skip 10 frames, q: quit)"
            
            # Frame counter
            self.frame_count = 0
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            raise

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Point added: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) >= 3:
                self.polygon_completed = True
                print("Polygon completed!")
                self.setup_tracking()
    
    def setup_tracking(self):
        """Initialize tracking with the selected polygon."""
        ret, self.original_frame = self.cap.read()
        if not ret:
            print("Failed to read initial frame for tracking setup.")
            return
        self.original_polygon = np.array(self.points, dtype=np.int32)
        self.current_polygon = self.original_polygon.copy()
        
        # Get bounding rectangle for the polygon
        x, y, w, h = cv2.boundingRect(self.original_polygon)
        
        # Extract ROI for feature detection
        roi = self.original_frame[y:y+h, x:x+w].copy()
        
        # Create mask for the ROI
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_poly = self.original_polygon - np.array([x, y])
        cv2.fillPoly(roi_mask, [shifted_poly], 255)
        
        # Detect keypoints in the ROI
        self.original_keypoints, self.original_descriptors = self.detector.detectAndCompute(roi, roi_mask)
        
        # Adjust keypoint coordinates to full frame
        for kp in self.original_keypoints:
            kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
        
        # Prepare initial warped replacement overlay
        self.update_warped_image()
    
    def skip_frames(self, num_frames=10):
        """Skip a specified number of frames."""
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                return None
            self.frame_count += 1
        return frame

    def update_tracking(self, frame):
        """Update polygon tracking using SIFT feature matching."""
        if not hasattr(self, 'original_keypoints') or not self.original_keypoints:
            return False

        # Detect keypoints in current frame
        frame_keypoints, frame_descriptors = self.detector.detectAndCompute(frame, None)
        if not frame_keypoints or frame_descriptors is None:
            self.tracking_quality = 0
            return False

        # Match keypoints using kNN matching
        matches = self.matcher.knnMatch(self.original_descriptors, frame_descriptors, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            self.tracking_quality = len(good_matches) / 8.0
            return False

        # Get matched keypoints
        src_pts = np.float32([self.original_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches])

        # Calculate homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            poly_points = self.original_polygon.reshape(-1, 1, 2).astype(np.float32)
            transformed_poly = cv2.perspectiveTransform(poly_points, H)
            self.current_polygon = transformed_poly.reshape(-1, 2).astype(np.int32)

            self.tracking_quality = np.sum(mask) / len(good_matches)

            # Update overlay image based on new polygon
            self.update_warped_image()
            return True

        self.tracking_quality = 0
        return False

    def update_warped_image(self):
        """Update the warped replacement image with high-quality oversampling and reduced extra processing."""
        if self.current_polygon is None or len(self.current_polygon) < 4:
            return
        try:
            # Compute bounding rectangle of the current polygon
            x, y, w, h = cv2.boundingRect(self.current_polygon)
            w = max(1, w)
            h = max(1, h)
            
            # Replacement image dimensions
            img_h, img_w = self.replacement_image.shape[:2]
            
            # Calculate scale factor to fit replacement image within the polygon (with oversampling)
            scale = min(w / img_w, h / img_h)
            target_w = int(img_w * scale * self.oversampling)
            target_h = int(img_h * scale * self.oversampling)
            
            # Resize replacement image using high-quality interpolation
            resized_img = cv2.resize(self.replacement_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Determine destination points for the perspective transform
            # Use the first 4 points of the polygon; if more, fallback to bounding box.
            if len(self.current_polygon) >= 4:
                dst_points = np.float32(self.current_polygon[:4])
            else:
                dst_points = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            
            # Define source points: corners of the resized replacement image
            src_points = np.float32([[0, 0],
                                    [target_w - 1, 0],
                                    [target_w - 1, target_h - 1],
                                    [0, target_h - 1]])
            
            # Compute perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Warp the resized image onto the full frame size directly
            warped = cv2.warpPerspective(resized_img, M, (self.width, self.height),
                                        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT)
            
            # If the warped image has an alpha channel, convert it to BGR for final overlay (if desired)
            if warped.shape[2] == 4:
                self.warped_image = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)
            else:
                self.warped_image = warped
            
        except Exception as e:
            print(f"Error updating warped image: {str(e)}")
            print(f"Current polygon shape: {self.current_polygon.shape if self.current_polygon is not None else 'None'}")
            self.warped_image = None

    def process_video(self, output_dir):
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Could not read video!")
                return
            print("First frame read successfully")
            
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            
            print(f"Output video: {output_path}")
            print("Draw your polygon on the first frame. Right-click to complete when done.")
            
            # Wait for polygon selection completion
            while not self.polygon_completed:
                display_frame = frame.copy()
                if len(self.points) > 0:
                    for point in self.points:
                        cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                    for i in range(len(self.points) - 1):
                        cv2.line(display_frame, self.points[i], self.points[i+1], (0, 255, 0), 2)
                    if len(self.points) >= 3:
                        cv2.line(display_frame, self.points[-1], self.points[0], (0, 255, 0), 2)
                cv2.imshow(self.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return
            
            print("Polygon completed! Starting video processing...")
            
            # Main processing loop
            while True:
                try:
                    display_frame = frame.copy()
                    
                    if self.polygon_completed:
                        tracking_success = self.update_tracking(frame)
                        # Only overlay if tracking quality is acceptable (and thus, pane is in-screen)
                        if tracking_success and self.tracking_quality > 0.3:
                            if self.warped_image is not None:
                                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                                cv2.fillPoly(mask, [self.current_polygon], 255)
                                # Alpha blend overlay onto frame
                                if len(self.warped_image.shape) == 3 and self.warped_image.shape[2] == 4:
                                    alpha = self.warped_image[:, :, 3] / 255.0
                                    alpha = np.expand_dims(alpha, axis=-1)
                                    display_frame = display_frame * (1 - alpha) + self.warped_image[:, :, :3] * alpha
                                else:
                                    mask_3ch = cv2.merge([mask, mask, mask])
                                    np.copyto(display_frame, self.warped_image, where=mask_3ch.astype(bool))
                    
                    cv2.imshow(self.window_name, display_frame)
                    if self.polygon_completed:
                        self.output_writer.write(display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('n'):
                        frame = self.skip_frames(10)
                        if frame is None:
                            break
                        continue
                    
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                        
                except Exception as e:
                    print(f"Error in main loop: {str(e)}")
                    break
            
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
        finally:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            if hasattr(self, 'output_writer') and self.output_writer is not None:
                self.output_writer.release()
            cv2.destroyAllWindows()

def main():
    try:
        video_path = VIDEO_PATH
        replacement_image_path = REPLACEMENT_PATH
        output_directory = OUTPUT_DIR
        
        if not os.path.exists(video_path):
            print(f"Error: {video_path} not found!")
            return
        if not os.path.exists(replacement_image_path):
            print(f"Error: {replacement_image_path} not found!")
            return
        
        mapper = VideoPolygonMapper(video_path, replacement_image_path)
        mapper.process_video(output_directory)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
