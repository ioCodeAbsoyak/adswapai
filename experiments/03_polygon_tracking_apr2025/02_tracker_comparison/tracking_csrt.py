"""AdSwapAI R&D, 2025-04-02: VideoPolygonMapperCSRT - a user-drawn polygon tracked via CSRT
on the polygon's bounding box (translation/scale only, relative corner offsets reapplied)."""

import cv2
import numpy as np
import os
from datetime import datetime

VIDEO_PATH = "data/adVideo1.mp4"   # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"

class VideoPolygonMapperCSRT:
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
            
            # Convert to RGBA if not already (for potential alpha blending)
            if len(self.replacement_image.shape) == 3:
                self.replacement_image = cv2.cvtColor(self.replacement_image, cv2.COLOR_BGR2BGRA)
            
            # Oversampling factor for improved overlay quality
            self.oversampling = 3.0
            
            # Polygon selection variables
            self.points = []
            self.polygon_completed = False
            
            # Tracking variables
            self.original_polygon = None
            self.current_polygon = None
            self.relative_polygon = None  # Relative coordinates of original polygon's corners
            self.warped_image = None
            
            # CSRT tracker variable
            self.tracker = None
            
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
        """Initialize tracking with the selected polygon using CSRT."""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read initial frame for tracking setup.")
            return
        self.original_polygon = np.array(self.points, dtype=np.int32)
        self.current_polygon = self.original_polygon.copy()
        
        # Compute bounding rectangle from the original polygon for tracker initialization
        bbox = cv2.boundingRect(self.original_polygon)
        print(f"Initializing CSRT tracker with bbox: {bbox}")
        
        # Compute relative coordinates for each corner with respect to the bounding box
        self.relative_polygon = []
        for pt in self.original_polygon:
            rel_x = (pt[0] - bbox[0]) / bbox[2]
            rel_y = (pt[1] - bbox[1]) / bbox[3]
            self.relative_polygon.append([rel_x, rel_y])
        self.relative_polygon = np.array(self.relative_polygon, dtype=np.float32)
        
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError as e:
            raise Exception("CSRT tracker not available. Please update to the latest opencv-contrib-python package.") from e
        self.tracker.init(frame, bbox)
        
        # Prepare initial warped overlay
        self.update_warped_image()

    def skip_frames(self, num_frames=10):
        """Skip a specified number of frames."""
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                return None
            self.frame_count += 1
        return frame

    def update_tracking_csrt(self, frame):
        """Update tracking using the CSRT tracker and update polygon using relative coordinates."""
        success, bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            # Reconstruct the polygon from the relative coordinates and new bbox
            new_polygon = []
            for rel in self.relative_polygon:
                new_x = int(x + rel[0] * w)
                new_y = int(y + rel[1] * h)
                new_polygon.append([new_x, new_y])
            self.current_polygon = np.array(new_polygon, dtype=np.int32)
            # Update overlay based on the new polygon
            self.update_warped_image()
            return True
        else:
            return False

    def update_warped_image(self):
        """Update the warped replacement image with high-quality oversampling."""
        if self.current_polygon is None or len(self.current_polygon) < 4:
            return
        try:
            # Compute bounding rectangle of current polygon
            x, y, w, h = cv2.boundingRect(self.current_polygon)
            w = max(1, w)
            h = max(1, h)
            
            # Replacement image dimensions
            img_h, img_w = self.replacement_image.shape[:2]
            
            # Calculate scale factor to fit replacement image (with oversampling)
            scale = min(w / img_w, h / img_h)
            target_w = int(img_w * scale * self.oversampling)
            target_h = int(img_h * scale * self.oversampling)
            
            # Resize replacement image with high-quality interpolation
            resized_img = cv2.resize(self.replacement_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Destination points are the current polygon's four corners
            dst_points = np.float32(self.current_polygon[:4])
            
            # Source points: corners of the resized replacement image
            src_points = np.float32([[0, 0],
                                     [target_w - 1, 0],
                                     [target_w - 1, target_h - 1],
                                     [0, target_h - 1]])
            
            # Compute perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Warp the resized image onto the full frame
            warped = cv2.warpPerspective(resized_img, M, (self.width, self.height),
                                         flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT)
            
            # If the warped image has an alpha channel, convert it to BGR for final overlay
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
            
            # Wait for polygon selection
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
                        tracking_success = self.update_tracking_csrt(frame)
                        # Only overlay if tracking succeeded
                        if tracking_success and self.warped_image is not None:
                            # Create a mask from the current polygon and overlay warped image
                            mask = np.zeros((self.height, self.width), dtype=np.uint8)
                            cv2.fillPoly(mask, [self.current_polygon], 255)
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
        
        mapper = VideoPolygonMapperCSRT(video_path, replacement_image_path)
        mapper.process_video(output_directory)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
