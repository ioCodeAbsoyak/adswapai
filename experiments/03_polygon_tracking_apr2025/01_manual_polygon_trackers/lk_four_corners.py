"""AdSwapAI R&D, 2025-04-01: click 4 board corners on frame 1, Lucas-Kanade tracks the corners,
perspective-warp + alpha-blend an ad onto the tracked quadrilateral."""

import cv2
import numpy as np
import os
from datetime import datetime

VIDEO_PATH = "data/adVideo1.mp4"   # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"

class FootballFieldAdReplacer:
    def __init__(self, video_path, replacement_image_path, output_directory):
        # Initialize video capture
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_directory = output_directory

        # Load replacement image (with alpha if available)
        self.replacement_image = self.load_replacement_image(replacement_image_path)
        
        # Tracking state variables (we use four corners)
        self.initial_corners = None  # (4,1,2) float32 array of the selected points
        self.tracked_corners = None  # Current positions of the four corners
        self.prev_gray = None        # Previous grayscale frame for optical flow
        
        # Lucas–Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # For mouse-based corner selection
        self.window_name = "Select 4 Corners (Order: TL, TR, BR, BL) - Press 'q' to quit"
        self.selected_points = []

    def load_replacement_image(self, image_path):
        """Load the replacement image, preserving alpha if available."""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        # If image has only 3 channels, add a full opaque alpha channel.
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255
            img = cv2.merge((b, g, r, alpha))
        return img

    def mouse_callback(self, event, x, y, flags, param):
        """Collect four corner points via mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selected_points) < 4:
                self.selected_points.append((x, y))
                print(f"Corner {len(self.selected_points)} selected: ({x}, {y})")
    
    def select_corners(self):
        """Display the first frame and let the user click 4 corners of the ad board."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read first frame from video.")
        clone = frame.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Click on the four corners of the ad board in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
        while True:
            disp = clone.copy()
            # Draw the selected points
            for pt in self.selected_points:
                cv2.circle(disp, pt, 5, (0, 255, 0), -1)
            cv2.imshow(self.window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            # Press 'q' to cancel
            if key == ord('q'):
                break
            if len(self.selected_points) == 4:
                break
        
        cv2.destroyWindow(self.window_name)
        if len(self.selected_points) != 4:
            raise ValueError("Four corner points were not selected.")
        
        # Save the initial corner positions as a (4,1,2) float32 array.
        self.initial_corners = np.array(self.selected_points, dtype=np.float32).reshape(-1, 1, 2)
        self.tracked_corners = self.initial_corners.copy()
        # Set the previous gray frame from the first frame
        self.prev_gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def process_video(self):
        """Process the video frame by frame, updating the four-corner tracker and overlaying the replacement image."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_directory, f"{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Track the four corners using Lucas–Kanade optical flow
            new_corners, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.tracked_corners, None, **self.lk_params
            )
            # If tracking was successful for at least three corners, update the tracked corners
            if new_corners is not None and np.count_nonzero(status) >= 3:
                self.tracked_corners = new_corners
            else:
                print(f"Tracking failure at frame {frame_idx}. Using previous corners.")
            self.prev_gray = frame_gray.copy()
            
            # Compute homography from replacement image to the current quadrilateral.
            rep_h, rep_w = self.replacement_image.shape[:2]
            src_pts = np.array([[0, 0],
                                [rep_w - 1, 0],
                                [rep_w - 1, rep_h - 1],
                                [0, rep_h - 1]], dtype=np.float32)
            dst_pts = self.tracked_corners.reshape(4, 2)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Warp the replacement image onto a canvas of the same size as the frame
            warped_replacement = cv2.warpPerspective(
                self.replacement_image, M, (self.width, self.height),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT
            )
            
            # Blend the warped replacement image with the original frame using the alpha channel.
            if warped_replacement.shape[2] == 4:
                alpha = warped_replacement[:, :, 3] / 255.0
                for c in range(3):
                    frame[:, :, c] = frame[:, :, c] * (1 - alpha) + warped_replacement[:, :, c] * alpha
            else:
                # Fallback if no alpha channel exists (direct overlay)
                mask = warped_replacement > 0
                frame[mask] = warped_replacement[mask]
            
            out.write(frame)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{self.total_frames} frames...")
        
        out.release()
        self.cap.release()
        print("Processing complete. Output saved to:", output_path)

def main():
    # Set your file paths here
    video_path = VIDEO_PATH
    replacement_image_path = REPLACEMENT_PATH
    output_directory = OUTPUT_DIR

    replacer = FootballFieldAdReplacer(video_path, replacement_image_path, output_directory)
    replacer.select_corners()
    replacer.process_video()

if __name__ == "__main__":
    main()
