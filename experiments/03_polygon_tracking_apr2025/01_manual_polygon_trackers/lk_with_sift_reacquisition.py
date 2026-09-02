"""AdSwapAI R&D, 2025-04-01: click 4 board corners on frame 1, Lucas-Kanade tracks the corners,
with SIFT re-detection of the board when LK loses them; perspective-warp + alpha-blend an ad."""

import cv2
import numpy as np
import os
from datetime import datetime

VIDEO_PATH = "data/adVideo1.mp4"   # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"

class RobustAdReplacer:
    def __init__(self, video_path, replacement_image_path, output_directory):
        # Video setup
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_directory = output_directory

        # Replacement image (with alpha if available)
        self.replacement_image = self.load_replacement_image(replacement_image_path)
        
        # For corner tracking
        self.initial_corners = None    # (4,1,2) float32 from user selection
        self.tracked_corners = None
        self.prev_gray = None
        
        # Lucas-Kanade params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Corner selection window
        self.window_name = "Select 4 Corners (TL, TR, BR, BL) - Press 'q' to quit"
        self.selected_points = []
        
        # Oversampling for higher quality warp
        self.oversampling = 3.0
        
        # --- SIFT (feature-based) re-detection ---
        # We'll store a reference region of the ad board from the first frame
        # and detect keypoints/descriptors. When corners go off-screen or become invalid,
        # we attempt to reacquire them via SIFT matching + findHomography.
        self.sift = cv2.SIFT_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Reference data (ROI, keypoints, descriptors)
        self.ref_kp = None
        self.ref_desc = None
        self.ref_img = None  # grayscale ROI
        self.ref_polygon = None  # the original polygon (for area reference)
        
        # RANSAC threshold for findHomography
        self.ransac_thresh = 5.0
        
    def load_replacement_image(self, image_path):
        """Load the replacement image, preserving alpha if present."""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # If only 3 channels, add full alpha
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255
            img = cv2.merge((b, g, r, alpha))
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        """Collect four corner points from mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selected_points) < 4:
                self.selected_points.append((x, y))
                print(f"Corner {len(self.selected_points)} selected: ({x}, {y})")
    
    def select_corners(self):
        """User selects 4 corners on the first frame (in order: TL, TR, BR, BL)."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read first frame from video.")
        clone = frame.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Click on the 4 corners of the ad board in order: TL, TR, BR, BL")
        while True:
            disp = clone.copy()
            # Draw selected corners
            for pt in self.selected_points:
                cv2.circle(disp, pt, 5, (0, 255, 0), -1)
            if len(self.selected_points) > 1:
                cv2.polylines(disp, [np.array(self.selected_points, dtype=np.int32)], True, (255,0,0), 2)
            cv2.imshow(self.window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if len(self.selected_points) == 4:
                break
        
        cv2.destroyWindow(self.window_name)
        if len(self.selected_points) != 4:
            raise ValueError("Four corner points not selected.")
        
        self.initial_corners = np.array(self.selected_points, dtype=np.float32).reshape(-1, 1, 2)
        self.tracked_corners = self.initial_corners.copy()
        self.prev_gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
        
        # Store reference data for SIFT re-detection
        self.setup_sift_reference(clone, self.initial_corners)
        
        # Reset video to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def setup_sift_reference(self, frame_bgr, corners):
        """
        Crop the ROI around the selected polygon in the first frame,
        detect SIFT keypoints/descriptors. We'll use this for reacquiring
        corners when they go off-screen.
        """
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Get bounding rect for the polygon
        poly_pts = corners.reshape(-1,2).astype(np.int32)
        x, y, w, h = cv2.boundingRect(poly_pts)
        if w < 5 or h < 5:
            return
        
        # Crop ROI
        roi_gray = frame_gray[y:y+h, x:x+w].copy()
        
        # Create a mask so we only detect keypoints inside the polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_pts = poly_pts - [x, y]
        cv2.fillPoly(mask, [shifted_pts], 255)
        
        # Detect SIFT in the ROI
        kp, desc = self.sift.detectAndCompute(roi_gray, mask)
        
        # Adjust keypoint coordinates to full frame
        for k in kp:
            k.pt = (k.pt[0] + x, k.pt[1] + y)
        
        self.ref_img = frame_gray
        self.ref_kp = kp
        self.ref_desc = desc
        self.ref_polygon = poly_pts  # store the original polygon for area reference

    def is_valid_quad(self, pts):
        """
        Check if the polygon is still 'valid':
         - not extremely out of frame
         - has a reasonable area
        """
        pts = pts.reshape(-1,2)
        margin = 0.1
        # Check if corners are within some margin of the screen
        if (np.any(pts[:,0] < -margin*self.width) or 
            np.any(pts[:,0] > (1+margin)*self.width) or
            np.any(pts[:,1] < -margin*self.height) or
            np.any(pts[:,1] > (1+margin)*self.height)):
            return False
        
        # Check area
        area = cv2.contourArea(pts.astype(np.float32))
        if area < 100:  # too small
            return False
        return True
    
    def get_warped_replacement(self, dst_pts):
        """
        Warp the replacement image (with oversampling) onto the quadrilateral dst_pts.
        """
        rep_h, rep_w = self.replacement_image.shape[:2]
        # Oversample
        up_w = int(rep_w * self.oversampling)
        up_h = int(rep_h * self.oversampling)
        upsampled = cv2.resize(self.replacement_image, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Source points (corners of upsampled replacement)
        src_pts = np.array([
            [0,0],
            [up_w-1, 0],
            [up_w-1, up_h-1],
            [0, up_h-1]
        ], dtype=np.float32)
        
        dst_pts = dst_pts.reshape(4,2).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        warped = cv2.warpPerspective(upsampled, M, (self.width, self.height),
                                     flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT)
        return warped

    def apply_overlay(self, frame, warped):
        """
        Blend the warped image onto the frame using alpha channel.
        """
        if warped.shape[2] == 4:
            alpha = warped[:,:,3] / 255.0
            for c in range(3):
                frame[:,:,c] = frame[:,:,c]*(1 - alpha) + warped[:,:,c]*alpha
        else:
            mask = warped > 0
            frame[mask] = warped[mask]
        return frame

    def sift_redetect(self, frame_gray):
        """
        Try to reacquire the corners using SIFT feature matching
        between the reference ROI and the current frame.
        If successful, update self.tracked_corners.
        """
        if self.ref_desc is None or self.ref_kp is None:
            return False
        
        # Detect SIFT in current frame
        kp2, desc2 = self.sift.detectAndCompute(frame_gray, None)
        if desc2 is None or len(kp2) < 10:
            return False
        
        # Match with BFMatcher
        matches = self.matcher.knnMatch(self.ref_desc, desc2, k=2)
        good = []
        ratio_thresh = 0.75
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        
        if len(good) < 8:
            return False
        
        # Construct src/dst arrays for findHomography
        ref_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        frame_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        
        H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return False
        
        # Warp the original polygon corners
        poly = self.ref_polygon.reshape(-1,1,2).astype(np.float32)
        new_poly = cv2.perspectiveTransform(poly, H)
        
        # We expect 4 corners in the original polygon
        if new_poly.shape[0] != 4:
            return False
        
        self.tracked_corners = new_poly
        return True

    def process_video(self):
        """
        Main loop:
         - Use optical flow for corners
         - If corners become invalid, skip overlay or attempt SIFT re-detection
         - If re-detection succeeds and corners are valid, overlay
        """
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
            
            # 1) Optical Flow update
            new_corners, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.tracked_corners, None, **self.lk_params
            )
            
            # If enough corners tracked, update
            if new_corners is not None and np.count_nonzero(status) >= 3:
                self.tracked_corners = new_corners
            else:
                # We lost tracking - try SIFT reacquisition
                success = self.sift_redetect(frame_gray)
                if not success:
                    # If reacquisition fails, skip overlay
                    out.write(frame)
                    self.prev_gray = frame_gray
                    frame_idx += 1
                    continue
            
            self.prev_gray = frame_gray
            
            # 2) Check validity of corners
            if not self.is_valid_quad(self.tracked_corners):
                # Attempt SIFT re-detect if corners are invalid
                success = self.sift_redetect(frame_gray)
                if not success or not self.is_valid_quad(self.tracked_corners):
                    # skip overlay if still invalid
                    out.write(frame)
                    frame_idx += 1
                    continue
            
            # 3) If valid, do high-quality warp + alpha blend
            warped_replacement = self.get_warped_replacement(self.tracked_corners)
            blended_frame = self.apply_overlay(frame, warped_replacement)
            
            out.write(blended_frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{self.total_frames} frames...")
        
        out.release()
        self.cap.release()
        print("Processing complete. Output saved to:", output_path)

def main():
    video_path = VIDEO_PATH
    replacement_image_path = REPLACEMENT_PATH
    output_directory = OUTPUT_DIR

    replacer = RobustAdReplacer(video_path, replacement_image_path, output_directory)
    replacer.select_corners()
    replacer.process_video()

if __name__ == "__main__":
    main()
