"""AdSwapAI R&D, 2025-04-13: YOLO picks the board on frame 1, SIFT homography tracks
it afterwards."""

import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

VIDEO_PATH = "data/adVideo2.mp4"   # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"
MODEL_PATH = "best.pt"   # single-class YOLOv8s-seg "billboard" model, see docs/assets.md

class AutoVideoPolygonMapper:
    def __init__(self, video_path, replacement_image_path):
        try:
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            print(f"Video properties: {self.width}x{self.height}, {self.fps} FPS")

            # Load replacement image (with high quality and alpha channel if available)
            self.replacement_image = cv2.imread(replacement_image_path, cv2.IMREAD_UNCHANGED)
            if self.replacement_image is None:
                raise ValueError(f"Could not load replacement image: {replacement_image_path}")
            print(f"Replacement image loaded: {self.replacement_image.shape}")
            # Convert image to RGBA if it is not already
            if self.replacement_image.shape[2] == 3:
                self.replacement_image = cv2.cvtColor(self.replacement_image, cv2.COLOR_BGR2BGRA)

            # Initialize YOLO model for automatic detection
            self.yolo_model = YOLO(MODEL_PATH)

            # Tracking variables
            self.original_polygon = None  # polygon detected in first frame
            self.current_polygon = None   # updated polygon (tracked)
            self.warped_image = None
            self.tracking_quality = 1.0

            # Feature detector (SIFT) and descriptor matcher (BFMatcher)
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self.original_keypoints = []
            self.original_descriptors = None

            # Output video writer
            self.output_writer = None

            # Frame counter
            self.frame_count = 0

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            if self.cap:
                self.cap.release()
            raise

    def auto_detect_polygon(self, frame):
        """
        Automatically detect an ad region in the first frame using YOLO.
        The candidate region is chosen based on being less than 50% of the frame width
        and having the highest confidence.
        """
        results = self.yolo_model(frame, augment=False)[0]
        candidate_polygon = None
        candidate_confidence = 0
        for i, box in enumerate(results.boxes):
            xyxy = box.xyxy.cpu().numpy().flatten()
            x1, y1, x2, y2 = xyxy
            target_width = int(x2 - x1)
            # Only consider detections smaller than half the frame width
            if target_width < 0.5 * self.width:
                conf = box.conf.cpu().numpy()[0]
                if conf > candidate_confidence:
                    candidate_confidence = conf
                    if results.masks is not None and i < len(results.masks.xy):
                        pts = results.masks.xy[i].reshape(-1, 2).astype(np.float32)
                        if pts.shape[0] >= 4:
                            epsilon = 0.02 * cv2.arcLength(pts, True)
                            approx = cv2.approxPolyDP(pts, epsilon, True)
                            if len(approx) >= 4:
                                candidate_polygon = approx.reshape(-1, 2).astype(np.int32)
                            else:
                                candidate_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                        else:
                            candidate_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    else:
                        candidate_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        if candidate_polygon is None:
            candidate_polygon = np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]], dtype=np.int32)
        print("Auto-detected polygon:", candidate_polygon)
        return candidate_polygon

    def setup_tracking(self, frame, polygon):
        """
        Initialize tracking using the detected polygon from YOLO.
        Extract the region of interest (ROI) and compute SIFT features.
        """
        self.original_polygon = polygon
        self.current_polygon = polygon.copy()
        # Compute bounding rectangle for the polygon
        x, y, w, h = cv2.boundingRect(self.original_polygon)
        roi = frame[y:y+h, x:x+w].copy()
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_poly = self.original_polygon - np.array([x, y])
        cv2.fillPoly(roi_mask, [shifted_poly], 255)
        self.original_keypoints, self.original_descriptors = self.detector.detectAndCompute(roi, roi_mask)
        # Adjust keypoint coordinates relative to the full frame
        for kp in self.original_keypoints:
            kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
        self.update_warped_image()

    def polygon_iou(self, poly1, poly2):
        """
        Compute the Intersection over Union (IoU) of two polygons.
        Polygons are expected to be numpy arrays of shape (N,2).
        """
        poly1 = poly1.astype(np.float32)
        poly2 = poly2.astype(np.float32)
        area1 = cv2.contourArea(poly1)
        area2 = cv2.contourArea(poly2)
        inter_area, _ = cv2.intersectConvexConvex(poly1, poly2)
        union_area = area1 + area2 - inter_area
        if union_area == 0:
            return 0
        return inter_area / union_area

    def update_tracking(self, frame):
        """
        Update the tracked polygon using SIFT features and homography estimation.
        Apply smoothing if the difference between the new and previous polygon is small.
        """
        frame_keypoints, frame_descriptors = self.detector.detectAndCompute(frame, None)
        if frame_keypoints is None or frame_descriptors is None or len(frame_keypoints) == 0:
            self.tracking_quality = 0
            return False

        matches = self.matcher.knnMatch(self.original_descriptors, frame_descriptors, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        if len(good_matches) < 8:
            self.tracking_quality = len(good_matches) / 8.0
            return False

        src_pts = np.float32([self.original_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches])
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            poly_points = self.original_polygon.reshape(-1, 1, 2).astype(np.float32)
            new_polygon = cv2.perspectiveTransform(poly_points, H)
            new_polygon = new_polygon.reshape(-1, 2).astype(np.int32)
            # Compute IoU between current polygon and newly calculated one
            if self.current_polygon is not None:
                iou_score = self.polygon_iou(self.current_polygon, new_polygon)
            else:
                iou_score = 0
            # If the new polygon is very similar to the current one, do not update
            if iou_score > 0.95:
                new_polygon = self.current_polygon.copy()
            else:
                # Otherwise, smooth the update with an exponential moving average
                smoothing_factor = 0.2
                new_polygon = (smoothing_factor * new_polygon + (1 - smoothing_factor) * self.current_polygon).astype(np.int32)
            self.current_polygon = new_polygon
            self.tracking_quality = float(np.sum(mask)) / float(len(good_matches))
            self.update_warped_image()
            return True
        else:
            self.tracking_quality = 0
            return False

    def update_warped_image(self):
        """
        Update the warped replacement image based on the current tracked polygon.
        """
        if self.current_polygon is None or len(self.current_polygon) < 4:
            return
        try:
            x, y, w, h = cv2.boundingRect(self.current_polygon)
            w = max(1, w)
            h = max(1, h)
            img_h, img_w = self.replacement_image.shape[:2]
            scale = min(w / img_w, h / img_h)
            target_w = int(img_w * scale)
            target_h = int(img_h * scale)
            resized_img = cv2.resize(self.replacement_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            # Use a rectangular region based on bounding box if the polygon has extra points
            if len(self.current_polygon) > 4:
                dst_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
            else:
                dst_points = self.current_polygon.astype(np.float32)
            src_points = np.array([
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1]
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            warped = cv2.warpPerspective(resized_img, M, (self.width, self.height),
                                         flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT)
            warped = warped.astype(np.uint8)
            alpha_channel = warped[:, :, 3:4] / 255.0
            canvas = canvas * (1 - alpha_channel) + warped * alpha_channel
            self.warped_image = canvas.astype(np.uint8)
            if self.warped_image.shape[2] == 4:
                self.warped_image = cv2.cvtColor(self.warped_image, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"Error updating warped image: {str(e)}")
            self.warped_image = None

    def process_video(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Could not read video!")
                return

            # Use YOLO to automatically detect the ad region in the first frame
            detected_polygon = self.auto_detect_polygon(frame)
            self.setup_tracking(frame, detected_polygon)
            print("Initial ad region established from YOLO detection.")

            # Prepare the output video file
            output_dir = OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            print(f"Output video will be saved to: {output_path}")

            # Process each frame without displaying any window
            while True:
                output_frame = frame.copy()
                tracking_success = self.update_tracking(frame)
                if tracking_success and self.tracking_quality > 0.3 and self.warped_image is not None:
                    # Create mask from the current polygon
                    mask = np.zeros((self.height, self.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [self.current_polygon], 255)
                    mask_3ch = cv2.merge([mask, mask, mask])
                    np.copyto(output_frame, self.warped_image, where=mask_3ch.astype(bool))
                self.output_writer.write(output_frame)
                self.frame_count += 1
                ret, frame = self.cap.read()
                if not ret:
                    break

            print("Video processing completed.")
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            if self.output_writer:
                self.output_writer.release()

def main():
    video_path = VIDEO_PATH
    replacement_image_path = REPLACEMENT_PATH
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return
    if not os.path.exists(replacement_image_path):
        print(f"Error: {replacement_image_path} not found!")
        return
    mapper = AutoVideoPolygonMapper(video_path, replacement_image_path)
    mapper.process_video()

if __name__ == "__main__":
    main()
