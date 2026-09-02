"""AdSwapAI R&D, 2025-04-03: several user-drawn polygons, one CSRT tracker each,
with a HOG people mask excluding pedestrians from the overlay."""

import cv2
import numpy as np
import os
from datetime import datetime

VIDEO_PATH = "data/adVideo2.mp4"   # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"

# This class represents a single ad polygon
class AdPolygon:
    def __init__(self, points, frame, tracker_color):
        """
        points: original corner points selected by the user (numpy array, int32, shape=(N,2))
        frame: the first frame the polygon was selected on (used to init the CSRT tracker)
        tracker_color: color used to draw this polygon (BGR tuple)
        """
        self.original_polygon = np.array(points, dtype=np.int32)  # original coordinates
        self.current_polygon = self.original_polygon.copy()       # updated as tracking progresses
        self.tracker_color = tracker_color
        # Compute the bounding box and relative coordinates:
        self.bbox = cv2.boundingRect(self.original_polygon)  # (x,y,w,h)
        self.relative_polygon = []
        for pt in self.original_polygon:
            rel_x = (pt[0] - self.bbox[0]) / self.bbox[2]
            rel_y = (pt[1] - self.bbox[1]) / self.bbox[3]
            self.relative_polygon.append([rel_x, rel_y])
        self.relative_polygon = np.array(self.relative_polygon, dtype=np.float32)
        # No tracker yet at construction time; the CSRT tracker is created later.
        self.tracker = None
        # Warped replacement image stored here for the overlay.
        self.warped_image = None

    def init_tracker(self, frame):
        """Initialize the CSRT tracker using the original bounding box."""
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError as e:
            raise Exception("CSRT tracker not available. Update opencv-contrib-python.") from e
        self.tracker.init(frame, self.bbox)

    def update_tracker(self, frame):
        """Update the CSRT tracker and recompute current_polygon from the bounding box using the relative coordinates."""
        success, bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            new_poly = []
            for rel in self.relative_polygon:
                new_x = int(x + rel[0] * w)
                new_y = int(y + rel[1] * h)
                new_poly.append([new_x, new_y])
            self.current_polygon = np.array(new_poly, dtype=np.int32)
        return success

class MultiPolygonVideoMapper:
    def __init__(self, video_path, replacement_image_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"Video properties: {self.width}x{self.height}, {self.fps} FPS")
        
        self.replacement_image = cv2.imread(replacement_image_path)
        if self.replacement_image is None:
            raise ValueError(f"Could not load replacement image: {replacement_image_path}")
        print(f"Replacement image loaded: {self.replacement_image.shape}")
        # Convert to RGBA for alpha blending
        if len(self.replacement_image.shape) == 3:
            self.replacement_image = cv2.cvtColor(self.replacement_image, cv2.COLOR_BGR2BGRA)

        self.oversampling = 3.0
        # For user polygon selection
        self.current_polygon_points = []  # temporary list of selected points
        self.polygons = []                # saved AdPolygon objects
        self.render_mode = False          # activated by pressing 'r'
        
        self.window_name = "Multi-Polygon Selector (Left click: add point, Right click: save polygon, r: render, q: quit)"
        self.output_writer = None
        self.frame_count = 0

    def mouse_callback(self, event, x, y, flags, param):
        if self.render_mode:
            return  # no new selection while in render mode
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon_points.append((x, y))
            print(f"Point added: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_polygon_points) >= 3:
                # Polygon completed; pick a color, e.g. first polygon red, second green, third blue...
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0,255,255)]
                color = colors[len(self.polygons) % len(colors)]
                poly = AdPolygon(self.current_polygon_points, self.current_frame, color)
                poly.init_tracker(self.current_frame)
                self.polygons.append(poly)
                print(f"Polygon {len(self.polygons)} saved - coordinates: {self.current_polygon_points}")
                # Clear the temporary point list
                self.current_polygon_points = []

    def detect_people_mask(self, frame):
        """
        Detect people in the frame using HOG detector.
        Returns a binary mask with detected people areas.
        """
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for (x, y, w, h) in rects:
            pad = 5
            cv2.rectangle(mask, (x - pad, y - pad), (x + w + pad, y + h + pad), 255, -1)
        return mask

    def skip_frames(self, num_frames=10):
        """Skip a specified number of frames by reading and discarding them."""
        frame = None
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                return None
            self.frame_count += 1
        return frame

    def update_warped_image(self, poly):
        """
        Update warped overlay for a given AdPolygon.
        This version uses max scaling so that the ad covers the entire polygon.
        """
        if poly.current_polygon is None or len(poly.current_polygon) < 4:
            return
        try:
            x, y, w, h = cv2.boundingRect(poly.current_polygon)
            w = max(1, w)
            h = max(1, h)
            img_h, img_w = self.replacement_image.shape[:2]
            scale = max(w / img_w, h / img_h)
            target_w = int(img_w * scale * self.oversampling)
            target_h = int(img_h * scale * self.oversampling)
            resized_img = cv2.resize(self.replacement_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            dst_points = np.float32(poly.current_polygon[:4])
            src_points = np.float32([[0, 0],
                                     [target_w - 1, 0],
                                     [target_w - 1, target_h - 1],
                                     [0, target_h - 1]])
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(resized_img, M, (self.width, self.height),
                                         flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT)
            if warped.shape[2] == 4:
                poly.warped_image = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)
            else:
                poly.warped_image = warped
        except Exception as e:
            print(f"Error updating warped image for polygon: {str(e)}")
            poly.warped_image = None

    def process_video(self, output_dir):
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
        print("Draw your polygons on the first frame. Left-click to add points, right-click to save a polygon.")
        print("When done, press 'r' to start rendering.")
        
        # First stage: let the user select polygons
        self.current_frame = frame.copy()  # selection happens on this frame
        while True:
            display_frame = self.current_frame.copy()
            # Draw the selected points (temporary)
            if len(self.current_polygon_points) > 0:
                for pt in self.current_polygon_points:
                    cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
                for i in range(len(self.current_polygon_points)-1):
                    cv2.line(display_frame, self.current_polygon_points[i], self.current_polygon_points[i+1], (0,255,0), 2)
            # Draw the saved polygons in their respective colors
            for poly in self.polygons:
                pts = poly.original_polygon.reshape((-1,1,2))
                cv2.polylines(display_frame, [pts], True, poly.tracker_color, 2)
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.render_mode = True
                break
            elif key == ord('q'):
                return
        
        print("Render mode activated. Starting video processing...")
        # Now in render mode; enter the video processing loop
        while True:
            try:
                display_frame = frame.copy()
                # First, get the detected people mask
                people_mask = self.detect_people_mask(frame)
                # For each polygon
                for poly in self.polygons:
                    tracking_success = poly.update_tracker(frame)
                    if tracking_success:
                        self.update_warped_image(poly)
                        if poly.warped_image is not None:
                            # Build the polygon mask
                            poly_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                            cv2.fillPoly(poly_mask, [poly.current_polygon], 255)
                            # Invert the people mask and combine it with the polygon mask
                            inv_people_mask = cv2.bitwise_not(people_mask)
                            final_mask = cv2.bitwise_and(poly_mask, inv_people_mask)
                            mask_3ch = cv2.merge([final_mask, final_mask, final_mask])
                            np.copyto(display_frame, poly.warped_image, where=mask_3ch.astype(bool))
                        # Also draw the current polygon's outline (e.g. red)
                        cv2.polylines(display_frame, [poly.current_polygon.reshape((-1,1,2))], True, (0,0,255), 2)
                    else:
                        # If tracking failed, mark the polygon outline in yellow
                        cv2.polylines(display_frame, [poly.current_polygon.reshape((-1,1,2))], True, (0,255,255), 2)
                
                cv2.imshow(self.window_name, display_frame)
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
                self.frame_count += 1
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                break
        self.cap.release()
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
        mapper = MultiPolygonVideoMapper(video_path, replacement_image_path)
        mapper.process_video(output_directory)
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
