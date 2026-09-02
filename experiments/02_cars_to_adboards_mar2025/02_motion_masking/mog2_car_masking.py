"""AdSwapAI R&D, 2025-03-03: classical MOG2 background subtraction, contour filtering, centroid tracker and per-car mask overlay video."""

import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

# put the input files next to this script or pass a path
DEFAULT_VIDEO = "data/Road.mp4"

def main():
    # Check if we're using headless OpenCV
    headless = True
    try:
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Test")
        headless = False
    except:
        print("Using headless OpenCV - GUI functions disabled")
    
    # Load video
    video_path = DEFAULT_VIDEO
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=False)
    
    # FPS tracking
    fps_start_time = 0
    fps_counter = 0
    fps = 0
    
    # Create random colors for visualization
    colors = np.random.randint(0, 255, (100, 3))
    
    # Check if CUDA is available
    use_gpu = False
    try:
        device_count = cuda.Device.count()
        if device_count > 0:
            cuda_device = cuda.Device(0)
            device_name = cuda_device.name()
            print(f"Using GPU: {device_name}")
            use_gpu = True
        else:
            print("No CUDA devices found")
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
    
    # Kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    
    # Create a simple tracker
    class SimpleTracker:
        def __init__(self, max_disappeared=10):
            self.next_object_id = 0
            self.objects = {}  # format: {object_id: (centroid, disappeared_count)}
            self.max_disappeared = max_disappeared
        
        def register(self, centroid):
            self.objects[self.next_object_id] = (centroid, 0)
            self.next_object_id += 1
        
        def deregister(self, object_id):
            del self.objects[object_id]
        
        def update(self, centroids):
            # Update tracked objects with new centroids
            
            # If no centroids, mark all objects as disappeared
            if len(centroids) == 0:
                for object_id in list(self.objects.keys()):
                    centroid, disappeared = self.objects[object_id]
                    self.objects[object_id] = (centroid, disappeared + 1)
                    
                    # Deregister if disappeared for too long
                    if disappeared > self.max_disappeared:
                        self.deregister(object_id)
                
                return self.objects
            
            # If no existing objects, register all centroids
            if len(self.objects) == 0:
                for centroid in centroids:
                    self.register(centroid)
            else:
                # Match existing objects to new centroids
                object_ids = list(self.objects.keys())
                object_centroids = [self.objects[id][0] for id in object_ids]
                
                # Calculate distances
                distances = np.zeros((len(object_centroids), len(centroids)))
                for i, object_centroid in enumerate(object_centroids):
                    for j, centroid in enumerate(centroids):
                        distances[i, j] = np.sqrt(
                            (object_centroid[0] - centroid[0])**2 + 
                            (object_centroid[1] - centroid[1])**2
                        )
                
                # Find the closest centroids for each object
                used_rows = set()
                used_cols = set()
                
                # Sort by closest distances
                for i in range(distances.shape[0]):
                    if i >= len(distances):
                        break
                        
                    min_dist_idx = np.argmin(distances[i])
                    min_dist = distances[i][min_dist_idx]
                    
                    # Only match if distance is reasonable (< 100 pixels)
                    if min_dist < 100 and min_dist_idx not in used_cols:
                        # Update position and reset disappeared counter
                        object_id = object_ids[i]
                        self.objects[object_id] = (centroids[min_dist_idx], 0)
                        used_rows.add(i)
                        used_cols.add(min_dist_idx)
                
                # Mark objects without matches as disappeared
                for i, object_id in enumerate(object_ids):
                    if i not in used_rows:
                        centroid, disappeared = self.objects[object_id]
                        self.objects[object_id] = (centroid, disappeared + 1)
                        
                        # Deregister if disappeared for too long
                        if disappeared > self.max_disappeared:
                            self.deregister(object_id)
                
                # Register new centroids
                for j, centroid in enumerate(centroids):
                    if j not in used_cols:
                        self.register(centroid)
            
            return self.objects
    
    # Create tracker
    tracker = SimpleTracker(max_disappeared=15)
    
    # Object detection parameters
    min_contour_area = 1000
    min_width, min_height = 50, 50
    max_width, max_height = 400, 300
    
    # Video writer setup
    output_path = "car_masking_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (frame_width, frame_height))
    
    print(f"Processing video: {video_path}")
    print(f"Output video will be saved to: {output_path}")
    
    # Storage for performance metrics
    processing_times = []
    num_frames = 0
    total_cars_detected = 0
    
    # Main processing loop
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start processing timer
        process_start = time.time()
        
        # Increment frame counter
        num_frames += 1
        
        # FPS calculation
        if fps_start_time == 0:
            fps_start_time = time.time()
        
        fps_counter += 1
        fps_end_time = time.time()
        fps_time = fps_end_time - fps_start_time
        
        # Update FPS every second
        if fps_time > 1:
            fps = fps_counter / fps_time
            fps_counter = 0
            fps_start_time = fps_end_time
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Remove noise with morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process detected contours
        valid_contours = []
        centroids = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by size
            if area > min_contour_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Apply filters to reduce false positives
                
                # 1. Size filter
                if w < min_width or h < min_height or w > max_width or h > max_height:
                    continue
                
                # 2. Aspect ratio filter (car shape)
                aspect_ratio = float(w) / h
                if not (0.5 <= aspect_ratio <= 2.5):
                    continue
                
                # 3. Density filter (how filled the contour is)
                rect_area = w * h
                extent = float(area) / rect_area
                if extent < 0.4:  # Contour fills less than 40% of its bounding rectangle
                    continue
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                centroids.append((cx, cy))
                valid_contours.append((x, y, w, h, contour))
        
        # Update tracker
        objects = tracker.update(centroids)
        
        # Create mask image for visualization
        mask_image = np.zeros_like(frame)
        
        # Draw tracking results
        for object_id, ((cx, cy), disappeared) in objects.items():
            # Skip objects that have disappeared
            if disappeared > 0:
                continue
                
            # Find the corresponding contour
            distances = [np.sqrt((cx - (x + w//2))**2 + (cy - (y + h//2))**2) 
                        for x, y, w, h, _ in valid_contours]
            
            if len(distances) > 0:
                closest_idx = np.argmin(distances)
                x, y, w, h, contour = valid_contours[closest_idx]
                
                # Get a stable color based on object ID
                color = colors[object_id % len(colors)].tolist()
                
                # Create mask for the contour
                car_mask = np.zeros_like(fg_mask)
                cv2.drawContours(car_mask, [contour], -1, 255, -1)
                
                # Color the mask area
                mask_image[car_mask == 255] = color
                
                # Draw bounding box and ID on display frame
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, f"Car {object_id}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(display_frame, (cx, cy), 4, color, -1)
        
        # Blend mask with original frame for better visualization
        alpha = 0.4
        mask_blend = cv2.addWeighted(display_frame, 1, mask_image, alpha, 0)
        
        # Calculate processing time
        process_time = (time.time() - process_start) * 1000  # ms
        processing_times.append(process_time)
        
        # Update car detection count
        car_count = len(objects)
        total_cars_detected += car_count
        
        # Add statistics overlay
        cv2.putText(mask_blend, f"FPS: {fps:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(mask_blend, f"Cars: {car_count}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(mask_blend, f"Process: {process_time:.1f} ms", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show GPU status
        gpu_status = "GPU: Enabled" if use_gpu else "GPU: Disabled"
        cv2.putText(mask_blend, gpu_status, (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if use_gpu else (255, 0, 0), 2)
        
        # Show Blackwell status if applicable
        if use_gpu:
            cv2.putText(mask_blend, "Blackwell", (frame_width - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame if not in headless mode
        if not headless:
            try:
                cv2.imshow("Car Detection", mask_blend)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception as e:
                print(f"GUI error (expected in headless mode): {e}")
        
        # Write frame to output video
        out.write(mask_blend)
        
        # Print progress every 100 frames
        if num_frames % 100 == 0:
            print(f"Processed {num_frames} frames, current FPS: {fps:.1f}, cars detected: {car_count}")
    
    # Clean up
    cap.release()
    out.release()
    
    if not headless:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    # Print summary statistics
    avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_fps = 1000 / avg_process_time if avg_process_time > 0 else 0
    
    print("\n===== Processing Complete =====")
    print(f"Total frames processed: {num_frames}")
    print(f"Average processing time: {avg_process_time:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total cars detected: {total_cars_detected}")
    print(f"Average cars per frame: {total_cars_detected/num_frames:.2f}")
    print(f"Output video saved to: {output_path}")

if __name__ == "__main__":
    main()