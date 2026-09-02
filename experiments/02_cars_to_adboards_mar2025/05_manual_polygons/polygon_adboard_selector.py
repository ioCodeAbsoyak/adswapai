"""AdSwapAI R&D, 2025-03-16: manual polygon annotation tool for ad boards."""

import cv2
import numpy as np
import os
import glob
import pickle

# Folder with the still images to annotate (see repo docs/assets.md)
DATA_DIR = "data/ads"

class PolygonAdboardSelector:
    """
    Tool to select advertising boards using polygon selection.
    User can click multiple times to create a custom shape around the ad boards.
    """
    
    def __init__(self, image_dir):
        """Initialize with directory of images"""
        self.image_dir = image_dir
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        
        if not self.image_files:
            raise ValueError(f"No JPG images found in: {image_dir}")
        
        self.current_idx = 0
        self.window_name = "Polygon Ad Board Selector"
        
        # Data for current image
        self.image = None
        self.original = None
        self.display_image = None
        
        # Polygon selection data
        self.points = []  # Current polygon points
        self.complete = False  # Is current polygon complete?
        self.polygons = {}  # Store polygons for each image: {image_path: [polygon1, polygon2, ...]}
        
        # Create output directory for saving results
        self.output_dir = os.path.join(image_dir, "selections")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Try to load existing selections
        self.selection_file = os.path.join(self.output_dir, "adboard_selections.pkl")
        self.load_selections()
        
        # Instructions text
        self.instructions = [
            "Left click: Add point to polygon",
            "Right click: Complete current polygon",
            "Key 'c': Clear current polygon",
            "Key 'n': Next image",
            "Key 'p': Previous image",
            "Key 's': Save selections",
            "Key 'q': Quit and save"
        ]
    
    def load_selections(self):
        """Load saved selections if they exist"""
        if os.path.exists(self.selection_file):
            try:
                with open(self.selection_file, 'rb') as f:
                    self.polygons = pickle.load(f)
                print(f"Loaded existing selections for {len(self.polygons)} images")
            except Exception as e:
                print(f"Error loading selections: {e}")
    
    def save_selections(self):
        """Save selections to file"""
        try:
            with open(self.selection_file, 'wb') as f:
                pickle.dump(self.polygons, f)
            print(f"Saved selections to {self.selection_file}")
        except Exception as e:
            print(f"Error saving selections: {e}")
    
    def run(self):
        """Main loop to display images and handle interactions"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Starting Polygon Ad Board Selector")
        print("\nInstructions:")
        for instruction in self.instructions:
            print(f"  {instruction}")
        
        self.load_current_image()
        
        while True:
            # Create display image with current selections
            self.update_display()
            
            # Show the image
            cv2.imshow(self.window_name, self.display_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):  # Next image
                self.save_current_polygon()
                self.current_idx = (self.current_idx + 1) % len(self.image_files)
                self.load_current_image()
            
            elif key == ord('p'):  # Previous image
                self.save_current_polygon()
                self.current_idx = (self.current_idx - 1) % len(self.image_files)
                self.load_current_image()
            
            elif key == ord('c'):  # Clear current polygon
                self.points = []
                self.complete = False
                print("Cleared current polygon")
            
            elif key == ord('s'):  # Save selections
                self.save_current_polygon()
                self.save_selections()
            
            elif key == ord('q'):  # Quit
                self.save_current_polygon()
                self.save_selections()
                break
        
        cv2.destroyAllWindows()
    
    def load_current_image(self):
        """Load the current image and its selections"""
        if 0 <= self.current_idx < len(self.image_files):
            # Reset polygon state
            self.points = []
            self.complete = False
            
            # Load the image
            file_path = self.image_files[self.current_idx]
            file_name = os.path.basename(file_path)
            
            print(f"\nImage {self.current_idx + 1}/{len(self.image_files)}: {file_name}")
            
            self.original = cv2.imread(file_path)
            if self.original is None:
                print(f"Error: Could not read image {file_path}")
                return
            
            self.image = self.original.copy()
            
            # Show existing selections for this image
            if file_path in self.polygons and self.polygons[file_path]:
                print(f"  Found {len(self.polygons[file_path])} existing selection(s)")
    
    def update_display(self):
        """Update the display image with current polygons and selection in progress"""
        # Start with a fresh copy of the original
        self.display_image = self.original.copy()
        
        # Draw existing polygons for this image
        current_path = self.image_files[self.current_idx]
        if current_path in self.polygons:
            for i, polygon in enumerate(self.polygons[current_path]):
                if len(polygon) > 2:
                    # Convert points to numpy array for drawing
                    pts = np.array(polygon, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    
                    # Draw filled polygon with transparency
                    overlay = self.display_image.copy()
                    cv2.fillPoly(overlay, [pts], (0, 0, 255))  # Red fill
                    cv2.addWeighted(overlay, 0.5, self.display_image, 0.5, 0, self.display_image)
                    
                    # Draw polygon outline
                    cv2.polylines(self.display_image, [pts], True, (0, 255, 0), 2)
                    
                    # Label the polygon
                    M = cv2.moments(pts)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(self.display_image, f"Ad {i+1}", (cx, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw current polygon in progress
        if self.points:
            # Draw the points
            for i, point in enumerate(self.points):
                cv2.circle(self.display_image, point, 5, (0, 255, 255), -1)
                
                # Draw lines connecting the points
                if i > 0:
                    cv2.line(self.display_image, self.points[i-1], point, (0, 255, 255), 2)
            
            # If polygon is complete, connect last point to first
            if self.complete and len(self.points) > 1:
                cv2.line(self.display_image, self.points[-1], self.points[0], (0, 255, 255), 2)
        
        # Add instructions and image number
        self.add_text_overlay()
    
    def add_text_overlay(self):
        """Add instruction text and image counter to the display"""
        # Add image counter
        cv2.putText(self.display_image, f"Image {self.current_idx + 1}/{len(self.image_files)}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add instructions as a semi-transparent overlay at the bottom
        h, w = self.display_image.shape[:2]
        overlay = self.display_image.copy()
        
        # Black background for text
        cv2.rectangle(overlay, (0, h - 30 * len(self.instructions)), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, self.display_image, 0.4, 0, self.display_image)
        
        # Add instruction text
        for i, instruction in enumerate(self.instructions):
            y = h - 30 * (len(self.instructions) - i)
            cv2.putText(self.display_image, instruction, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: Add point to polygon
            if not self.complete:
                self.points.append((x, y))
                print(f"Added point {len(self.points)} at ({x}, {y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: Complete polygon if we have at least 3 points
            if len(self.points) >= 3:
                self.complete = True
                print("Completed polygon")
                self.save_current_polygon()
                
                # Start a new polygon
                self.points = []
                self.complete = False
            else:
                print("Need at least 3 points to complete a polygon")
    
    def save_current_polygon(self):
        """Save the current polygon if it's complete"""
        if self.complete and len(self.points) >= 3:
            current_path = self.image_files[self.current_idx]
            
            if current_path not in self.polygons:
                self.polygons[current_path] = []
            
            self.polygons[current_path].append(self.points)
            print(f"Saved polygon with {len(self.points)} points")
            
            # Reset polygon
            self.points = []
            self.complete = False
            
            # Save to disk
            self.save_selections()
            
            # Save visualization
            self.save_visualization()
    
    def save_visualization(self):
        """Save a visualization of the current image with all polygons"""
        current_path = self.image_files[self.current_idx]
        if current_path in self.polygons and self.polygons[current_path]:
            # Create visualization
            vis_image = self.original.copy()
            
            for polygon in self.polygons[current_path]:
                if len(polygon) >= 3:
                    pts = np.array(polygon, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    
                    # Draw filled polygon with transparency
                    overlay = vis_image.copy()
                    cv2.fillPoly(overlay, [pts], (0, 0, 255))  # Red fill
                    cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
                    
                    # Draw polygon outline
                    cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
            
            # Save the visualization
            base_name = os.path.basename(current_path)
            name, ext = os.path.splitext(base_name)
            vis_path = os.path.join(self.output_dir, f"{name}_selection{ext}")
            
            cv2.imwrite(vis_path, vis_image)
            print(f"Saved visualization to {vis_path}")
            
            # Also save a mask
            h, w = self.original.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            for polygon in self.polygons[current_path]:
                if len(polygon) >= 3:
                    pts = np.array(polygon, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
            
            mask_path = os.path.join(self.output_dir, f"{name}_mask.png")
            cv2.imwrite(mask_path, mask)
            print(f"Saved mask to {mask_path}")

# Main function
def main():
    # Directory containing the football images
    image_dir = DATA_DIR

    # Create and run the polygon selector
    selector = PolygonAdboardSelector(image_dir)
    selector.run()

if __name__ == "__main__":
    main()