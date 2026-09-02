"""AdSwapAI R&D, 2025-03-16: polygon tool that subtracts people using Mask R-CNN person masks (GrabCut/HOG fallbacks)."""

import cv2
import numpy as np
import os
import glob
import pickle
import time
import torch
import torchvision

# Folder with the still images to annotate (see repo docs/assets.md)
DATA_DIR = "data/ads"

class PreciseMaskSelector:
    """
    Advanced tool to select advertising boards using polygon selection.
    Features:
    - Multiple polygon selection for ad boards
    - Precise person segmentation (not just bounding boxes)
    - Save and load selections
    - Navigate between images
    """
    
    def __init__(self, image_dir):
        """Initialize with directory of images"""
        self.image_dir = image_dir
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        
        if not self.image_files:
            raise ValueError(f"No JPG images found in: {image_dir}")
        
        self.current_idx = 0
        self.window_name = "Precise Mask Selector"
        
        # Data for current image
        self.image = None
        self.original = None
        self.display_image = None
        
        # Polygon selection data
        self.points = []  # Current polygon points
        self.complete = False  # Is current polygon complete?
        self.selections = {}  # Store selections for each image
        
        # Create output directory for saving results
        self.output_dir = os.path.join(image_dir, "selections")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Try to load existing selections
        self.selection_file = os.path.join(self.output_dir, "adboard_selections.pkl")
        self.load_selections()
        
        # For precise person segmentation
        self.segmentation_model = self.initialize_segmentation_model()
        self.objects_to_subtract = ["person"]  # Can be expanded to include other objects
        
        # Instructions text
        self.instructions = [
            "Left click: Add point to polygon",
            "Right click: Complete current polygon",
            "Key 'c': Clear current polygon",
            "Key 's': Process and save selection",
            "Key 'n': Next image",
            "Key 'p': Previous image",
            "Key 'q': Quit and save"
        ]
    
    def initialize_segmentation_model(self):
        """Initialize precise segmentation model"""
        print("Initializing precise segmentation model...")
        
        # First try to load a semantic segmentation model
        try:
            # Check if we can use a pre-trained DeepLabV3+ model
            print("Trying to load DeepLabV3+ model...")
            model = cv2.dnn.readNetFromTensorflow('deeplabv3_xception_ade20k.pb')
            print("DeepLabV3+ model loaded")
            return {"model": model, "type": "deeplab"}
        except Exception as e:
            print(f"Couldn't load DeepLabV3+ model: {e}")
            
            # Try to use MaskRCNN if available
            try:
                print("Trying to load Mask R-CNN model...")
                # Check if torchvision is available
                import torchvision
                from torchvision.models.detection import maskrcnn_resnet50_fpn
                
                # Create a Mask R-CNN model
                model = maskrcnn_resnet50_fpn(pretrained=True)
                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()
                
                print("Mask R-CNN model loaded")
                return {"model": model, "type": "maskrcnn"}
            except Exception as e:
                print(f"Couldn't load Mask R-CNN: {e}")
        
        # If all else fails, use grabcut algorithm
        print("Using OpenCV GrabCut for person segmentation")
        return {"type": "grabcut"}
    
    def detect_persons_precise(self, image):
        """
        Detect persons in the image with precise segmentation (not just bounding boxes)
        Returns a mask with persons marked as white on black background
        """
        height, width = image.shape[:2]
        person_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Use different methods based on available models
        if self.segmentation_model["type"] == "deeplab":
            # Use DeepLabV3+ for semantic segmentation
            model = self.segmentation_model["model"]
            
            # Prepare input
            blob = cv2.dnn.blobFromImage(image, 1.0/127.5, (513, 513), (127.5, 127.5, 127.5))
            model.setInput(blob)
            
            # Run segmentation
            output = model.forward()
            output = output[0, 0, :, :]
            
            # Resize output to image size
            output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Create mask for person class (usually class 15 in ADE20K dataset)
            person_mask = np.zeros((height, width), dtype=np.uint8)
            person_mask[output == 15] = 255  # Person class
            
            # Visualize detection on display
            self.display_image[person_mask > 0] = (0, 0, 255)  # Red for persons
            
        elif self.segmentation_model["type"] == "maskrcnn":
            # Use Mask R-CNN for instance segmentation
            model = self.segmentation_model["model"]
            
            # Convert to tensor
            import torch
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
            img_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # Run detection
            with torch.no_grad():
                prediction = model([img_tensor])[0]
            
            # Process predictions
            for i in range(len(prediction["labels"])):
                label = prediction["labels"][i].item()
                score = prediction["scores"][i].item()
                
                # Check if it's a person with high confidence
                if label == 1 and score > 0.75:  # 1 is person in COCO dataset
                    # Get mask
                    mask = prediction["masks"][i, 0].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Resize to image size if needed
                    if mask.shape[:2] != (height, width):
                        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                    
                    # Add to person mask
                    person_mask = cv2.bitwise_or(person_mask, mask)
                    
                    # Visualize detection on display
                    overlay = self.display_image.copy()
                    overlay[mask > 0] = (0, 0, 255)  # Red for persons
                    cv2.addWeighted(overlay, 0.5, self.display_image, 0.5, 0, self.display_image)
        
        else:
            # Fallback to findContours + GrabCut
            # First detect persons using HOG
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect people
            boxes, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(16, 16), scale=1.05)
            
            # Use GrabCut for precise segmentation of each detection
            for (x, y, w, h) in boxes:
                # Create a rectangle slightly larger than the detection
                rect = (max(0, x-10), max(0, y-10), min(w+20, width-x), min(h+20, height-y))
                
                # Prepare mask for GrabCut
                gc_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Set rectangle area to probable foreground
                gc_mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = cv2.GC_PR_FGD
                
                # Set a smaller rectangle as definite foreground
                inner_margin = int(min(w, h) * 0.2)
                inner_rect = (x+inner_margin, y+inner_margin, w-2*inner_margin, h-2*inner_margin)
                gc_mask[inner_rect[1]:inner_rect[1]+inner_rect[3], inner_rect[0]:inner_rect[0]+inner_rect[2]] = cv2.GC_FGD
                
                # Set margins as background
                margin = 2
                gc_mask[:margin, :] = gc_mask[height-margin:, :] = gc_mask[:, :margin] = gc_mask[:, width-margin:] = cv2.GC_BGD
                
                # Prepare GrabCut arrays
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Run GrabCut
                try:
                    cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                    
                    # Create mask where foreground (GC_FGD) or probable foreground (GC_PR_FGD)
                    person_segment = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
                    
                    # Add to person mask
                    person_mask = cv2.bitwise_or(person_mask, person_segment)
                    
                    # Visualize on display image
                    cv2.rectangle(self.display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    overlay = self.display_image.copy()
                    overlay[person_segment > 0] = (0, 0, 255)  # Red for persons
                    cv2.addWeighted(overlay, 0.5, self.display_image, 0.5, 0, self.display_image)
                    
                except cv2.error as e:
                    print(f"Error in GrabCut: {e}")
                    # Fallback to simple rectangle
                    cv2.rectangle(person_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Apply morphological operations to improve mask
        kernel = np.ones((5, 5), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        
        return person_mask
    
    def load_selections(self):
        """Load saved selections if they exist"""
        if os.path.exists(self.selection_file):
            try:
                with open(self.selection_file, 'rb') as f:
                    self.selections = pickle.load(f)
                print(f"Loaded existing selections for {len(self.selections)} images")
            except Exception as e:
                print(f"Error loading selections: {e}")
    
    def save_selections(self):
        """Save selections to file"""
        try:
            with open(self.selection_file, 'wb') as f:
                pickle.dump(self.selections, f)
            print(f"Saved selections to {self.selection_file}")
        except Exception as e:
            print(f"Error saving selections: {e}")
    
    def run(self):
        """Main loop to display images and handle interactions"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Starting Precise Mask Selector")
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
                self.current_idx = (self.current_idx + 1) % len(self.image_files)
                self.load_current_image()
            
            elif key == ord('p'):  # Previous image
                self.current_idx = (self.current_idx - 1) % len(self.image_files)
                self.load_current_image()
            
            elif key == ord('c'):  # Clear current polygon
                self.points = []
                self.complete = False
                print("Cleared current polygon")
            
            elif key == ord('s'):  # Process and save selection
                if self.complete and len(self.points) >= 3:
                    self.process_and_save_selection()
                else:
                    print("No complete polygon to process. Right-click to complete the current polygon.")
            
            elif key == ord('q'):  # Quit
                if self.complete and len(self.points) >= 3:
                    self.process_and_save_selection()
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
            self.display_image = self.original.copy()
            
            # Show existing selections for this image
            if file_path in self.selections and self.selections[file_path]:
                selections = self.selections[file_path]
                print(f"  Found {len(selections)} existing selection(s)")
                
                # Apply existing masks if available
                if "processed_mask" in selections:
                    print("  Showing previously processed mask")
                    # Display existing mask
                    self.display_with_mask(selections["processed_mask"])
    
    def display_with_mask(self, mask):
        """Display existing mask on the image"""
        # Create overlay
        overlay = self.display_image.copy()
        
        # Fill areas where mask is non-zero with red
        overlay[mask > 0] = [0, 0, 255]  # BGR format
        
        # Apply overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, self.display_image, 1-alpha, 0, self.display_image)
    
    def update_display(self):
        """Update the display image with current polygons and selection in progress"""
        # Start with a fresh copy of the original
        self.display_image = self.original.copy()
        
        # Draw existing processed mask if available
        current_path = self.image_files[self.current_idx]
        if current_path in self.selections and "processed_mask" in self.selections[current_path]:
            mask = self.selections[current_path]["processed_mask"]
            self.display_with_mask(mask)
        
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
                print("Completed polygon. Press 's' to process and save.")
            else:
                print("Need at least 3 points to complete a polygon")
    
    def process_and_save_selection(self):
        """Process the completed polygon: create mask, subtract persons, and save"""
        current_path = self.image_files[self.current_idx]
        
        # Initialize selection data if needed
        if current_path not in self.selections:
            self.selections[current_path] = {}
        
        # Get image dimensions
        h, w = self.original.shape[:2]
        
        # Create polygon mask
        polygon_mask = np.zeros((h, w), dtype=np.uint8)
        if len(self.points) >= 3:
            pts = np.array(self.points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(polygon_mask, [pts], 255)
        
        # Detect and subtract persons from mask with precise contours
        person_mask = self.detect_persons_precise(self.original)
        
        # Subtract person mask from polygon mask
        processed_mask = cv2.subtract(polygon_mask, person_mask)
        
        # Store the polygon points and processed mask
        if "polygons" not in self.selections[current_path]:
            self.selections[current_path]["polygons"] = []
        
        self.selections[current_path]["polygons"].append(self.points)
        self.selections[current_path]["processed_mask"] = processed_mask
        
        print(f"Processed and saved polygon with {len(self.points)} points")
        
        # Save visualization
        self.save_visualization(processed_mask, polygon_mask, person_mask)
        
        # Reset polygon for next selection
        self.points = []
        self.complete = False
        
        # Save to disk
        self.save_selections()
    
    def save_visualization(self, processed_mask, original_mask, person_mask):
        """Save visualization of masks for debugging and verification"""
        current_path = self.image_files[self.current_idx]
        base_name = os.path.basename(current_path)
        name, ext = os.path.splitext(base_name)
        
        # Create visualization image
        vis_image = self.original.copy()
        
        # Create overlay for the processed mask
        overlay = vis_image.copy()
        overlay[processed_mask > 0] = [0, 0, 255]  # Red in BGR
        cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
        
        # Create visualization showing both polygon and person masks
        debug_vis = self.original.copy()
        # Green for original polygon
        debug_vis[original_mask > 0] = [0, 255, 0]
        # Blue for detected persons
        debug_vis[person_mask > 0] = [255, 0, 0]
        # Red for final mask (after subtraction)
        debug_vis[processed_mask > 0] = [0, 0, 255]
        
        # Save visualizations
        cv2.imwrite(os.path.join(self.output_dir, f"{name}_visualization{ext}"), vis_image)
        cv2.imwrite(os.path.join(self.output_dir, f"{name}_debug_vis{ext}"), debug_vis)
        cv2.imwrite(os.path.join(self.output_dir, f"{name}_original_mask.png"), original_mask)
        cv2.imwrite(os.path.join(self.output_dir, f"{name}_person_mask.png"), person_mask)
        cv2.imwrite(os.path.join(self.output_dir, f"{name}_processed_mask.png"), processed_mask)
        
        print(f"Saved visualization and masks to {self.output_dir}")

# Main function
def main():
    # Directory containing the football images
    image_dir = DATA_DIR

    # Create and run the polygon selector
    selector = PreciseMaskSelector(image_dir)
    selector.run()

if __name__ == "__main__":
    main()