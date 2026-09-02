"""AdSwapAI R&D, 2025-03-14: classical ad-board detection from the pitch edge (green mask, Sobel, Hough, strip analysis)."""

import cv2
import numpy as np
import os
import glob
import argparse
from matplotlib import pyplot as plt

class FootballAdboardDetector:
    """
    Specialized detector for LED advertising boards in football/soccer matches.
    """
    
    def __init__(self):
        """Initialize the detector"""
        pass
        
    def detect(self, image):
        """
        Detect LED advertising boards in a football match image.
        
        Args:
            image: OpenCV BGR image
            
        Returns:
            Tuple of (result image, mask, overlay image)
        """
        # Create a copy for results
        result = image.copy()
        height, width = image.shape[:2]
        
        # Convert to HSV color space for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract saturation channel - advertising boards typically have high saturation
        sat = hsv[:, :, 1]
        
        # Find the green playing field
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        field_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)
        
        # Find horizontal edges - focusing on transition between field and ads
        # First calculate vertical gradient to find horizontal edges
        sobelx = cv2.Sobel(field_mask, cv2.CV_64F, 0, 1, ksize=5)
        sobelx = np.absolute(sobelx)
        sobelx = np.uint8(255 * sobelx / np.max(sobelx))
        
        # Threshold to get strong edges
        _, strong_edges = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)
        
        # Find horizontal lines using Hough transform
        edges = cv2.Canny(strong_edges, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=100, 
                               minLineLength=width//4, 
                               maxLineGap=50)
        
        # Identify field boundary line
        field_edge_y = None
        if lines is not None:
            # Consider lines in the middle third of the image (typical ad board location)
            field_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is horizontal (small y difference)
                if abs(y2 - y1) < 10 and abs(x2 - x1) > width//4:
                    # Check if it's in the middle portion of the image
                    if y1 > height * 0.4 and y1 < height * 0.7:
                        field_lines.append((line[0], abs(x2 - x1)))  # Store with line length
            
            # Sort by line length (longest first)
            field_lines.sort(key=lambda x: x[1], reverse=True)
            
            if field_lines:
                x1, y1, x2, y2 = field_lines[0][0]
                field_edge_y = (y1 + y2) // 2
        
        # If no clear field edge found, estimate based on color transition
        if field_edge_y is None:
            # Analyze vertical color profile to find green-to-other transition
            green_profile = np.sum(field_mask, axis=1) / width
            
            # Look for significant drop in green
            for y in range(height // 2, height - 20):
                if green_profile[y] > 200 and green_profile[y+10] < 100:
                    field_edge_y = y
                    break
            
            # Fallback if still not found
            if field_edge_y is None:
                field_edge_y = int(height * 0.6)  # Default position
        
        # Define the ad board region
        # Ad boards are typically right above the field edge
        ad_height = int(height * 0.08)  # Typical ad board height (~8% of image)
        
        # Ad boards are above the field edge
        ad_y_start = max(0, field_edge_y - ad_height)
        ad_y_end = field_edge_y
        
        # Create mask for ad board region
        ad_board_mask = np.zeros((height, width), dtype=np.uint8)
        ad_board_mask[ad_y_start:ad_y_end, :] = 255
        
        # Refine the mask based on color characteristics
        roi = image[ad_y_start:ad_y_end, :]
        roi_hsv = hsv[ad_y_start:ad_y_end, :]
        
        # Analyze color variation in columns
        color_var = np.var(roi, axis=0).sum(axis=1)
        high_var_cols = color_var > np.percentile(color_var, 60)
        
        # Analyze saturation
        sat_mean = np.mean(roi_hsv[:,:,1], axis=0)
        high_sat_cols = sat_mean > np.percentile(sat_mean, 50)
        
        # Detect bright and colorful regions (common in ads)
        value_mean = np.mean(roi_hsv[:,:,2], axis=0)
        bright_cols = value_mean > np.percentile(value_mean, 50)
        
        # Combine criteria
        ad_cols = np.logical_and(np.logical_and(high_var_cols, high_sat_cols), bright_cols)
        
        # Update mask based on detected columns
        refined_mask = np.zeros((height, width), dtype=np.uint8)
        refined_mask[ad_y_start:ad_y_end, ad_cols] = 255
        
        # Apply morphological operations to connect nearby regions
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, 
                                      np.ones((5, 20), np.uint8))
        
        # Find contours in the refined mask
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        final_mask = np.zeros_like(refined_mask)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it has reasonable dimensions for an ad board
            if w > width * 0.05 and h > 10:  # Minimum width 5% of image width
                aspect_ratio = w / h
                
                # Ad boards typically have large aspect ratio (width >> height)
                if aspect_ratio > 3:
                    # Draw rectangle on result
                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add to final mask
                    cv2.drawContours(final_mask, [contour], 0, 255, -1)
        
        # If no specific segments found, use the whole strip
        if np.sum(final_mask) == 0:
            final_mask = ad_board_mask
            cv2.rectangle(result, (0, ad_y_start), (width, ad_y_end), (0, 255, 0), 2)
        
        # Create overlay visualization
        overlay = image.copy()
        red_mask = np.zeros_like(image)
        red_mask[final_mask > 0] = [0, 0, 255]  # Red in BGR
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(red_mask, alpha, overlay, 1-alpha, 0, overlay)
        
        # Draw field edge line for reference
        cv2.line(result, (0, field_edge_y), (width, field_edge_y), (0, 255, 255), 1)
        
        return result, final_mask, overlay
    
    def process_image(self, image_path, output_dir=None):
        """
        Process a single image to detect LED advertising boards
        
        Args:
            image_path: Path to the input image
            output_dir: Optional directory to save results
            
        Returns:
            Tuple of (result image, mask, overlay image)
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None, None, None
        
        # Detect advertising boards
        result, mask, overlay = self.detect(image)
        
        # Save results if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            
            # Save images
            cv2.imwrite(os.path.join(output_dir, f"{name}_detected{ext}"), result)
            cv2.imwrite(os.path.join(output_dir, f"{name}_mask.png"), mask)
            cv2.imwrite(os.path.join(output_dir, f"{name}_overlay{ext}"), overlay)
            
            # Create visualization
            self.create_visualization(image, result, mask, overlay, 
                                     os.path.join(output_dir, f"{name}_analysis.png"))
            
            print(f"Results saved to {output_dir}")
        
        return result, mask, overlay
    
    def create_visualization(self, original, result, mask, overlay, save_path=None):
        """Create a visualization of the detection process"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Detected Ad Boards')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('Ad Board Mask')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title('Ad Board Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect LED advertising boards in football matches')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', '-o', type=str, default='adboard_results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Initialize detector
    detector = FootballAdboardDetector()
    
    # Check if input is a directory or single file
    if os.path.isdir(args.input):
        # Process all images in directory
        image_files = glob.glob(os.path.join(args.input, '*.jpg')) + \
                     glob.glob(os.path.join(args.input, '*.png'))
        
        print(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            print(f"Processing: {os.path.basename(image_path)}")
            detector.process_image(image_path, args.output)
    else:
        # Process single image
        print(f"Processing: {os.path.basename(args.input)}")
        result, mask, overlay = detector.process_image(args.input, args.output)
        
        # Display results
        cv2.imshow('Detected Ad Boards', result)
        cv2.imshow('Ad Board Overlay', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()