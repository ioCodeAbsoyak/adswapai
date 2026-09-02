"""AdSwapAI R&D, 2025-04-20: per-frame billboard replacement using mask contour ->
quadrilateral -> getPerspectiveTransform (Detectron2 Mask R-CNN billboard model)."""

import os
import cv2
import argparse
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances

def setup_predictor(model_path, ann_file):
    register_coco_instances("billboard_train", {}, ann_file, "")
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TEST = ("billboard_train",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda"
    return DefaultPredictor(cfg)

def find_corners_from_mask(mask):
    """
    Extract the four corners of a quadrilateral from a binary mask using OpenCV
    """
    # Convert the mask to a binary image (required by findContours)
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon with fewer points
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If we don't get exactly 4 points, try to find the 4 corners
    if len(approx) != 4:
        # Find the convex hull
        hull = cv2.convexHull(contour)
        
        # If the hull has more than 4 points, approximate it further
        if len(hull) > 4:
            epsilon = 0.1 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            # If we still don't have 4 points, take the 4 extreme points
            if len(approx) != 4:
                # Get the bounding rectangle corners as a fallback
                rect = cv2.minAreaRect(contour)
                approx = cv2.boxPoints(rect)
                approx = np.array(approx).astype(np.int32)
        else:
            approx = hull
    
    # Ensure we have exactly 4 points
    if len(approx) != 4:
        # Get the bounding rectangle corners as a fallback
        x, y, w, h = cv2.boundingRect(contour)
        approx = np.array([
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]]
        ])
    
    # Convert to the right format
    corners = np.squeeze(approx).astype(np.float32)
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    # Calculate center point
    center = np.mean(corners, axis=0)
    
    # Calculate angles from center to each point
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    
    # Sort points by angle
    sorted_indices = np.argsort(angles)
    corners = corners[sorted_indices]
    
    # Reorder to start from top-left (smallest y, smallest x)
    # Find the point with smallest y (topmost)
    top_idx = np.argmin(corners[:, 1])
    corners = np.roll(corners, -top_idx, axis=0)
    
    return corners

def main(args):
    # 1) predictor
    predictor = setup_predictor(args.model_path, args.ann_file)
    
    # 2) Load all replacement images
    repl_dir = args.replace_dir
    repl_files = sorted([
        f for f in os.listdir(repl_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    repl_imgs = [cv2.imread(os.path.join(repl_dir, f)) for f in repl_files]
    n_repl = len(repl_imgs)
    
    if n_repl == 0:
        raise RuntimeError(f"No images found in {repl_dir}!")
    
    # 3) Video I/O
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {args.input_video}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        working_frame = frame.copy()
        
        # Run inference on the frame
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        
        # For each detected billboard
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # Get the replacement image (cycling through available ones)
            repl_img = repl_imgs[i % n_repl].copy()
            
            # Convert box to integers
            x1, y1, x2, y2 = box.astype(int)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract the mask for this billboard
            mask_roi = mask[y1:y2, x1:x2]
            
            try:
                # Find corners of the mask
                mask_corners = find_corners_from_mask(mask_roi)
                if mask_corners is None or len(mask_corners) != 4:
                    continue
                    
                # Adjust corners to global image coordinates
                mask_corners[:, 0] += x1
                mask_corners[:, 1] += y1
                
                # Create a mask for the warped region
                billboard_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(billboard_mask, [mask_corners.astype(np.int32)], 255)
                
                # Get the dimensions of the replacement image
                h_repl, w_repl = repl_img.shape[:2]
                
                # Define source points (corners of the replacement image)
                src_points = np.array([
                    [0, 0],
                    [w_repl - 1, 0],
                    [w_repl - 1, h_repl - 1],
                    [0, h_repl - 1]
                ], dtype=np.float32)
                
                # Calculate perspective transform matrix
                M = cv2.getPerspectiveTransform(src_points, mask_corners)
                
                # Create a warped version of the replacement image
                warped_repl = cv2.warpPerspective(repl_img, M, (width, height))
                
                # Apply the warped image to the frame using the billboard mask
                mask_bool = billboard_mask.astype(bool)
                for c in range(3):
                    working_frame[:, :, c] = np.where(mask_bool, warped_repl[:, :, c], working_frame[:, :, c])
                
            except Exception as e:
                print(f"Error processing billboard {i}: {e}")
                continue
        
        # Write the processed frame
        out.write(working_frame)
        
        # Print progress
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames...")
    
    # Clean up
    cap.release()
    out.release()
    print(f"Done! Output written to {args.output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace billboards in video with perspective correction")
    parser.add_argument("--model-path", default="model_final.pth", help="Path to Detectron2 model weights (custom billboard model, see docs/assets.md)")
    parser.add_argument("--ann-file", default="data/annotations.json", help="Path to COCO annotation file")
    parser.add_argument("--input-video", default="data/adVideo1.mp4", help="Path to input video (sample clip, see repo docs/assets.md)")
    parser.add_argument("--replace-dir", default="data/replace", help="Directory containing replacement images")
    parser.add_argument("--output-video", default="output/result.mp4", help="Path to output video")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_video) or ".", exist_ok=True)
    main(args)