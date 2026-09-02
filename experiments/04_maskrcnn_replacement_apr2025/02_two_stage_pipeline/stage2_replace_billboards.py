"""AdSwapAI R&D, 2025-04-22: two-stage pipeline, stage 2 - read the stage 1 JSON and
replay the billboard replacement per track id (a different ad per track)."""

import cv2
import argparse
import numpy as np
import os
import json
import time

def order_points(pts):
    """Order points in clockwise order starting from top-left"""
    pts = np.array(pts)
    
    # Calculate center
    center = np.mean(pts, axis=0)
    
    # Calculate angles from center to each point
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # Sort points by angle
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    
    # Ensure the first point is the top-left
    # Find the point with the smallest sum of coordinates
    min_sum_idx = np.argmin(np.sum(sorted_pts, axis=1))
    sorted_pts = np.roll(sorted_pts, -min_sum_idx, axis=0)
    
    return sorted_pts

def replace_billboard(frame, mask_points, corners, replacement_img):
    """Replace billboard in the frame with the replacement image"""
    # Create mask from points
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for y, x in mask_points:
        if 0 <= y < h and 0 <= x < w:
            mask[y, x] = 255
    
    # Ensure corners is numpy array
    corners = np.array(corners)
    
    # Order corners for proper perspective transform
    try:
        corners = order_points(corners).astype(np.float32)
    except:
        # If ordering fails, draw a simple mask
        if len(corners) == 4:
            cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
        return frame
    
    # Get dimensions of replacement image
    h_repl, w_repl = replacement_img.shape[:2]
    
    # Define source points (corners of the replacement image)
    src_points = np.array([
        [0, 0],
        [w_repl - 1, 0],
        [w_repl - 1, h_repl - 1],
        [0, h_repl - 1]
    ], dtype=np.float32)
    
    # Calculate perspective transform
    try:
        M = cv2.getPerspectiveTransform(src_points, corners)
    except:
        # If perspective transform fails, draw a simple mask
        if len(corners) == 4:
            cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
        return frame
    
    # Create a warped version of the replacement image
    warped = cv2.warpPerspective(replacement_img, M, (w, h))
    
    # Create a mask from the warped image
    warp_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Ensure corners are valid for fillPoly
    if np.any(np.isnan(corners)) or np.any(np.isinf(corners)):
        # Use the original mask instead
        warp_mask = mask
    else:
        cv2.fillPoly(warp_mask, [corners.astype(np.int32)], 255)
    
    # Merge masks to ensure complete coverage
    final_mask = cv2.bitwise_or(mask, warp_mask)
    final_mask_bool = final_mask.astype(bool)
    
    # Combine the original and replacement images
    result = frame.copy()
    for c in range(3):
        result[:, :, c] = np.where(final_mask_bool, warped[:, :, c], result[:, :, c])
    
    return result

def main(args):
    # Load detection data
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    video_info = data['video_info']
    detections = data['detections']
    
    # Convert frame indices to strings since JSON keys are strings
    detections = {int(k): v for k, v in detections.items()}
    
    # Load video
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {args.input_video}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load replacement image
    replacement_img = cv2.imread(args.replace_img)
    if replacement_img is None:
        raise ValueError(f"Could not load replacement image: {args.replace_img}")
    
    # Map of billboard IDs to replacement images (we could use different images per ID in the future)
    billboard_images = {}
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_video) or '.', exist_ok=True)
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    # For FPS calculation
    frame_count = 0
    start_time = time.time()
    
    # Dictionary to store billboard info across frames (for time-consistency)
    billboard_info = {}
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        result_frame = frame.copy()
        
        # Get detections for this frame
        frame_detections = detections.get(frame_count, [])
        
        # Process each detection
        for det in frame_detections:
            billboard_id = det['id']
            mask_points = det['mask_points']
            corners = det['corners']
            
            # Assign a replacement image if not already assigned
            if billboard_id not in billboard_images:
                billboard_images[billboard_id] = replacement_img
            
            # Get the replacement image for this billboard
            repl_img = billboard_images[billboard_id]
            
            # Update billboard info for time-consistency
            billboard_info[billboard_id] = {
                'last_frame': frame_count,
                'mask_points': mask_points,
                'corners': corners
            }
            
            # Apply replacement
            try:
                result_frame = replace_billboard(
                    result_frame,
                    mask_points,
                    corners,
                    repl_img
                )
            except Exception as e:
                print(f"Error replacing billboard {billboard_id} in frame {frame_count}: {e}")
        
        # Write the frame
        out.write(result_frame)
        
        # Calculate and print FPS
        if frame_count % 20 == 0:
            elapsed_time = time.time() - start_time
            fps_calc = frame_count / elapsed_time
            print(f"Processed {frame_count}/{total_frames} frames... FPS: {fps_calc:.2f}")
    
    # Clean up
    cap.release()
    out.release()
    print(f"Done! Output written to {args.output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace billboards using detection data")
    parser.add_argument("--input-json", default="output/detections.json", help="Path to detection JSON file")
    parser.add_argument("--input-video", default="data/adVideo1.mp4", help="Path to input video (sample clip, see repo docs/assets.md)")
    parser.add_argument("--replace-img", default="data/replace.jpg", help="Path to replacement image")
    parser.add_argument("--output-video", default="output/result.mp4", help="Path to output video")

    args = parser.parse_args()
    main(args)