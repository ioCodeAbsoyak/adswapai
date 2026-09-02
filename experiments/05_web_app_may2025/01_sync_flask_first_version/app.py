#!/usr/bin/env python3
"""AdSwapAI R&D, 2025-05-08: first synchronous Flask endpoint (Detectron2 billboard + COCO human models, mask/image replacement, blocking /process)."""
import os
import sys
import time
import tempfile
import logging
import cv2
import numpy as np
import torch
import subprocess
from datetime import datetime
from flask import Flask, request, send_file, abort, jsonify
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('BillboardService')

logger = setup_logging()

# Model setup
BILLBOARD_MODEL_PATH = 'model_final.pth'
CONF_THRESH = 0.5
HUMAN_CONF_THRESH = 0.5

def setup_billboard_predictor(conf_threshold=CONF_THRESH):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = os.path.abspath(BILLBOARD_MODEL_PATH)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = DefaultPredictor(cfg)
    logger.info(f"Loaded billboard predictor on {cfg.MODEL.DEVICE} with confidence threshold {conf_threshold}")
    return predictor

def setup_human_predictor(conf_threshold=HUMAN_CONF_THRESH):
    """Setup a predictor for detecting humans and sports balls using COCO pretrained model"""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    # Use original 80 COCO classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = DefaultPredictor(cfg)
    logger.info(f"Loaded human detection predictor on {cfg.MODEL.DEVICE} with confidence threshold {conf_threshold}")
    return predictor

# Initialize predictors with default thresholds
billboard_predictor_default = setup_billboard_predictor()
human_predictor_default = setup_human_predictor()

def calculate_mask_overlap(mask1, mask2):
    """Calculate overlap percentage between two masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    
    # Calculate overlap relative to the smaller mask
    size1 = mask1.sum()
    size2 = mask2.sum()
    smaller_size = min(size1, size2)
    
    overlap_percent = (intersection / smaller_size) * 100
    return overlap_percent

# Helper for mask replacement
def replace_using_mask(frame: np.ndarray, mask: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return frame
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h, w = y2 - y1 + 1, x2 - x1 + 1
    rep_resized = cv2.resize(replacement, (w, h), interpolation=cv2.INTER_AREA)
    result = frame.copy()
    mask_crop = mask[y1:y2+1, x1:x2+1]
    region = result[y1:y2+1, x1:x2+1]
    for c in range(3):
        ch = region[:, :, c]
        ch[mask_crop] = rep_resized[:, :, c][mask_crop]
        region[:, :, c] = ch
    result[y1:y2+1, x1:x2+1] = region
    return result

# Flask app setup
app = Flask(__name__)

# Add CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

@app.route('/videos/<filename>', methods=['GET'])
def serve_video(filename):
    """Serve a processed video file."""
    video_dir = os.path.join(os.getcwd(), 'processed_videos')
    return send_file(
        os.path.join(video_dir, filename),
        mimetype='video/mp4'
    )

@app.route('/process', methods=['POST'])
def process():
    # Validate video file
    if 'video' not in request.files:
        abort(400, 'Missing video file')
    video_file = request.files['video']
    mode = request.form.get('mode', 'mask')
    
    # Get confidence threshold for billboard detection
    try:
        conf_threshold = float(request.form.get('conf_threshold', '0.5'))
        # Ensure value is in valid range
        conf_threshold = max(0.1, min(0.9, conf_threshold))
    except ValueError:
        conf_threshold = 0.5
    
    logger.info(f"Using billboard confidence threshold: {conf_threshold}")
    
    # Get confidence threshold for human detection
    try:
        human_conf_threshold = float(request.form.get('human_conf_threshold', '0.5'))
        # Ensure value is in valid range
        human_conf_threshold = max(0.1, min(0.9, human_conf_threshold))
    except ValueError:
        human_conf_threshold = 0.5
    
    logger.info(f"Using human confidence threshold: {human_conf_threshold}")
    
    # Get maximum allowed overlap percentage
    try:
        max_overlap_percent = float(request.form.get('max_overlap_percent', '20'))
        # Ensure reasonable range
        max_overlap_percent = max(0, min(100, max_overlap_percent))
    except ValueError:
        max_overlap_percent = 20
    
    logger.info(f"Using maximum overlap percentage: {max_overlap_percent}%")
    
    # Check if human filtering is enabled
    enable_human_filter = request.form.get('enable_human_filter', 'false').lower() in ['true', '1', 'yes', 'on']
    logger.info(f"Human filtering enabled: {enable_human_filter}")
    
    # Get minimum mask size (as percentage of frame)
    try:
        min_mask_size = float(request.form.get('min_mask_size', '0'))
        # Ensure non-negative
        min_mask_size = max(0, min_mask_size)
    except ValueError:
        min_mask_size = 0
    
    logger.info(f"Using minimum mask size ratio: {min_mask_size}")
    
    # Get mask parameters if in mask mode
    mask_color = (0, 255, 0)  # Default green (BGR)
    mask_alpha = 0.5  # Default opacity
    
    if mode == 'mask':
        try:
            mask_color_r = int(request.form.get('mask_color_r', '0'))
            mask_color_g = int(request.form.get('mask_color_g', '255'))
            mask_color_b = int(request.form.get('mask_color_b', '0'))
            mask_color = (mask_color_b, mask_color_g, mask_color_r)  # Note: OpenCV uses BGR
            
            mask_alpha = float(request.form.get('mask_alpha', '0.5'))
            mask_alpha = max(0.1, min(1.0, mask_alpha))  # Constrain between 0.1 and 1.0
        except ValueError:
            # Use defaults if any conversion fails
            pass
        
        logger.info(f"Using mask color: {mask_color} and alpha: {mask_alpha}")

    # For image mode, ensure replacement provided
    replacement_img = None
    if mode == 'image':
        if 'replacement' not in request.files:
            abort(400, 'Missing replacement image')
        rep_file = request.files['replacement']
        rep_bytes = rep_file.read()
        rep_arr = np.frombuffer(rep_bytes, np.uint8)
        replacement_img = cv2.imdecode(rep_arr, cv2.IMREAD_COLOR)
        if replacement_img is None:
            abort(400, 'Failed to decode replacement image')

    # Create a persistent directory for processed videos
    storage_dir = os.path.join(os.getcwd(), 'processed_videos')
    os.makedirs(storage_dir, exist_ok=True)
    
    # Save uploaded video to a temp file
    tmpdir = tempfile.mkdtemp()
    in_path = os.path.join(tmpdir, video_file.filename)
    video_file.save(in_path)
    
    # Prepare output path in the persistent directory
    out_name = f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
    out_path = os.path.join(storage_dir, out_name)
    
    logger.info(f"Will save processed video to {out_path}")

    # Open input video
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        abort(500, f"Cannot open video {in_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Use a simple, reliable codec (mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.error("Failed to initialize mp4v VideoWriter")
        abort(500, "Video encoding not supported on server")
    
    logger.info(f"Using codec 'mp4v' for VideoWriter")

    # Configure predictors with user-defined thresholds if different from defaults
    billboard_predictor = billboard_predictor_default
    if conf_threshold != CONF_THRESH:
        billboard_predictor = setup_billboard_predictor(conf_threshold)
    
    human_predictor = human_predictor_default
    if human_conf_threshold != HUMAN_CONF_THRESH:
        human_predictor = setup_human_predictor(human_conf_threshold)

    # Process frames
    logger.info(f"Start processing {in_path} in mode={mode}")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            logger.info(f"Processing frame {frame_count}")
        
        # Get billboard detections
        billboard_outputs = billboard_predictor(frame)
        billboard_instances = billboard_outputs['instances'].to('cpu')
        
        # Create a combined human mask if human filtering is enabled
        combined_human_mask = None
        if enable_human_filter:
            # Detect humans and sports balls
            human_outputs = human_predictor(frame)
            human_instances = human_outputs['instances'].to('cpu')
            
            if human_instances.has('pred_masks') and human_instances.has('pred_classes'):
                human_masks_tensor = human_instances.pred_masks
                human_classes = human_instances.pred_classes
                human_scores = human_instances.scores
                
                # Create a combined mask for all humans and balls
                combined_human_mask = np.zeros((height, width), dtype=bool)
                
                # Person is class 0, sports ball is class 32 in COCO
                for i, class_id in enumerate(human_classes):
                    if (class_id == 0 or class_id == 32) and human_scores[i] >= human_conf_threshold:
                        # Add this human/ball to the combined mask
                        combined_human_mask = combined_human_mask | human_masks_tensor[i].numpy()
        
        # Process billboard masks if present
        if billboard_instances.has('pred_masks'):
            billboard_masks = billboard_instances.pred_masks.numpy()
            billboard_scores = billboard_instances.scores.numpy()
            
            for i, (mask, score) in enumerate(zip(billboard_masks, billboard_scores)):
                if score < conf_threshold:
                    continue
                
                # Filter by mask size if minimum size is set
                if min_mask_size > 0:
                    mask_size_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
                    if mask_size_ratio < min_mask_size:
                        continue
                
                # Subtract human masks from billboard mask if enabled
                filtered_mask = mask.copy()
                if enable_human_filter and combined_human_mask is not None:
                    # Remove areas where humans/balls are detected
                    filtered_mask = mask & ~combined_human_mask
                
                # Skip if entire mask was removed
                if filtered_mask.sum() == 0:
                    continue
                
                # Apply mask or replacement to the non-human areas
                if mode == 'mask':
                    # Apply color mask with user-defined color and alpha
                    m = filtered_mask.astype(bool)
                    for c in range(3):
                        frame[:, :, c][m] = frame[:, :, c][m] * (1 - mask_alpha) + mask_color[c] * mask_alpha
                else:
                    frame = replace_using_mask(frame, filtered_mask, replacement_img)
        
        writer.write(frame)

    cap.release()
    writer.release()
    logger.info(f"Finished processing, output at {out_path}")

    # Transcode video to H.264 for browser compatibility
    web_video_path = os.path.join(storage_dir, f"web_{os.path.basename(out_path)}")
    try:
        # Use FFmpeg to transcode to H.264 for browser compatibility
        logger.info(f"Transcoding video to H.264 for browser compatibility: {web_video_path}")
        subprocess.run([
            'ffmpeg', '-i', out_path, 
            '-c:v', 'libx264', '-preset', 'fast',  # Use H.264 codec
            '-pix_fmt', 'yuv420p',  # Required for browser compatibility
            '-movflags', '+faststart',  # For streaming optimization
            web_video_path
        ], check=True)
        logger.info(f"Transcoding complete: {web_video_path}")
        
        # Use the web-compatible version
        video_filename = os.path.basename(web_video_path)
    except Exception as e:
        # Fallback to original if transcoding fails
        logger.error(f"Failed to transcode video: {e}")
        video_filename = os.path.basename(out_path)

    # Return the filename
    return jsonify({
        'success': True,
        'filename': video_filename,
        'path': f'/videos/{video_filename}'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)