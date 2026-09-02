#!/usr/bin/env python3
"""AdSwapAI R&D, 2025-05-24: final May 2025 Flask backend (job tracking, threaded processing, perspective paste, smart tiling for wide boards, ffmpeg transcode)."""
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
import uuid
from threading import Thread

processing_status = {
    "is_processing": False,
    "current_frame": 0,
    "total_frames": 0,
    "file_name": ""
}

processing_jobs = {}

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

    # 1) Find the contour and extract the minimum-area rectangle
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 100:
        return frame

    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect).astype(np.float32)  # 4×2

    # 2) Sort the box points into [TL, TR, BR, BL] order
    #    (this way we always compute a proper perspective matrix)
    def order_pts(pts):
        # Top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    dst_pts = order_pts(box)

    # 3) Fix the target width and height based on the real billboard ratio
    widthA  = np.linalg.norm(dst_pts[2] - dst_pts[3])
    widthB  = np.linalg.norm(dst_pts[1] - dst_pts[0])
    maxW    = int(max(widthA, widthB))

    heightA = np.linalg.norm(dst_pts[1] - dst_pts[2])
    heightB = np.linalg.norm(dst_pts[0] - dst_pts[3])
    maxH    = int(max(heightA, heightB))

    # 4) Resize the replacement to these dimensions
    if maxW < 10 or maxH < 10:
        return frame
    rep_resized = cv2.resize(replacement, (maxW, maxH), interpolation=cv2.INTER_AREA)

    # 5) Define source points for the perspective transform
    src_pts = np.array([
        [0,      0],
        [maxW-1, 0],
        [maxW-1, maxH-1],
        [0,      maxH-1]
    ], dtype=np.float32)

    # 6) Compute the homography and warp
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(rep_resized, M, (frame.shape[1], frame.shape[0]))

    # 7) Place only the warped area onto the frame
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_warp = mask_bin.astype(bool)

    out = frame.copy()
    out[mask_warp] = warped[mask_warp]
    return out

def smart_tile_replacement(frame: np.ndarray, mask: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    try:
        logger.info(f"Smart tiling called - mask size: {mask.sum()}, replacement shape: {replacement.shape}")
    
        ys, xs = np.where(mask)
        if ys.size == 0:
            logger.warning("Empty mask in smart tiling")
            return frame
        
        # 1) Find the contour
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("No contours found")
            return frame
        
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        logger.info(f"Contour area: {area}")
        
        # ISSUE 1: This threshold might be too low!
        if area < 50000:  # Too small for big ads!
            logger.warning(f"Contour area too small: {area}")
            return frame
        
        # 2) Corner detection - find sharp corners
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        logger.info(f"Initial corners found: {len(approx)}")

        # 3) If it isn't 4-cornered, find the best 4 corners
        if len(approx) != 4:
            try:
                corners = cv2.goodFeaturesToTrack(mask_u8, maxCorners=4, qualityLevel=0.01, minDistance=50)
                if corners is not None and len(corners) == 4:
                    approx = corners.reshape(-1, 1, 2).astype(np.int32)
                    logger.info(f"Harris corners used: {len(corners)}")
                else:
                    logger.warning(f"Harris corners failed, found: {len(corners) if corners is not None else 0}")
            except Exception as e:
                logger.error(f"goodFeaturesToTrack exception: {e}")

        if len(approx) < 4:
            # Fallback to simple rectangle
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            approx = box.reshape(-1, 1, 2).astype(np.int32)
            logger.info("Using minAreaRect fallback")

        logger.info(f"Final corners count: {len(approx)}")

        def order_pts(pts):
            # Top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        # 4) Arrange for the perspective transform
        def simple_order_corners(pts):
            """Custom corner ordering for big ads"""
            pts = pts.reshape(4, 2)

            center = pts.mean(axis=0)

            # Assign the 4 corners based on the bounding box
            rect = np.zeros((4, 2), dtype="float32")

            # Assign each corner to the nearest bounding-box corner
            for pt in pts:
                if pt[0] < center[0] and pt[1] < center[1]:  # Top left
                    rect[0] = pt
                elif pt[0] > center[0] and pt[1] < center[1]:  # Top right
                    rect[1] = pt
                elif pt[0] > center[0] and pt[1] > center[1]:  # Bottom right
                    rect[2] = pt
                else:  # Bottom left
                    rect[3] = pt

            return rect

        if mask.sum() > 80000:  # For big ads
            dst_pts = simple_order_corners(approx[:4])
            logger.info("Using simple corner ordering for large ads")
        else:
            dst_pts = order_pts(approx[:4])
            
        logger.info(f"Ordered corners: {dst_pts}")

        # 5) Compute real-world dimensions
        width1 = np.linalg.norm(dst_pts[1] - dst_pts[0])
        width2 = np.linalg.norm(dst_pts[2] - dst_pts[3])
        height1 = np.linalg.norm(dst_pts[3] - dst_pts[0])
        height2 = np.linalg.norm(dst_pts[2] - dst_pts[1])

        max_width = int(max(width1, width2))
        max_height = int(max(height1, height2))

        logger.info(f"Calculated dimensions: {max_width}x{max_height}")

        if max_width < 50 or max_height < 50:
            logger.warning(f"Dimensions too small: {max_width}x{max_height}")
            return frame

        # 6) Build the tiling pattern - FIX
        tile_size = min(replacement.shape[:2])  # 400 (too small!)

        # NEW: bigger tile size
        tile_size = max(200, min(max_width//3, max_height//2))  # More reasonable size
        tiles_x = max(1, max_width // tile_size + 1)
        tiles_y = max(1, max_height // tile_size + 1)

        logger.info(f"Tile pattern: {tiles_x}x{tiles_y}, tile_size: {tile_size}")

        # Limit if there end up being too many tiles
        if tiles_x * tiles_y > 6:  # Maximum 6 tiles
            tile_size = max(max_width//2, max_height//1)  # Make it bigger
            tiles_x = max(1, max_width // tile_size + 1) 
            tiles_y = max(1, max_height // tile_size + 1)
            logger.info(f"Reduced tile pattern: {tiles_x}x{tiles_y}, tile_size: {tile_size}")

        # Tile the replacement
        rep_resized = cv2.resize(replacement, (tile_size, tile_size))
        tiled = np.tile(rep_resized, (tiles_y, tiles_x, 1))
        tiled = tiled[:max_height, :max_width]  # Crop to size

        logger.info(f"Tiled image shape: {tiled.shape}")

        # 7) Perspective warp
        src_pts = np.array([
            [0, 0], [max_width-1, 0], 
            [max_width-1, max_height-1], [0, max_height-1]
        ], dtype=np.float32)

        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(tiled, M, (frame.shape[1], frame.shape[0]))
            logger.info("Perspective transform successful")
            
            # 8) Apply the mask
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, warp_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            final_mask = warp_mask.astype(bool) & mask
            
            applied_pixels = final_mask.sum()
            logger.info(f"Applied to {applied_pixels} pixels")
            
            if applied_pixels == 0:
                logger.warning("NO PIXELS APPLIED - final mask is empty!")
                return frame
            
            out = frame.copy()
            out[final_mask] = warped[final_mask]
            logger.info(f"SUCCESS: Smart tiling applied to {applied_pixels} pixels")
            return out
            
        except Exception as e:
            logger.error(f"Perspective transform failed: {e}")
            logger.error(f"src_pts: {src_pts}")
            logger.error(f"dst_pts: {dst_pts}")
            return frame
    except Exception as e:
        logger.error(f"Smart tiling failed: {e}, falling back to simple overlay")
        # Simple overlay as fallback
        overlay = cv2.resize(replacement, (frame.shape[1], frame.shape[0]))
        result = frame.copy()
        result[mask] = overlay[mask]
        return result

# Flask app setup
app = Flask(__name__)

@app.route('/process-status', methods=['GET'])
def process_status():
    """Return the current processing status."""
    job_id = request.args.get('job_id')
    browser_session = request.args.get('browser_session')
    
    # Log the request
    logging.info(f"Status request received - job_id: {job_id}, session: {browser_session}")
    
    # If specific job ID requested
    if job_id and job_id in processing_jobs:
        job_info = processing_jobs[job_id].copy()
        
        # Add video readiness info
        if not job_info.get("is_processing", True) and "final_output_filename" in job_info:
            video_filename = job_info["final_output_filename"]
            video_path = os.path.join(os.getcwd(), 'processed_videos', video_filename)
            
            # Check if the video is ready
            job_info['video_ready'] = is_video_ready(video_path)
            if job_info['video_ready']:
                job_info['video_size'] = os.path.getsize(video_path)
                job_info['video_url'] = f'/videos/{video_filename}'
        
        # Format response for compatibility with frontend
        logging.info(f"Returning status for job {job_id}: processing={job_info.get('is_processing')}, video_ready={job_info.get('video_ready', False)}")
        return jsonify({job_id: job_info})
    
    # If browser session filter is provided
    elif browser_session:
        # Return all jobs for this browser session
        session_jobs = {}
        for jid, job in processing_jobs.items():
            if job.get('browser_session') == browser_session:
                session_jobs[jid] = job
        return jsonify(session_jobs)
    
    # Return all jobs if no specific parameters
    return jsonify(processing_jobs)

@app.route('/process', methods=['POST'])
def process():
    client_job_id = request.form.get('client_job_id') or str(uuid.uuid4())
    browser_session = request.form.get('browser_session', 'unknown-browser')
    job_id = client_job_id

    # Validate video file
    if 'video' not in request.files:
        abort(400, 'Missing video file')
    video_file = request.files['video']
    mode = request.form.get('mode', 'mask')

    processing_jobs[job_id] = {
        "is_processing": True,
        "current_frame": 0,
        "total_frames": 0,
        "file_name": video_file.filename,
        "job_id": job_id,
        "browser_session": browser_session,
        "start_time": time.time()
    }

    # Params
    def get_float(name, default, minval=None, maxval=None):
        try:
            v = float(request.form.get(name, default))
            if minval is not None: v = max(minval, v)
            if maxval is not None: v = min(maxval, v)
            return v
        except: return default

    conf_threshold = get_float('conf_threshold', 0.5, 0.1, 0.9)
    human_conf_threshold = get_float('human_conf_threshold', 0.5, 0.1, 0.9)
    min_mask_size = get_float('min_mask_size', 0, 0)
    enable_human_filter = request.form.get('enable_human_filter', 'false').lower() in ['true', '1', 'yes', 'on']

    mask_color = (0, 255, 0)
    mask_alpha = 0.5
    if mode == 'mask':
        try:
            mask_color_r = int(request.form.get('mask_color_r', '0'))
            mask_color_g = int(request.form.get('mask_color_g', '255'))
            mask_color_b = int(request.form.get('mask_color_b', '0'))
            mask_color = (mask_color_b, mask_color_g, mask_color_r)
            mask_alpha = get_float('mask_alpha', 0.5, 0.1, 1.0)
        except: pass

    replacement_img = None
    if mode == 'image':
        if 'replacement' not in request.files:
            abort(400, 'Missing replacement image')
        rep_file = request.files['replacement']
        rep_bytes = rep_file.read()
        rep_arr = np.frombuffer(rep_bytes, np.uint8)
        replacement_img = cv2.imdecode(rep_arr, cv2.IMREAD_COLOR)
        logger.info(f"Replacement image loaded: {replacement_img.shape if replacement_img is not None else 'FAILED'}")
        if replacement_img is None:
            abort(400, 'Failed to decode replacement image')

    # File save
    storage_dir = os.path.join(os.getcwd(), 'processed_videos')
    os.makedirs(storage_dir, exist_ok=True)
    tmpdir = tempfile.mkdtemp()
    in_path = os.path.join(tmpdir, video_file.filename)
    video_file.save(in_path)
    out_name = f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
    out_path = os.path.join(storage_dir, out_name)

    processing_jobs[job_id]["output_filename"] = out_name
    processing_jobs[job_id]["web_output_filename"] = f"web_{out_name}"

    logger.info(f"Job {job_id}: Processing {in_path} -> {out_path}")

    # ---- START ASYNC PROCESS ----
    t = Thread(
        target=process_video_job,
        args=(
            in_path, out_path, mode, mask_color, mask_alpha, replacement_img,
            job_id, conf_threshold, human_conf_threshold, min_mask_size, enable_human_filter
        ),
        daemon=True
    )
    t.start()

    # ** Return the response IMMEDIATELY! **
    return jsonify({
        'success': True,
        'message': 'Processing started',
        'job_id': job_id
    })

@app.route('/all-jobs', methods=['GET'])
def all_jobs():
    """Return all jobs."""
    return jsonify(processing_jobs)

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
    video_path = os.path.join(video_dir, filename)
    
    # Check if the video is ready
    if not is_video_ready(video_path):
        logging.error(f"Video file not found or not ready: {video_path}")
        abort(404, f"Video file not found or not ready: {filename}")

    # Log the file size
    try:
        file_size = os.path.getsize(video_path)
        logging.info(f"Serving video {filename}, size: {file_size} bytes")
    except Exception as e:
        logging.warning(f"Could not get file size for {filename}: {e}")

    # Send the video file
    try:
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True  # Support ETag and Range
        )
    except Exception as e:
        logging.error(f"Error serving video {filename}: {e}")
        abort(500, f"Error serving video: {e}")

@app.route('/processed_videos', methods=['GET'])
def list_processed_videos():
    video_dir = os.path.join(os.getcwd(), 'processed_videos')
    try:
        files = sorted(os.listdir(video_dir))
        # Filter if desired, e.g. only .mp4
        videos = [f for f in files if f.lower().endswith('.mp4')]
        return jsonify(videos)
    except Exception as e:
        abort(500, f"Could not list videos: {e}")

def process_video_job(
    in_path, out_path, mode, mask_color, mask_alpha, replacement_img,
    job_id, conf_threshold, human_conf_threshold, min_mask_size, enable_human_filter
):
    try:
        storage_dir = os.path.join(os.getcwd(), 'processed_videos')
        cap = cv2.VideoCapture(in_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processing_jobs[job_id]["total_frames"] = total_frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not cap.isOpened():
            processing_jobs[job_id]["is_processing"] = False
            processing_jobs[job_id]["error"] = f"Cannot open video {in_path}"
            return
        if not writer.isOpened():
            processing_jobs[job_id]["is_processing"] = False
            processing_jobs[job_id]["error"] = "Video encoding not supported on server"
            return

        billboard_predictor = billboard_predictor_default
        if conf_threshold != CONF_THRESH:
            billboard_predictor = setup_billboard_predictor(conf_threshold)
        human_predictor = human_predictor_default
        if human_conf_threshold != HUMAN_CONF_THRESH:
            human_predictor = setup_human_predictor(human_conf_threshold)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            processing_jobs[job_id]["current_frame"] = frame_count
            if frame_count % 10 == 0:
                logger.info(f"Processing frame {frame_count}")

            # 1) Detect billboards
            billboard_outputs = billboard_predictor(frame)
            billboard_instances = billboard_outputs['instances'].to('cpu')

            # 2) Build combined human/ball mask if enabled
            combined_human_mask = None
            if enable_human_filter:
                human_outputs = human_predictor(frame)
                human_instances = human_outputs['instances'].to('cpu')
                if human_instances.has('pred_masks') and human_instances.has('pred_classes'):
                    combined_human_mask = np.zeros((height, width), dtype=bool)
                    for i, cls in enumerate(human_instances.pred_classes):
                        if cls in [0, 32] and human_instances.scores[i] >= human_conf_threshold:
                            combined_human_mask |= human_instances.pred_masks[i].numpy()

            # 3) Split masks into “big” vs “small” by width >60% frame
            big_masks, small_masks = [], []
            if billboard_instances.has('pred_masks'):
                masks = billboard_instances.pred_masks.numpy()
                scores = billboard_instances.scores.numpy()
                for mask, score in zip(masks, scores):
                    if score < conf_threshold:
                        continue
                    if min_mask_size > 0 and mask.sum()/(height*width) < min_mask_size:
                        continue
                    m = mask.copy()
                    if combined_human_mask is not None:
                        m &= ~combined_human_mask
                    if m.sum() == 0:
                        continue

                    xs = np.where(m)[1]
                    rel_width = (xs.max() - xs.min()) / width
                    if rel_width > 0.6:
                        big_masks.append(m)
                    else:
                        small_masks.append(m)

            # 4) Apply replacements in Z-order
            frame_out = frame.copy()

            # 4a) Back: big ads -> 90% white
            if replacement_img is not None:  # If a replacement image is provided
                logger.info(f"Processing {len(big_masks)} big masks with smart tiling")
                for i, m in enumerate(big_masks):
                    logger.info(f"Processing big mask {i}, area: {m.sum()}")
                    frame_out = smart_tile_replacement(frame_out, m, replacement_img)
            else:
                logger.info(f"No replacement image, using white paint for {len(big_masks)} big masks")
                # Fallback: paint white (existing behavior)
                for m in big_masks:
                    for c in range(3):
                        frame_out[:, :, c][m] = (
                            frame_out[:, :, c][m] * (1 - 0.9)
                            + 255 * 0.9
                        )

            # 4b) Middle: small ads
            if mode == 'mask':
                for m in small_masks:
                    for c in range(3):
                        frame_out[:, :, c][m] = (
                            frame_out[:, :, c][m] * (1 - mask_alpha)
                            + mask_color[c] * mask_alpha
                        )
            else:  # image mode
                for m in small_masks:
                    frame_out = replace_using_mask(frame_out, m, replacement_img)

            # 4c) Front: humans/ball were never overwritten (we subtracted them)

            writer.write(frame_out)

        cap.release()
        writer.release()
        processing_jobs[job_id]["is_processing"] = False
        processing_jobs[job_id]["end_time"] = time.time()

        # Transcode
        web_video_path = os.path.join(storage_dir, f"web_{os.path.basename(out_path)}")
        try:
            subprocess.run([
                'ffmpeg', '-i', out_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                web_video_path
            ], check=True)
            video_filename = os.path.basename(web_video_path)
            processing_jobs[job_id]["video_ready"] = is_video_ready(web_video_path)
            processing_jobs[job_id]["final_output_filename"] = video_filename
        except Exception as e:
            processing_jobs[job_id]["is_processing"] = False
    except Exception as e:
        processing_jobs[job_id]["is_processing"] = False
        # Fallback: still save the main video file
        video_filename = os.path.basename(out_path)
        processing_jobs[job_id]["final_output_filename"] = video_filename
        processing_jobs[job_id]["video_ready"] = is_video_ready(out_path)


    cleanup_old_jobs()

def cleanup_old_jobs():
    """Remove jobs older than 2 hours if completed, 5 hours if stuck"""
    current_time = time.time()
    to_remove = []
    
    for job_id, job in processing_jobs.items():
        # If job has end_time and it's more than 2 hours old
        if not job.get("is_processing") and job.get("end_time") and (current_time - job["end_time"]) > 7200:
            to_remove.append(job_id)
        # Or if job is too old even if not completed (5 hours)
        elif job.get("start_time") and (current_time - job["start_time"]) > 18000:
            to_remove.append(job_id)
    
    for job_id in to_remove:
        logger.info(f"Cleaning up old job: {job_id}")
        del processing_jobs[job_id]

# Add elsewhere in app.py, as a helper function
def is_video_ready(video_path, min_size=1024):
    """Check if video file exists and is not empty"""
    try:
        if not os.path.exists(video_path):
            return False
        
        # Minimum size check - very small files might be corrupt
        size = os.path.getsize(video_path)
        return size > min_size
    except Exception as e:
        logger.error(f"Error checking video file {video_path}: {e}")
        return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)