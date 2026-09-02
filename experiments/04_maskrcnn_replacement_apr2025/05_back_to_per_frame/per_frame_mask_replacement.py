#!/usr/bin/env python3
"""AdSwapAI R&D, 2025-05-08: deliberate simplification - no tracking, one predictor,
per-frame mask replacement (replace_using_mask); this became the core of the web app
(Detectron2 Mask R-CNN billboard model)."""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
import logging
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

MODEL_PATH = "model_final.pth"           # custom Detectron2 billboard model, see docs/assets.md
VIDEO_PATH = "data/adVideo1.mp4"        # sample clip, see repo docs/assets.md
REPLACEMENT_PATH = "data/replace.jpg"
OUTPUT_DIR = "output"

def setup_billboard_predictor(model_path: str, conf_thresh: float = 0.5):
    """
    Load a Detectron2 Mask R-CNN model for billboard detection.
    """
    cfg = get_cfg()
    # Base config
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    # Single class: billboard
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # Set model weights
    cfg.MODEL.WEIGHTS = os.path.abspath(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh
    # Use GPU if available
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        torch.cuda.init()
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        cfg.MODEL.DEVICE = "cpu"
        logging.info("GPU not available, using CPU")
    predictor = DefaultPredictor(cfg)
    return predictor


def replace_using_mask(frame: np.ndarray, mask: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    """
    Replace pixels in frame where mask is True with pixels from replacement image.
    The replacement is resized to the bounding box of the mask and applied exactly on masked pixels.
    """
    # Find bounding box of mask
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return frame
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h, w = y2 - y1 + 1, x2 - x1 + 1

    # Resize replacement image to bbox size
    rep_resized = cv2.resize(replacement, (w, h), interpolation=cv2.INTER_AREA)

    # Apply replacement only on mask region
    result = frame.copy()
    mask_crop = mask[y1:y2+1, x1:x2+1]
    region = result[y1:y2+1, x1:x2+1]

    # Vectorized replacement
    for c in range(3):
        channel = region[:, :, c]
        channel[mask_crop] = rep_resized[:, :, c][mask_crop]
        region[:, :, c] = channel
    result[y1:y2+1, x1:x2+1] = region
    return result


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("MaskLogicBillboards")

    # File paths
    model_path = MODEL_PATH
    input_video = VIDEO_PATH
    replacement_image = REPLACEMENT_PATH

    # Check files
    for f in [model_path, input_video, replacement_image]:
        if not os.path.exists(f):
            logger.error(f"File not found: {f}")
            sys.exit(1)

    # Prepare predictor and replacement
    predictor = setup_billboard_predictor(model_path)
    rep_img = cv2.imread(replacement_image)
    if rep_img is None:
        logger.error("Failed to load replacement image.")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_video}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output filename with hhmmss timestamp
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start = time.time()
    frame_idx = 0
    logger.info(f"Starting processing: {total_frames} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Predict (frame is automatically moved to correct device by DefaultPredictor)
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        if instances.has("pred_masks"):
            masks = instances.pred_masks.numpy()
            scores = instances.scores.numpy()
            # For each detection above threshold
            for mask, score in zip(masks, scores):
                if score < predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                    continue
                # Apply unified mask-based replacement
                frame = replace_using_mask(frame, mask, rep_img)

        writer.write(frame)
        # Log progress every 100 frames
        if frame_idx % 100 == 0:
            elapsed = time.time() - start
            logger.info(f"Frame {frame_idx}/{total_frames} ({frame_idx/total_frames:.1%}) - {elapsed:.1f}s elapsed")

    cap.release()
    writer.release()
    total_time = time.time() - start
    logger.info(f"Processing complete in {total_time:.1f}s. Output: {output_path}")

if __name__ == "__main__":
    main()
