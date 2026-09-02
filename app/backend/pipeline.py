#!/usr/bin/env python3
"""
AdSwapAI billboard detection & replacement pipeline.

Pure processing code with no web-framework dependency. It is used by
``app.py`` (Flask service) and ``cli.py`` (command line runner).

Pipeline per frame:
  1. Custom Mask R-CNN (Detectron2) detects billboard instances.
  2. Optional COCO Mask R-CNN detects people / sports balls; their pixels are
     subtracted from every billboard mask so foreground subjects stay visible.
  3. A light IoU tracker keeps a mask alive for a few frames when detection
     drops out, which removes most of the frame-to-frame flicker.
  4. Masks are split into "big" (wide perimeter boards) and "small" boards.
     Big boards get a horizontally repeated ad, small boards get a single
     perspective-warped ad. In mask mode every board is tinted instead.
  5. Frames are streamed to ffmpeg (libx264, faststart) so the output is
     web-ready in a single pass; the source audio track is kept.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

logger = logging.getLogger("adswap.pipeline")

BASE_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
COCO_PERSON = 0
COCO_SPORTS_BALL = 32
# Predictors are built once with a low score floor; per-job thresholds are
# applied in Python so a new threshold never triggers a model rebuild.
PREDICTOR_SCORE_FLOOR = 0.1


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_cfg(num_classes: int, weights: str, device: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = PREDICTOR_SCORE_FLOOR
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


class Models:
    """Holds the custom billboard predictor and the COCO predictor."""

    def __init__(self, billboard_weights: str, coco_weights: Optional[str] = None,
                 device: Optional[str] = None):
        self.device = device or get_device()
        if not os.path.isfile(billboard_weights):
            raise FileNotFoundError(f"Billboard weights not found: {billboard_weights}")

        t0 = time.time()
        self.billboard = DefaultPredictor(_make_cfg(1, os.path.abspath(billboard_weights), self.device))
        logger.info("Billboard predictor ready (%s, %.1fs)", self.device, time.time() - t0)

        if coco_weights and os.path.isfile(coco_weights):
            weights = os.path.abspath(coco_weights)
        else:
            weights = model_zoo.get_checkpoint_url(BASE_CONFIG)
            logger.info("Local COCO weights not found, using model zoo URL: %s", weights)
        t0 = time.time()
        self.coco = DefaultPredictor(_make_cfg(80, weights, self.device))
        logger.info("COCO predictor ready (%.1fs)", time.time() - t0)

    def warm_up(self, width: int = 1280, height: int = 720) -> None:
        """Run one dummy inference so the first real frame is not slow."""
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        with torch.inference_mode():
            self.billboard(dummy)
            self.coco(dummy)
        logger.info("Models warmed up")


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_billboards(predictor: DefaultPredictor, frame: np.ndarray,
                      conf_threshold: float) -> Tuple[List[np.ndarray], List[float]]:
    """Return billboard masks (bool HxW) and their scores above the threshold."""
    with torch.inference_mode():
        inst = predictor(frame)["instances"]
    if len(inst) == 0 or not inst.has("pred_masks"):
        return [], []
    scores = inst.scores.cpu().numpy()
    keep = scores >= conf_threshold
    if not keep.any():
        return [], []
    masks = inst.pred_masks.cpu().numpy()[keep].astype(bool)
    return list(masks), [float(s) for s in scores[keep]]


def detect_protected_mask(predictor: DefaultPredictor, frame: np.ndarray, conf_threshold: float,
                          classes: Sequence[int] = (COCO_PERSON, COCO_SPORTS_BALL)) -> Optional[np.ndarray]:
    """Union mask of people and sports balls, or None when nothing was found."""
    with torch.inference_mode():
        inst = predictor(frame)["instances"]
    if len(inst) == 0 or not inst.has("pred_masks"):
        return None
    cls = inst.pred_classes.cpu().numpy()
    scores = inst.scores.cpu().numpy()
    keep = np.isin(cls, list(classes)) & (scores >= conf_threshold)
    if not keep.any():
        return None
    masks = inst.pred_masks.cpu().numpy()[keep]
    return np.any(masks, axis=0)


# --------------------------------------------------------------------------- #
# Temporal smoothing
# --------------------------------------------------------------------------- #
def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def _bbox_overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


@dataclass
class _Track:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    misses: int = 0


class MaskTracker:
    """Keeps a billboard mask alive for ``hold_frames`` frames after detection drops out."""

    def __init__(self, hold_frames: int = 3, iou_threshold: float = 0.3):
        self.hold_frames = max(0, int(hold_frames))
        self.iou_threshold = iou_threshold
        self.tracks: List[_Track] = []

    def update(self, masks: Sequence[np.ndarray]) -> List[np.ndarray]:
        if self.hold_frames <= 0:
            return list(masks)

        matched = set()
        output: List[np.ndarray] = []
        for m in masks:
            box = _bbox(m)
            best_idx, best_iou = None, self.iou_threshold
            for idx, track in enumerate(self.tracks):
                if idx in matched or not _bbox_overlap(track.bbox, box):
                    continue
                iou = _mask_iou(track.mask, m)
                if iou > best_iou:
                    best_idx, best_iou = idx, iou
            if best_idx is None:
                self.tracks.append(_Track(m, box))
                matched.add(len(self.tracks) - 1)
            else:
                track = self.tracks[best_idx]
                track.mask, track.bbox, track.misses = m, box, 0
                matched.add(best_idx)
            output.append(m)

        survivors: List[_Track] = []
        for idx, track in enumerate(self.tracks):
            if idx in matched:
                survivors.append(track)
                continue
            track.misses += 1
            if track.misses <= self.hold_frames:
                survivors.append(track)
                output.append(track.mask)
        self.tracks = survivors
        return output


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _order_box(box: np.ndarray) -> np.ndarray:
    """Order the 4 corners of a rotated rectangle as [TL, TR, BR, BL].

    The two long edges are taken as top and bottom, so a thin, slanted
    perimeter strip still gets a valid (non-degenerate) quadrilateral where
    the x+y min/max heuristic collapses corners.
    """
    pts = box.reshape(4, 2).astype(np.float32)
    edge_len = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    if edge_len[0] + edge_len[2] >= edge_len[1] + edge_len[3]:
        edges = [(pts[0], pts[1]), (pts[2], pts[3])]
    else:
        edges = [(pts[1], pts[2]), (pts[3], pts[0])]
    edges.sort(key=lambda e: (e[0][1] + e[1][1]))       # smaller mean y = top edge
    top, bottom = edges
    tl, tr = sorted(top, key=lambda p: p[0])
    bl, br = sorted(bottom, key=lambda p: p[0])
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_from_contour(cnt: np.ndarray, snap_to_hull: bool = True) -> np.ndarray:
    """Quadrilateral [TL, TR, BR, BL] describing a board contour.

    Starts from the minimum-area rectangle (always well-formed) and, when
    requested, snaps each corner to the nearest convex-hull point so a board
    seen in perspective keeps its trapezoid shape. The snapped quad is only
    used when it stays clearly non-degenerate.
    """
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    quad = _order_box(box)
    if not snap_to_hull:
        return quad
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32)
    snapped = np.array([hull[np.argmin(np.linalg.norm(hull - c, axis=1))] for c in quad], dtype=np.float32)
    distinct = len({(round(float(p[0])), round(float(p[1]))) for p in snapped}) == 4
    if not distinct:
        return quad
    snapped_area = cv2.contourArea(snapped.reshape(-1, 1, 2))
    box_area = cv2.contourArea(box.reshape(-1, 1, 2))
    if box_area <= 0 or snapped_area < 0.5 * box_area:
        return quad
    return snapped


def _largest_contour(mask: np.ndarray):
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _quad_size(quad: np.ndarray) -> Tuple[int, int]:
    width = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    height = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    return int(round(width)), int(round(height))


def _warp_into_frame(texture: np.ndarray, quad: np.ndarray, frame_shape) -> Tuple[np.ndarray, np.ndarray]:
    """Warp ``texture`` onto ``quad``; returns (warped image, coverage mask)."""
    h, w = texture.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    size = (frame_shape[1], frame_shape[0])
    warped = cv2.warpPerspective(texture, matrix, size, flags=cv2.INTER_LINEAR)
    coverage = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
    return warped, coverage > 0


def blend_region(frame: np.ndarray, source: np.ndarray, mask: np.ndarray, feather: float = 2.0) -> np.ndarray:
    """Alpha-blend ``source`` into ``frame`` where ``mask`` is set, with a soft inner edge."""
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return frame
    cols = np.where(mask.any(axis=0))[0]
    pad = int(feather * 3) + 1
    y0, y1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, frame.shape[0])
    x0, x1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, frame.shape[1])

    region_mask = mask[y0:y1, x0:x1].astype(np.float32)
    if feather > 0:
        alpha = cv2.GaussianBlur(region_mask, (0, 0), feather)
        alpha = np.minimum(alpha, region_mask)  # feather inward only: no dark halo outside
    else:
        alpha = region_mask
    alpha = alpha[..., None]

    roi = frame[y0:y1, x0:x1].astype(np.float32)
    src = source[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi * (1.0 - alpha) + src * alpha + 0.5, 0, 255).astype(np.uint8)
    return frame


# --------------------------------------------------------------------------- #
# Rendering primitives
# --------------------------------------------------------------------------- #
def tint_mask(frame: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float) -> np.ndarray:
    if not mask.any():
        return frame
    color = np.array(color_bgr, dtype=np.float32)
    frame[mask] = np.clip(frame[mask].astype(np.float32) * (1.0 - alpha) + color * alpha + 0.5, 0, 255).astype(np.uint8)
    return frame


def replace_perspective(frame: np.ndarray, mask: np.ndarray, replacement: np.ndarray,
                        feather: float = 2.0, min_area: int = 100) -> np.ndarray:
    """Single ad warped onto the minimum-area rectangle of a (small) board."""
    cnt = _largest_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < min_area:
        return frame
    quad = _quad_from_contour(cnt, snap_to_hull=False)
    width, height = _quad_size(quad)
    if width < 8 or height < 4:
        return frame
    texture = cv2.resize(replacement, (width, height), interpolation=cv2.INTER_AREA)
    warped, coverage = _warp_into_frame(texture, quad, frame.shape)
    return blend_region(frame, warped, coverage & mask, feather)


def replace_tiled(frame: np.ndarray, mask: np.ndarray, replacement: np.ndarray,
                  feather: float = 2.0, min_area: int = 1_500) -> np.ndarray:
    """Ad repeated horizontally (aspect preserved) across a wide perimeter board."""
    cnt = _largest_contour(mask)
    if cnt is None:
        return frame
    area = cv2.contourArea(cnt)
    if area < min_area:
        logger.debug("Tiled replacement skipped, contour area too small: %.0f", area)
        return frame

    quad = _quad_from_contour(cnt, snap_to_hull=True)
    width, height = _quad_size(quad)
    if width < 30 or height < 6:
        logger.debug("Tiled replacement skipped, board too small: %dx%d", width, height)
        return frame

    rep_h, rep_w = replacement.shape[:2]
    tile_w = max(1, int(round(height * rep_w / max(rep_h, 1))))
    tile = cv2.resize(replacement, (tile_w, height), interpolation=cv2.INTER_AREA)
    repeats = max(1, int(math.ceil(width / tile_w)))
    texture = np.tile(tile, (1, repeats, 1))[:, :width]

    try:
        warped, coverage = _warp_into_frame(texture, quad, frame.shape)
    except cv2.error as exc:
        logger.warning("Perspective warp failed (%s), falling back to flat fill", exc)
        flat = cv2.resize(replacement, (frame.shape[1], frame.shape[0]))
        return blend_region(frame, flat, mask, feather)

    paste = coverage & mask
    if not paste.any():
        return frame
    return blend_region(frame, warped, paste, feather)


# --------------------------------------------------------------------------- #
# Video writers
# --------------------------------------------------------------------------- #
class FFmpegWriter:
    """Streams raw BGR frames to ffmpeg and encodes H.264 (web-ready) in one pass."""

    def __init__(self, path: str, width: int, height: int, fps: float,
                 audio_source: Optional[str] = None, crf: int = 20, preset: str = "fast"):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.6f}", "-i", "pipe:0",
        ]
        if audio_source:
            cmd += ["-i", audio_source, "-map", "0:v:0", "-map", "1:a?",
                    "-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += [
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
        ]
        self.path = path
        # stderr goes to a temp file: a filled-up pipe would block ffmpeg and deadlock the writer
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._stderr)
        self.frames = 0

    def _read_stderr(self) -> str:
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode(errors="ignore").strip()
        except Exception:  # noqa: BLE001
            return ""
        finally:
            self._stderr.close()

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            self.frames += 1
        except (BrokenPipeError, OSError):
            self.proc.wait()
            raise RuntimeError(f"ffmpeg stopped: {self._read_stderr()}")

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait()
        err = self._read_stderr()
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({self.proc.returncode}): {err}")


class OpenCVWriter:
    """Fallback writer (mp4v) used only when ffmpeg is unavailable."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError("cv2.VideoWriter could not be opened")
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)
        self.frames += 1

    def close(self) -> None:
        self.writer.release()


def open_video_writer(path: str, width: int, height: int, fps: float, audio_source: Optional[str] = None):
    if shutil.which("ffmpeg"):
        try:
            return FFmpegWriter(path, width, height, fps, audio_source=audio_source)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ffmpeg writer unavailable (%s), falling back to OpenCV", exc)
    else:
        logger.warning("ffmpeg not found on PATH, falling back to OpenCV writer (mp4v)")
    return OpenCVWriter(path, width, height, fps)


# --------------------------------------------------------------------------- #
# Job parameters and main loop
# --------------------------------------------------------------------------- #
@dataclass
class JobParams:
    mode: str = "image"                    # "image" or "mask"
    conf_threshold: float = 0.5            # billboard score threshold
    human_conf_threshold: float = 0.5      # person / ball score threshold
    min_mask_size: float = 0.0             # min mask area as a fraction of the frame
    enable_human_filter: bool = True
    mask_color_bgr: Tuple[int, int, int] = (0, 255, 0)
    mask_alpha: float = 0.5
    hold_frames: int = 3                   # temporal smoothing (0 disables)
    big_width_ratio: float = 0.6           # wider than this fraction of the frame => "big" board
    feather: float = 2.0                   # edge softness in pixels


class JobCancelled(Exception):
    pass


def decode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def render_frame(frame: np.ndarray, masks: Sequence[np.ndarray], protected: Optional[np.ndarray],
                 params: JobParams, replacement: Optional[np.ndarray]) -> np.ndarray:
    """Apply mask tint or ad replacement for one frame."""
    width = frame.shape[1]
    out = frame.copy()
    big, small = [], []
    for m in masks:
        if protected is not None:
            m = m & ~protected
        cols = np.where(m.any(axis=0))[0]
        if cols.size == 0:
            continue
        rel_width = (cols[-1] - cols[0]) / width
        (big if rel_width > params.big_width_ratio else small).append(m)

    if params.mode == "mask" or replacement is None:
        for m in big + small:
            tint_mask(out, m, params.mask_color_bgr, params.mask_alpha)
        return out

    for m in big:      # background first
        out = replace_tiled(out, m, replacement, params.feather)
    for m in small:    # then the nearer boards
        out = replace_perspective(out, m, replacement, params.feather)
    return out


ProgressCallback = Callable[[int, int], None]


def process_video(in_path: str, out_path: str, params: JobParams, models: Models,
                  replacement: Optional[np.ndarray] = None,
                  progress_cb: Optional[ProgressCallback] = None,
                  should_cancel: Optional[Callable[[], bool]] = None) -> dict:
    """Process a whole video file. Returns a small stats dict."""
    if params.mode == "image" and replacement is None:
        raise ValueError("Image mode requires a replacement image")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video reports invalid dimensions")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = open_video_writer(out_path, width, height, fps, audio_source=in_path)
    tracker = MaskTracker(params.hold_frames)
    frame_area = float(width * height)
    frames = 0
    t0 = time.time()
    failed = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            if should_cancel and should_cancel():
                raise JobCancelled("Cancelled by user")

            masks, _ = detect_billboards(models.billboard, frame, params.conf_threshold)
            if params.min_mask_size > 0:
                masks = [m for m in masks if m.sum() / frame_area >= params.min_mask_size]
            masks = tracker.update(masks)

            protected = None
            if params.enable_human_filter:
                protected = detect_protected_mask(models.coco, frame, params.human_conf_threshold)

            writer.write(render_frame(frame, masks, protected, params, replacement))

            if progress_cb and (frames % 5 == 0 or frames == total):
                progress_cb(frames, total)
    except BaseException:
        failed = True
        raise
    finally:
        cap.release()
        try:
            writer.close()
        except Exception as exc:  # noqa: BLE001
            if not failed:
                raise
            logger.warning("Writer close after failure: %s", exc)

    elapsed = time.time() - t0
    stats = {
        "frames": frames, "width": width, "height": height, "fps": fps,
        "elapsed_seconds": round(elapsed, 2),
        "processing_fps": round(frames / elapsed, 2) if elapsed > 0 else None,
    }
    logger.info("Processed %s -> %s: %s", in_path, out_path, stats)
    return stats
