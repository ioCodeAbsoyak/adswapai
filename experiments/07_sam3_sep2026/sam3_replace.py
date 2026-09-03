"""AdSwapAI R&D, 2026-09-03, step 4: replace every tracked board with a real ad, clean output video.

Detection + camera-motion tracking come from sam3_hybrid_track.py (SAM3 every
DETECT_EVERY-th frame, RAFT homography in between, everything on the GPU).
For every track and frame this script:
  1. takes the board's mask, fits a quadrilateral to it (minimum-area rectangle
     with the long edges as top/bottom, corners snapped to the convex hull so a
     board seen in perspective keeps its trapezoid),
  2. maps the ad into that quadrilateral on the GPU (inverse homography per
     pixel of the board's bounding box, bilinear sampling of the ad texture):
     wide boards get the ad repeated with its aspect ratio preserved, narrow
     boards get its centre crop,
  3. paints the ad where the mask is set (players standing in front of a board
     are not in the mask, so they stay in front of the ad).
No boxes, no labels. The source audio track is copied. A contact sheet with
original | replaced pairs is written next to the video.
"""
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import sam3_hybrid_track as H

# ----------------------------------------------------------------------------- settings
VIDEO = "data/2.mp4"
AD_IMAGE = "../../app/frontend/static/images/bilboardsArtboard4.jpg"
MAX_FRAMES = None            # None = whole clip
DETECT_EVERY = 5             # SAM3 on every N-th frame, RAFT homography propagation in between
                             # (clip 2: 5 -> 26 fps, match IoU 0.81; 3 -> 19 fps, 0.84)
MIN_QUAD = (24, 6)           # boards narrower / lower than this (px) are left untouched
FEATHER = 1.0                # soften the mask edge by this many pixels (0 = hard edge)
KEEP_AUDIO = True
SHEET_EVERY = 40             # original | replaced pairs in the contact sheet
OUTPUT_DIR = "output/replace"
# --------------------------------------------------------------------------------------
DEVICE = H.DEVICE


# ----------------------------------------------------------------------------- board geometry
def order_quad(box):
    """Corners of a rotated rectangle as [TL, TR, BR, BL]; the two long edges become top and bottom."""
    pts = box.reshape(4, 2).astype(np.float32)
    edge = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    edges = [(pts[0], pts[1]), (pts[2], pts[3])] if edge[0] + edge[2] >= edge[1] + edge[3] else [(pts[1], pts[2]), (pts[3], pts[0])]
    edges.sort(key=lambda e: e[0][1] + e[1][1])
    (a, b), (c, d) = edges
    tl, tr = sorted((a, b), key=lambda p: p[0])
    bl, br = sorted((c, d), key=lambda p: p[0])
    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_of_mask(mask_u8):
    """Quadrilateral [TL, TR, BR, BL] of the largest blob in a uint8 mask, or None."""
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    quad = order_quad(box)
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32)
    snapped = np.array([hull[np.argmin(np.linalg.norm(hull - c, axis=1))] for c in quad], dtype=np.float32)
    distinct = len({(round(float(p[0])), round(float(p[1]))) for p in snapped}) == 4
    if distinct:
        box_area = cv2.contourArea(box.reshape(-1, 1, 2))
        if box_area > 0 and cv2.contourArea(snapped.reshape(-1, 1, 2)) >= 0.5 * box_area:
            return snapped
    return quad


# ----------------------------------------------------------------------------- rendering
class AdRenderer:
    def __init__(self, ad_path):
        ad = cv2.imread(ad_path, cv2.IMREAD_COLOR)
        if ad is None:
            raise FileNotFoundError(ad_path)
        self.ad = torch.from_numpy(ad).to(DEVICE).permute(2, 0, 1).float().unsqueeze(0)   # (1,3,h,w) BGR
        self.aspect = ad.shape[1] / ad.shape[0]
        self.unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self.boards = self.fits = 0

    @torch.no_grad()
    def paint(self, out, track):
        """Paint the ad into ``out`` (H,W,3 float on the GPU) where the track's mask is set."""
        x0, y0, x1, y1 = [int(v) for v in track["box"]]
        x0, y0 = max(x0 - 1, 0), max(y0 - 1, 0)
        x1, y1 = min(x1 + 2, out.shape[1]), min(y1 + 2, out.shape[0])
        if x1 - x0 < MIN_QUAD[0] or y1 - y0 < MIN_QUAD[1]:
            return
        mask = track["full"][y0:y1, x0:x1]
        quad = track.get("quad")
        if quad is None:                     # fitted once per detection; propagated frames move the corners with H
            quad = quad_of_mask(mask.to(torch.uint8).cpu().numpy() * 255)
            if quad is None:
                return
            quad = quad + np.array([x0, y0], dtype=np.float32)
            track["quad"] = quad
            self.fits += 1
        quad = quad - np.array([x0, y0], dtype=np.float32)
        qw = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
        qh = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
        if qw < MIN_QUAD[0] or qh < MIN_QUAD[1]:
            return
        # pixel (x, y) of the crop -> (u, v) in the unit square of the board
        M = cv2.getPerspectiveTransform(self.unit, quad)
        Minv = torch.as_tensor(np.linalg.inv(M), dtype=torch.float32, device=DEVICE)
        h, w = y1 - y0, x1 - x0
        ys, xs = torch.meshgrid(torch.arange(h, device=DEVICE, dtype=torch.float32) + 0.5,
                                torch.arange(w, device=DEVICE, dtype=torch.float32) + 0.5, indexing="ij")
        p = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3) @ Minv.T
        uv = p[:, :2] / p[:, 2:3].clamp(min=1e-6)
        u, v = uv[:, 0].reshape(h, w), uv[:, 1].reshape(h, w)
        inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        board_aspect = qw / qh
        if board_aspect >= self.aspect:                                   # wide strip: repeat the ad
            tiles = max(1.0, round(board_aspect / self.aspect))
            u_src = (u * tiles) % 1.0
        else:                                                            # narrow board: centre crop of the ad
            u_src = 0.5 + (u - 0.5) * (board_aspect / self.aspect)
        grid = torch.stack([u_src * 2 - 1, v * 2 - 1], -1).unsqueeze(0)
        sample = F.grid_sample(self.ad, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]  # (3,h,w)
        paint = (mask & inside).float()
        if FEATHER > 0:
            k = int(FEATHER * 2) * 2 + 1
            alpha = F.avg_pool2d(paint[None, None], k, stride=1, padding=k // 2)[0, 0]
            alpha = torch.minimum(alpha, paint)                             # feather inwards only
        else:
            alpha = paint
        region = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = region * (1 - alpha[..., None]) + sample.permute(1, 2, 0) * alpha[..., None]
        self.boards += 1


class AudioFFmpegWriter(H.FFmpegWriter):
    """FFmpegWriter that also copies the source audio track."""

    def __init__(self, path, width, height, fps, audio_source):
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "pipe:0",
               "-i", audio_source, "-map", "0:v:0", "-map", "1:a?", "-c:a", "aac", "-b:a", "128k", "-shortest",
               "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "23", "-b:v", "0",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._stderr)


class OrderedSink:
    """Writes frames on a thread, in order."""

    def __init__(self, writer):
        self.writer, self.error, self.q = writer, None, queue.Queue(maxsize=8)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    return
                self.writer.write(item)
        except Exception as exc:  # noqa: BLE001
            self.error = exc

    def put(self, frame):
        if self.error:
            raise RuntimeError(f"writer thread failed: {self.error!r}")
        self.q.put(frame)

    def close(self):
        self.q.put(None)
        self.thread.join()
        self.writer.release()
        if self.error:
            raise RuntimeError(f"writer thread failed: {self.error!r}")


# ----------------------------------------------------------------------------- main loop
def main():
    from sam3.model.sam3_image_processor import Sam3Processor

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_load = time.time()
    model = H.build_model()
    processor = Sam3Processor(model, confidence_threshold=H.CONFIDENCE)
    detector = H.Sam3Detector(processor, H.PROMPT)
    renderer = AdRenderer(AD_IMAGE)
    load_s = time.time() - t_load

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, MAX_FRAMES) if MAX_FRAMES else total
    stem = Path(VIDEO).stem
    ad_stem = Path(AD_IMAGE).stem
    frame_pixels = float(width * height)
    box_scale = torch.tensor([width, height, width, height], device=DEVICE, dtype=torch.float32)
    motion = H.CameraMotion(width, height)

    t_warm = time.time()
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {VIDEO}")
    first_gpu = torch.from_numpy(first).to(DEVICE)
    for _ in range(H.WARMUP_FRAMES):
        logits, probs, boxes = detector.detect(detector.backbone(first_gpu))
        H.clean_detections(logits, probs, boxes, frame_pixels)
        motion.update(first_gpu)
    motion.reset()
    motion.inlier_ratios, motion.failures = [], 0
    H.warp_masks(torch.zeros((1, height, width), dtype=torch.bool, device=DEVICE), np.eye(3))
    torch.cuda.synchronize()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    warm_s = time.time() - t_warm
    print(f"ready: load {load_s:.1f}s, warm-up {warm_s:.1f}s; {VIDEO} {width}x{height} {limit} frames, ad {ad_stem} "
          f"(aspect {renderer.aspect:.1f}), detect every {DETECT_EVERY}", flush=True)

    out_path = os.path.join(OUTPUT_DIR, f"{stem}_{ad_stem}.mp4")
    writer = AudioFFmpegWriter(out_path, width, height, fps, VIDEO) if KEEP_AUDIO else H.FFmpegWriter(out_path, width, height, fps)
    reader = H.FrameReader(cap, limit, True)
    sink = OrderedSink(writer)
    tracker, timer = H.Tracker(), H.StageTimer()
    prev_hist = None
    cuts, sheet = [], []
    t_start = time.time()
    fi = 0
    while fi < limit:
        with timer.stage("decode"):
            frame = reader.read()
        if frame is None:
            break
        with timer.stage("shotcut"):
            hist = H.hist_of(frame)
            cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > H.CUT_HIST_DIST
            prev_hist = hist
            if cut:
                tracker.reset()
                cuts.append(fi)
        with timer.stage("upload"):
            frame_gpu = torch.from_numpy(frame).to(DEVICE)
        with timer.stage("flow"):
            Hm = None if cut else motion.update(frame_gpu)
        with timer.stage("propagate"):
            if Hm is not None and tracker.tracks:
                tracker.propagate(Hm)
        if fi % DETECT_EVERY == 0 or cut:
            with timer.stage("backbone"):
                state = detector.backbone(frame_gpu)
            with timer.stage("decoder"):
                logits, probs, boxes = detector.detect(state)
                low_hw = tuple(logits.shape[-2:])
            with timer.stage("dedupe"):
                logits, probs, boxes, low = H.clean_detections(logits, probs, boxes, frame_pixels)
            with timer.stage("upsample"):
                if logits.shape[0]:
                    full = F.interpolate(logits.unsqueeze(1), (height, width), mode="bilinear", align_corners=False)[:, 0] > 0
                    px_boxes = (boxes * box_scale).round().long().cpu().numpy()
                else:
                    full = torch.zeros((0, height, width), dtype=torch.bool, device=DEVICE)
                    px_boxes = np.zeros((0, 4), np.int64)
            with timer.stage("track"):
                tracked = tracker.update(low, full, px_boxes, probs.cpu().numpy(), low_hw)
        else:
            tracked = tracker.current()

        with timer.stage("render"):
            out = frame_gpu.float()
            for t, _state in tracked:
                renderer.paint(out, t)
            img = out.clamp(0, 255).to(torch.uint8).cpu().numpy()
        with timer.stage("encode"):
            sink.put(img)
        timer.frames += 1
        if fi % SHEET_EVERY == 0:
            sheet.append(np.hstack([cv2.resize(frame, (640, 360)), cv2.resize(img, (640, 360))]))
        if fi % 50 == 0:
            print(f"frame {fi}: {len(tracked)} boards{'  CUT' if cut else ''}", flush=True)
        fi += 1

    sink.close()
    cap.release()
    elapsed = time.time() - t_start
    summary = {
        "video": VIDEO, "ad": AD_IMAGE, "frames": fi, "detect_every": DETECT_EVERY, "fp8": H.FP8,
        "elapsed_s": round(elapsed, 1), "fps": round(fi / elapsed, 2), "video_fps": fps,
        "stage_ms": timer.ms_per_frame(), "boards_painted": renderer.boards,
        "boards_per_frame": round(renderer.boards / max(fi, 1), 2), "shot_cuts": cuts,
        "unique_ids": tracker.next_id, "mean_match_iou": round(float(np.mean(tracker.match_ious)), 3) if tracker.match_ious else None,
        "homography_failures": motion.failures, "output": out_path,
    }
    with open(os.path.join(OUTPUT_DIR, f"{stem}_{ad_stem}_stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if sheet:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{stem}_{ad_stem}_sheet.jpg"), np.vstack(sheet[:8]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps(summary, indent=2))
    print("ms per frame: " + ", ".join(f"{k} {v}" for k, v in summary["stage_ms"].items()))


if __name__ == "__main__":
    main()
