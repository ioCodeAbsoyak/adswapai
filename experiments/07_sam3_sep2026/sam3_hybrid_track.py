"""AdSwapAI R&D, 2026-09-02/03: hybrid tracking = SAM3 image detection + camera-motion propagation, on the GPU.

Why: SAM3's video predictor gives stable masks but runs at about 0.2-1 fps
with many objects, while the SAM3 image model finds every board in ~85 ms.
This script runs the image model on every DETECT_EVERY-th frame (and on shot
cuts), de-duplicates overlapping masks, drops HUD false positives, links
detections across frames by mask IoU with a short hold, and resets ids on a
shot cut. Between detections every track is moved with the camera: RAFT-small
optical flow on the GPU at quarter resolution, a homography fitted to it with
RANSAC, masks warped on the GPU. Output: diagnostic video (colour per track
id), stats JSON with fps and the per-stage time split.

3 Sep 2026, what keeps the GPU busy (sam3_speed_probe.py, sam3_gpu_util_probe.py,
sam3_fp8_probe.py):
  * text embedding computed once; de-duplication and IoU association on the
    decoder's low-resolution masks with one matmul each; only surviving masks
    up-sampled; overlay blended on the GPU; H.264 encoded on NVENC (ffmpeg);
  * backbone weights bf16 (or fp8 with torchao for the ViT linears), SAM3's
    transformer encoder + decoder and RAFT under torch.compile
    "reduce-overhead" (CUDA graphs);
  * video decoding on a reader thread ahead of the model, drawing + encoding
    on a writer thread behind it.
"""
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# ----------------------------------------------------------------------------- settings
VIDEO = "data/2.mp4"
PROMPT = "sponsor banner"
CONFIDENCE = 0.35            # SAM3 score threshold
MAX_FRAMES = 150             # None = whole clip
DETECT_EVERY = 3             # run SAM3 on every N-th frame (and on shot cuts); tracks are propagated in between
PROPAGATE = "raft"           # camera-motion propagation between detections: "raft" (RAFT-small flow on the GPU +
                             # RANSAC homography, masks warped on the GPU) or None (tracks are frozen while held)
FLOW_SIZE = (480, 272)       # optical-flow resolution (w, h), multiples of 8
FLOW_ITERS = 6               # RAFT refinement iterations
FLOW_GRID = 8                # one correspondence every N flow pixels (480x272 / 8 = 2040 points)
RANSAC_PX = 1.5              # homography inlier threshold in flow pixels (x4 at frame resolution)
MIN_INLIERS = 150            # fewer inliers = no reliable camera motion; tracks stay where they are
COMPILE_FLOW = True          # RAFT under torch.compile "reduce-overhead": 15 -> 2.5 ms per frame
DEDUPE_IOU = 0.6             # masks overlapping more than this are merged (higher score wins)
CONTAIN_RATIO = 0.8          # a mask lying 80 % inside a bigger one is a duplicate part -> dropped
HUD_TOP_FRACTION = 0.08      # masks entirely inside the top 8 % of the frame are HUD, dropped
HUD_CORNER = (0.78, 0.16)    # masks entirely inside the top-right corner (x >= 78 %, y <= 16 %) are HUD too:
                             # the broadcaster logo sits below the top band and was replaced with the ad
MIN_AREA_PX = 400            # tiny fragments dropped (area at frame resolution)
MATCH_IOU = 0.3              # association threshold between a propagated track and a new detection
HOLD_FRAMES = 5              # keep a track alive this many detection rounds without a match
CUT_HIST_DIST = 0.5          # HSV histogram distance above this = shot cut (Bhattacharyya)
ENCODER = "nvenc"            # "nvenc" (ffmpeg h264_nvenc, GPU), "x264" (ffmpeg libx264) or "opencv" (mp4v)
BF16_WEIGHTS = True          # convert the vision-language backbone to bf16 once (removes per-frame weight casts)
COMPILE_BACKBONE = True      # torch.compile the ViT + pixel decoder at build time: about -6 ms/frame, 30-60 s first compile
FP8 = True                   # torchao float8 (per-tensor dynamic scaling) on the 128 ViT trunk Linear layers; needs
                             # COMPILE_BACKBONE (eager fp8 is slower than bf16). Backbone 90 -> 59 ms; masks differ
                             # from bf16 by ~3 % IoU, ~2 % of kept queries change (sam3_fp8_probe.py)
COMPILE_DECODER = "reduce-overhead"   # transformer encoder + decoder under CUDA graphs (48 -> 25 ms/frame); None = off
ATTENTION = "cudnn"          # SDPA kernel priority for the backbone: "cudnn" (1.3 ms per global block on the 5070 Ti,
                             # torch's default picks "efficient" at 2.5 ms; flash is unavailable on sm_120) or "default"
PIPELINE_THREADS = True      # reader thread ahead of the model, writer thread behind it
WARMUP_FRAMES = 3            # model runs on the first frame this many times before the timed loop (compile, allocator)
OUTPUT_DIR = "output/hybrid_track"
# --------------------------------------------------------------------------------------

PALETTE = [
    (0, 200, 255), (0, 255, 120), (255, 80, 80), (255, 200, 0), (200, 0, 255),
    (0, 120, 255), (120, 255, 0), (255, 0, 180), (80, 255, 255), (255, 140, 0),
    (160, 160, 255), (255, 255, 120), (120, 255, 200), (255, 120, 200), (200, 255, 0),
]
DEVICE = "cuda"
SDPA_PRIORITY = ([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH] if ATTENTION == "cudnn"
                 else [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH])


# ----------------------------------------------------------------------------- timing
class StageTimer:
    """Wall-clock seconds per stage; the CUDA stream is synchronised at both ends."""

    def __init__(self):
        self.totals, self.frames = {}, 0

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t

    def ms_per_frame(self):
        return {k: round(v * 1000 / max(self.frames, 1), 1) for k, v in self.totals.items()}


# ----------------------------------------------------------------------------- model setup
def to_bf16(module):
    """Convert float parameters and buffers to bf16 in place; complex (RoPE) and integer buffers stay."""
    n = 0
    for p in module.parameters():
        if p.is_floating_point() and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
            n += 1
    for m in module.modules():
        for name, b in list(m._buffers.items()):
            if b is not None and b.is_floating_point() and b.dtype != torch.bfloat16:
                m._buffers[name] = b.to(torch.bfloat16)
                n += 1
    return n


def patch_fused_mlp():
    """SAM3's ViT MLP calls aten._addmm_activation on fc1's raw weight (fused linear + GELU); that bypasses the
    module and is not implemented for torchao's Float8Tensor. With fp8 weights use linear + GELU instead."""
    import sam3.model.vitdet as vitdet
    original = vitdet.addmm_act

    def addmm_act_fp8_aware(activation, linear, mat1):
        if "Float8" in type(linear.weight).__name__:
            act = activation() if isinstance(activation, type) else activation
            return act(linear(mat1.to(torch.bfloat16)))
        return original(activation, linear, mat1)

    vitdet.addmm_act = addmm_act_fp8_aware


def quantize_fp8(vision_backbone):
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor, quantize_
    patch_fused_mlp()
    cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor(), set_inductor_config=False)
    quantize_(vision_backbone, cfg, filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and "trunk.blocks" in fqn)
    n = 0
    for _, m in vision_backbone.named_modules():
        if isinstance(m, torch.nn.Linear) and "Float8" in type(m.weight).__name__:
            # autocast leaves LayerNorm output in fp32; the fp8 linear wants bf16 in (and gives bf16 out)
            m.register_forward_pre_hook(lambda mod, args: (args[0].to(torch.bfloat16),) + tuple(args[1:]))
            n += 1
    return n


def build_model():
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(compile=COMPILE_BACKBONE)   # the ViT compiles lazily at its first forward
    if BF16_WEIGHTS or FP8:
        to_bf16(model.backbone)                       # the grounding decoder keeps fp32 parts on purpose
    if FP8:
        print(f"fp8: {quantize_fp8(model.backbone.vision_backbone)} ViT Linear layers quantised", flush=True)
    if COMPILE_DECODER:
        dec, enc = model.transformer.decoder, model.transformer.encoder
        dec.compile_mode = COMPILE_DECODER            # the decoder compiles itself at the end of its first forward
        enc.forward = torch.compile(enc.forward, mode=COMPILE_DECODER, fullgraph=True)
    return model


class Sam3Detector:
    """SAM3 grounding with a cached text embedding. Returns low-resolution mask logits, scores, boxes."""

    def __init__(self, processor, prompt):
        from sam3.model import box_ops
        self.p = processor
        self.box_ops = box_ops
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.text = processor.model.backbone.forward_text([prompt], device=processor.device)
        self.prompt = processor.model._get_dummy_prompt()

    @torch.no_grad()                                  # not inference_mode: torchao's fp8 tensors reject it
    def backbone(self, frame_bgr_gpu):
        rgb = frame_bgr_gpu.flip(-1).permute(2, 0, 1).contiguous()      # HWC BGR -> CHW RGB, on the GPU
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), sdpa_kernel(SDPA_PRIORITY, set_priority=True):
            return self.p.set_image(rgb)                                # resize + normalise + backbone

    @torch.no_grad()
    def detect(self, state):
        if COMPILE_DECODER == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()                   # previous graph outputs may be reused
        state["backbone_out"].update(self.text)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.p.model.forward_grounding(backbone_out=state["backbone_out"], find_input=self.p.find_stage,
                                                 geometric_prompt=self.prompt, find_target=None)
        probs = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
        keep = probs > self.p.confidence_threshold
        probs = probs[keep].float()
        logits = out["pred_masks"][keep].float()                         # (K, h, w) decoder resolution
        boxes = self.box_ops.box_cxcywh_to_xyxy(out["pred_boxes"][keep].float())   # normalised xyxy
        return logits, probs, boxes


def clean_detections(logits, probs, boxes, frame_pixels):
    """Drop HUD / tiny masks, then remove duplicates and contained parts (higher score wins).

    Works on the low-resolution masks: pairwise intersections are one matmul.
    Returns (logits, probs, boxes, low) with ``low`` = flattened binary masks (K, P).
    """
    if logits.shape[0] == 0:
        return logits, probs, boxes, logits.flatten(1)
    low = (logits > 0).flatten(1).float()                                # sigmoid > 0.5  <=>  logit > 0
    area = low.sum(1)
    corner = (boxes[:, 0] >= HUD_CORNER[0]) & (boxes[:, 3] <= HUD_CORNER[1])
    ok = (area * (frame_pixels / low.shape[1]) >= MIN_AREA_PX) & (boxes[:, 3] >= HUD_TOP_FRACTION) & ~corner
    low, area, probs, logits, boxes = low[ok], area[ok], probs[ok], logits[ok], boxes[ok]
    if low.shape[0] == 0:
        return logits, probs, boxes, low
    order = torch.argsort(probs, descending=True)
    low, area, probs, logits, boxes = low[order], area[order], probs[order], logits[order], boxes[order]
    inter = low @ low.T
    iou = (inter / (area[:, None] + area[None, :] - inter).clamp(min=1)).cpu().numpy()
    contain = (inter / area[:, None].clamp(min=1)).cpu().numpy()        # fraction of i inside j
    kept = []
    for i in range(low.shape[0]):
        if all(iou[i, j] <= DEDUPE_IOU and contain[i, j] <= CONTAIN_RATIO for j in kept):
            kept.append(i)
    idx = torch.tensor(kept, device=low.device, dtype=torch.long)
    return logits[idx], probs[idx], boxes[idx], low[idx]


# ----------------------------------------------------------------------------- camera motion
class CameraMotion:
    """Frame-to-frame camera homography: RAFT-small optical flow on the GPU at FLOW_SIZE, sampled on a grid,
    fitted with RANSAC (cv2, ~1.5 ms for 2 000 points). Returns the homography in frame pixels."""

    def __init__(self, width, height):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(DEVICE).eval()
        fw, fh = FLOW_SIZE
        self.size = (fh, fw)
        S = np.diag([width / fw, height / fh, 1.0])
        self.S, self.Sinv = S, np.linalg.inv(S)
        ys, xs = torch.meshgrid(torch.arange(FLOW_GRID // 2, fh, FLOW_GRID), torch.arange(FLOW_GRID // 2, fw, FLOW_GRID), indexing="ij")
        self.iy, self.ix = ys.reshape(-1).to(DEVICE), xs.reshape(-1).to(DEVICE)
        self.pts = torch.stack([xs, ys], -1).reshape(-1, 2).float().to(DEVICE)   # (N, 2) x, y in flow pixels
        self.prev = None
        raw = lambda a, b: self.model(a, b, num_flow_updates=FLOW_ITERS)[-1]  # noqa: E731
        self.flow_fn = torch.compile(raw, mode="reduce-overhead") if COMPILE_FLOW else raw
        self.inlier_ratios, self.failures = [], 0

    def _prep(self, frame_gpu):
        x = frame_gpu.flip(-1).permute(2, 0, 1).float().unsqueeze(0)          # BGR -> RGB, (1,3,H,W)
        x = F.interpolate(x, self.size, mode="bilinear", align_corners=False, antialias=True)
        return x / 127.5 - 1.0

    def reset(self):
        self.prev = None

    @torch.no_grad()
    def update(self, frame_gpu):
        """Homography mapping previous-frame pixels to this frame, or None (first frame / unreliable fit)."""
        cur = self._prep(frame_gpu)
        prev, self.prev = self.prev, cur
        if prev is None:
            return None
        if COMPILE_FLOW:
            torch.compiler.cudagraph_mark_step_begin()
        flow = self.flow_fn(prev, cur)[0]                                     # (2, fh, fw), prev -> cur
        d = flow[:, self.iy, self.ix].T                                       # (N, 2)
        p1 = self.pts.cpu().numpy()
        p2 = (self.pts + d).cpu().numpy()
        H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_PX)
        n_inl = int(inl.sum()) if inl is not None else 0
        self.inlier_ratios.append(n_inl / len(p1))
        if H is None or n_inl < MIN_INLIERS:
            self.failures += 1
            return None
        return self.S @ H @ self.Sinv


_GRID_CACHE = {}


def warp_masks(masks, H):
    """Warp bool masks (T, H, W) by the frame-pixel homography H (previous -> current) with nearest sampling."""
    T, height, width = masks.shape
    key = (height, width)
    if key not in _GRID_CACHE:
        ys, xs = torch.meshgrid(torch.arange(height, device=DEVICE, dtype=torch.float32),
                                torch.arange(width, device=DEVICE, dtype=torch.float32), indexing="ij")
        _GRID_CACHE[key] = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3)   # (H*W, 3)
    Hinv = torch.linalg.inv(torch.as_tensor(H, dtype=torch.float32, device=DEVICE))
    src = _GRID_CACHE[key] @ Hinv.T                                           # where each output pixel comes from
    src = src[:, :2] / src[:, 2:3].clamp(min=1e-6)
    grid = torch.stack([src[:, 0] / (width - 1) * 2 - 1, src[:, 1] / (height - 1) * 2 - 1], -1).reshape(1, height, width, 2)
    out = F.grid_sample(masks.float().unsqueeze(1), grid.expand(T, -1, -1, -1), mode="nearest",
                        padding_mode="zeros", align_corners=True)
    return out[:, 0] > 0.5


def boxes_of(masks):
    """Pixel boxes (T, 4) x0 y0 x1 y1 of bool masks (T, H, W); empty masks give x1 < x0."""
    rows, cols = masks.any(2), masks.any(1)
    T, height, width = masks.shape
    ys = torch.arange(height, device=masks.device)
    xs = torch.arange(width, device=masks.device)
    y0 = torch.where(rows, ys, height).amin(1)
    y1 = torch.where(rows, ys, -1).amax(1)
    x0 = torch.where(cols, xs, width).amin(1)
    x1 = torch.where(cols, xs, -1).amax(1)
    return torch.stack([x0, y0, x1, y1], 1).cpu().numpy()


# ----------------------------------------------------------------------------- tracking
class Tracker:
    """IoU association on the GPU. A track keeps its low-res vector (for IoU) and its full-res mask (for rendering);
    between detections both are moved with the camera homography."""

    def __init__(self):
        self.tracks, self.next_id = [], 0        # track = dict(id, low, full, box, score, misses, age, state)
        self.low_hw = None
        self.match_ious = []

    def reset(self):
        self.tracks = []

    def propagate(self, H):
        if not self.tracks or H is None:
            return
        full = warp_masks(torch.stack([t["full"] for t in self.tracks]), H)
        low = (F.interpolate(full.float().unsqueeze(1), self.low_hw, mode="area")[:, 0] > 0.5).flatten(1).float()
        boxes = boxes_of(full)
        Hn = np.asarray(H, dtype=np.float64)
        survivors = []
        for t, f, l, b in zip(self.tracks, full, low, boxes):
            if b[2] < b[0] or l.sum() < 1:                                    # left the frame
                continue
            t["full"], t["low"], t["box"], t["state"] = f, l, b, "~"
            if t.get("quad") is not None:                                     # board corners move with the camera too
                q = np.concatenate([t["quad"], np.ones((4, 1), np.float32)], 1) @ Hn.T
                t["quad"] = (q[:, :2] / q[:, 2:3]).astype(np.float32)
            survivors.append(t)
        self.tracks = survivors

    def current(self):
        return [(t, t["state"]) for t in self.tracks]

    def update(self, low, full, boxes, scores, low_hw):
        self.low_hw = low_hw
        n_det, n_trk = low.shape[0], len(self.tracks)
        if n_det and n_trk:
            trk = torch.stack([t["low"] for t in self.tracks])
            inter = low @ trk.T
            iou = (inter / (low.sum(1)[:, None] + trk.sum(1)[None, :] - inter).clamp(min=1)).cpu().numpy()
        else:
            iou = np.zeros((n_det, n_trk), np.float32)
        matched = set()
        for d in range(n_det):
            best, best_iou = None, MATCH_IOU
            for i in range(n_trk):
                if i not in matched and iou[d, i] > best_iou:
                    best, best_iou = i, iou[d, i]
            rec = {"low": low[d], "full": full[d], "box": boxes[d], "score": float(scores[d]), "misses": 0, "age": 1, "state": ""}
            if best is None:
                rec["id"] = self.next_id
                self.next_id += 1
                self.tracks.append(rec)
                matched.add(len(self.tracks) - 1)
            else:
                rec["id"], rec["age"] = self.tracks[best]["id"], self.tracks[best]["age"] + 1
                self.tracks[best] = rec
                matched.add(best)
                self.match_ious.append(float(best_iou))
        survivors = []
        for i, t in enumerate(self.tracks):
            if i in matched:
                survivors.append(t)
            else:
                t["misses"] += 1
                if t["misses"] <= HOLD_FRAMES:
                    t["state"] = "*"
                    survivors.append(t)
        self.tracks = survivors
        return self.current()


# ----------------------------------------------------------------------------- shot cut, drawing, encoding
def hist_of(frame):
    hsv = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def blend_gpu(frame_gpu, tracked):
    """Mask tint on the GPU; returns the frame as a CPU uint8 array."""
    out = frame_gpu.float()
    for t, _state in tracked:
        c = torch.tensor(PALETTE[t["id"] % len(PALETTE)], device=out.device, dtype=torch.float32)
        m = t["full"]
        out[m] = out[m] * 0.5 + c * 0.5
    return out.to(torch.uint8).cpu().numpy()


def draw_labels(out, labels, fi, ms, cut, n_raw):
    """Boxes, ids and the status bar (CPU). ``labels`` = list of (id, box, score, state); state "" detected,
    "*" held without a match, "~" propagated with the camera."""
    h, w = out.shape[:2]
    for tid, box, score, state in labels:
        c = PALETTE[tid % len(PALETTE)]
        x0, y0, x1, y1 = box
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 2 if state == "" else 1)
        cv2.putText(out, f"id{tid}{state} {score:.2f}", (x0, max(y0 - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    what = f"raw {n_raw} ->" if n_raw >= 0 else "propagated ->"
    status = f"frame {fi}  {what} tracks {len(labels)}  {ms:.0f} ms" + ("  SHOT CUT" if cut else "")
    cv2.putText(out, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class FFmpegWriter:
    """Raw BGR frames piped to ffmpeg; H.264 on NVENC (GPU) or libx264, web-ready mp4."""

    def __init__(self, path, width, height, fps, encoder="nvenc"):
        codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "23", "-b:v", "0"] \
            if encoder == "nvenc" else ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "pipe:0",
               *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self._stderr = tempfile.TemporaryFile()            # a full stderr pipe would block ffmpeg
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._stderr)

    def write(self, frame):
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            self.proc.wait()
            raise RuntimeError(f"ffmpeg stopped: {self._error()}")

    def _error(self):
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="ignore").strip()

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({self.proc.returncode}): {self._error()}")
        self._stderr.close()


def open_writer(path, width, height, fps):
    if ENCODER in ("nvenc", "x264") and shutil.which("ffmpeg"):
        return FFmpegWriter(path, width, height, fps, ENCODER), ENCODER
    if ENCODER != "opencv":
        print("ffmpeg not found on PATH, falling back to OpenCV mp4v")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)), "opencv"


# ----------------------------------------------------------------------------- CPU work on threads
class FrameReader:
    """Decodes frames on a thread, a few frames ahead of the consumer."""

    def __init__(self, cap, limit, threaded=True, depth=4):
        self.cap, self.limit = cap, limit
        self.q = queue.Queue(maxsize=depth) if threaded else None
        if threaded:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        n = 0
        while n < self.limit:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.q.put(frame)
            n += 1
        self.q.put(None)

    def read(self):
        if self.q is None:
            ok, frame = self.cap.read()
            return frame if ok else None
        return self.q.get()


class FrameSink:
    """Draws labels and writes frames on a thread, in order, behind the consumer."""

    def __init__(self, writer, threaded=True, depth=8):
        self.writer, self.sheet, self.error = writer, {}, None
        self.q = queue.Queue(maxsize=depth) if threaded else None
        self.thread = threading.Thread(target=self._run, daemon=True) if threaded else None
        if self.thread:
            self.thread.start()

    def _handle(self, item):
        img, labels, fi, ms, cut, n_raw = item
        img = draw_labels(img, labels, fi, ms, cut, n_raw)
        self.writer.write(img)
        if fi % 25 == 0:
            self.sheet[fi] = img

    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    return
                self._handle(item)
        except Exception as exc:  # noqa: BLE001
            self.error = exc

    def put(self, item):
        if self.error:
            raise RuntimeError(f"writer thread failed: {self.error!r}")
        if self.q is None:
            self._handle(item)
        else:
            self.q.put(item)

    def close(self):
        if self.q is not None:
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
    model = build_model()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE)
    detector = Sam3Detector(processor, PROMPT)
    load_s = time.time() - t_load

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, MAX_FRAMES) if MAX_FRAMES else total
    stem = Path(VIDEO).stem
    tag = f"{stem}_{PROMPT.replace(' ', '_')}_hybrid"
    frame_pixels = float(width * height)
    box_scale = torch.tensor([width, height, width, height], device=DEVICE, dtype=torch.float32)
    empty_low = torch.zeros((0, 1), device=DEVICE)
    motion = CameraMotion(width, height) if PROPAGATE == "raft" else None

    # warm-up on the first frame: compilation, CUDA graph capture, allocator
    t_warm = time.time()
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {VIDEO}")
    first_gpu = torch.from_numpy(first).to(DEVICE)
    for _ in range(WARMUP_FRAMES):
        logits, probs, boxes = detector.detect(detector.backbone(first_gpu))
        clean_detections(logits, probs, boxes, frame_pixels)
        if motion:
            motion.update(first_gpu)
    if motion:
        motion.reset()
        motion.inlier_ratios, motion.failures = [], 0
        warp_masks(torch.zeros((1, height, width), dtype=torch.bool, device=DEVICE), np.eye(3))
    torch.cuda.synchronize()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    warm_s = time.time() - t_warm
    print(f"model ready: load {load_s:.1f}s, warm-up {warm_s:.1f}s  ({torch.cuda.get_device_name(0)}, bf16={BF16_WEIGHTS}, "
          f"fp8={FP8}, compile backbone={COMPILE_BACKBONE}, decoder={COMPILE_DECODER}, threads={PIPELINE_THREADS}, "
          f"detect every {DETECT_EVERY}, propagate={PROPAGATE})", flush=True)

    writer, encoder = open_writer(os.path.join(OUTPUT_DIR, f"{tag}_track.mp4"), width, height, fps)
    reader = FrameReader(cap, limit, PIPELINE_THREADS)
    sink = FrameSink(writer, PIPELINE_THREADS)
    tracker, timer = Tracker(), StageTimer()
    prev_hist = None
    stats, id_counter, cuts = [], Counter(), []
    detections = propagated = 0
    t_start = time.time()
    fi = 0
    while fi < limit:
        with timer.stage("decode"):
            frame = reader.read()
        if frame is None:
            break
        tic = time.perf_counter()
        with timer.stage("shotcut"):
            h = hist_of(frame)
            cut = prev_hist is not None and cv2.compareHist(prev_hist, h, cv2.HISTCMP_BHATTACHARYYA) > CUT_HIST_DIST
            prev_hist = h
            if cut:
                tracker.reset()
                cuts.append(fi)
        with timer.stage("upload"):
            frame_gpu = torch.from_numpy(frame).to(DEVICE)
        with timer.stage("flow"):
            H = motion.update(frame_gpu) if motion else None
            if cut and motion:
                H = None
        with timer.stage("propagate"):
            if H is not None and tracker.tracks:
                tracker.propagate(H)
                propagated += 1

        n_raw, n_kept = -1, 0
        if fi % DETECT_EVERY == 0 or cut:
            detections += 1
            with timer.stage("backbone"):
                state = detector.backbone(frame_gpu)
            with timer.stage("decoder"):
                logits, probs, boxes = detector.detect(state)
                n_raw = int(logits.shape[0])
                low_hw = tuple(logits.shape[-2:])
            with timer.stage("dedupe"):
                logits, probs, boxes, low = clean_detections(logits, probs, boxes, frame_pixels)
                n_kept = int(logits.shape[0])
            with timer.stage("upsample"):
                if n_kept:
                    full = F.interpolate(logits.unsqueeze(1), (height, width), mode="bilinear", align_corners=False)[:, 0] > 0
                    px_boxes = (boxes * box_scale).round().long().cpu().numpy()
                else:
                    full = torch.zeros((0, height, width), dtype=torch.bool, device=DEVICE)
                    px_boxes = np.zeros((0, 4), np.int64)
            with timer.stage("track"):
                tracked = tracker.update(low, full, px_boxes, probs.cpu().numpy(), low_hw)
        else:
            with timer.stage("track"):
                tracked = tracker.current()
        ms = (time.perf_counter() - tic) * 1000

        with timer.stage("draw"):
            img = blend_gpu(frame_gpu, tracked)
            labels = [(t["id"], [int(v) for v in t["box"]], t["score"], state) for t, state in tracked]
        with timer.stage("encode"):
            sink.put((img, labels, fi, ms, cut, n_raw))
        timer.frames += 1

        ids = [t["id"] for t, _ in tracked]
        id_counter.update(ids)
        stats.append({"frame": fi, "raw": n_raw, "kept": n_kept, "ids": ids,
                      "held": [t["id"] for t, s in tracked if s == "*"], "ms": round(ms), "cut": bool(cut)})
        if fi % 25 == 0:
            what = f"raw {n_raw} -> {n_kept} kept ->" if n_raw >= 0 else "propagated ->"
            print(f"frame {fi}: {what} {len(tracked)} tracks {ids}  {ms:.0f} ms{'  CUT' if cut else ''}", flush=True)
        fi += 1

    sink.close()
    cap.release()
    elapsed = time.time() - t_start
    summary = {
        "video": VIDEO, "prompt": PROMPT, "confidence": CONFIDENCE, "frames": fi, "detect_every": DETECT_EVERY,
        "propagate": PROPAGATE, "encoder": encoder, "bf16_weights": BF16_WEIGHTS, "fp8": FP8,
        "compile_backbone": COMPILE_BACKBONE, "compile_decoder": COMPILE_DECODER, "attention": ATTENTION,
        "pipeline_threads": PIPELINE_THREADS,
        "load_s": round(load_s, 1), "warmup_s": round(warm_s, 1),
        "elapsed_s": round(elapsed, 1), "fps": round(fi / elapsed, 2),
        "stage_ms": timer.ms_per_frame(),
        "detection_frames": detections, "propagated_frames": propagated,
        "homography_failures": motion.failures if motion else None,
        "mean_inlier_ratio": round(float(np.mean(motion.inlier_ratios)), 3) if motion and motion.inlier_ratios else None,
        "mean_match_iou": round(float(np.mean(tracker.match_ious)), 3) if tracker.match_ious else None,
        "shot_cuts": cuts, "unique_ids": len(id_counter),
        "ids_20_frames_or_more": sum(1 for v in id_counter.values() if v >= 20),
        "frames_per_id": dict(sorted(id_counter.items())),
        "avg_tracks_per_frame": round(sum(len(s["ids"]) for s in stats) / max(fi, 1), 2),
        "median_ms": int(np.median([s["ms"] for s in stats])) if stats else 0,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    with open(os.path.join(OUTPUT_DIR, f"{tag}_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": stats}, f, indent=2)
    keys = sorted(sink.sheet)[:6]
    if keys:
        cells = [cv2.resize(sink.sheet[k], (640, int(height * 640 / width))) for k in keys]
        if len(cells) % 2:
            cells.append(np.zeros_like(cells[0]))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_sheet.jpg"),
                    np.vstack([np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(json.dumps({k: v for k, v in summary.items() if k != "frames_per_id"}, indent=2))
    print("ms per frame: " + ", ".join(f"{k} {v}" for k, v in summary["stage_ms"].items()))


if __name__ == "__main__":
    main()
