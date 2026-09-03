"""AdSwapAI R&D, 2026-09-03: where does the time go in the hybrid tracker, and what speeds it up?

The hybrid tracker (sam3_hybrid_track.py) runs at about 3.4 fps. This script
times every stage of that loop per frame (decode, shot-cut check, SAM3 backbone,
prompt/decoder, GPU->CPU copy, de-duplication, tracking, drawing, encoding) and
then repeats the run with a set of speed-ups so each one can be judged on its
own numbers. Same clip, same prompt, same thresholds as the hybrid tracker.

Variants:
  baseline      the hybrid loop exactly as written (masks copied to the CPU as
                float32 at full resolution, numpy de-duplication and tracking,
                per-track numpy drawing)
  fast          text embedding computed once, de-duplication and IoU tracking on
                the GPU at the decoder's low mask resolution, only the surviving
                masks up-sampled to full resolution, overlay blended on the GPU
  fast_res768   fast + SAM3 input 768 px instead of 1008
  fast_res640   fast + SAM3 input 640 px
  fast_every2   fast + SAM3 on every 2nd frame (tracks are held in between)
  fast_compile  fast + torch.compile on the vision encoder / decoder

Output: output/speed_probe/summary.json, one diagnostic video per variant and a
printed table (fps, ms per stage, detections per frame, unique ids).
"""
import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ----------------------------------------------------------------------------- settings
VIDEO = "data/2.mp4"
PROMPT = "sponsor banner"
CONFIDENCE = 0.35
MAX_FRAMES = 150
WARMUP_FRAMES = 3            # first frames run but are not timed (cudnn autotune, allocator)
VARIANTS = ["baseline", "fast", "fast_res768", "fast_res640", "fast_every2", "fast_compile"]
DEDUPE_IOU = 0.6
CONTAIN_RATIO = 0.8
HUD_TOP_FRACTION = 0.08
MIN_AREA_PX = 400
MATCH_IOU = 0.3
HOLD_FRAMES = 5
CUT_HIST_DIST = 0.5
OUTPUT_DIR = "output/speed_probe"
# --------------------------------------------------------------------------------------

PALETTE = [
    (0, 200, 255), (0, 255, 120), (255, 80, 80), (255, 200, 0), (200, 0, 255),
    (0, 120, 255), (120, 255, 0), (255, 0, 180), (80, 255, 255), (255, 140, 0),
    (160, 160, 255), (255, 255, 120), (120, 255, 200), (255, 120, 200), (200, 255, 0),
]


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


def hist_of(frame):
    hsv = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def hud_and_text(out, tracks, fi, ms, cut, n_raw):
    """Boxes, ids and the status bar; ``tracks`` = list of (id, box, score, held)."""
    h, w = out.shape[:2]
    for tid, box, score, held in tracks:
        c = PALETTE[tid % len(PALETTE)]
        x0, y0, x1, y1 = box
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 1 if held else 2)
        label = f"id{tid}{'*' if held else ''} {score:.2f}"
        cv2.putText(out, label, (x0, max(y0 - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    status = f"frame {fi}  raw {n_raw} -> tracks {len(tracks)}  {ms:.0f} ms" + ("  SHOT CUT" if cut else "")
    cv2.putText(out, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ============================================================================ baseline
# Copied from sam3_hybrid_track.py so the timing is of the code as it is.
def to_np(x):
    return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def bbox_of(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])) if rows.size else None


def boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def clean_detections(masks, scores, frame_h):
    keep = []
    for m, s in zip(masks, scores):
        area = int(m.sum())
        if area < MIN_AREA_PX:
            continue
        box = bbox_of(m)
        if box is None or box[3] < HUD_TOP_FRACTION * frame_h:
            continue
        keep.append((float(s), m, box, area))
    keep.sort(key=lambda k: -k[0])
    result = []
    for s, m, box, area in keep:
        duplicate = False
        for s2, m2, box2, area2 in result:
            if not boxes_overlap(box, box2):
                continue
            inter = np.logical_and(m, m2).sum()
            if inter == 0:
                continue
            iou = inter / (area + area2 - inter)
            if iou > DEDUPE_IOU or inter / area > CONTAIN_RATIO:
                duplicate = True
                break
        if not duplicate:
            result.append((s, m, box, area))
    return result


class Track:
    def __init__(self, tid, mask, box, score):
        self.id, self.mask, self.box, self.score, self.misses, self.age = tid, mask, box, score, 0, 1


class Tracker:
    def __init__(self):
        self.tracks, self.next_id = [], 0

    def reset(self):
        self.tracks = []

    def update(self, dets):
        matched = set()
        out = []
        for s, m, box, area in dets:
            best, best_iou = None, MATCH_IOU
            for i, t in enumerate(self.tracks):
                if i in matched or not boxes_overlap(t.box, box):
                    continue
                iou = mask_iou(t.mask, m)
                if iou > best_iou:
                    best, best_iou = i, iou
            if best is None:
                self.tracks.append(Track(self.next_id, m, box, s))
                self.next_id += 1
                matched.add(len(self.tracks) - 1)
            else:
                t = self.tracks[best]
                t.mask, t.box, t.score, t.misses = m, box, s, 0
                t.age += 1
                matched.add(best)
        survivors = []
        for i, t in enumerate(self.tracks):
            if i in matched:
                survivors.append(t)
                out.append((t, False))
            else:
                t.misses += 1
                if t.misses <= HOLD_FRAMES:
                    survivors.append(t)
                    out.append((t, True))
        self.tracks = survivors
        return out


def draw_numpy(frame, tracked):
    out = frame.copy()
    for t, held in tracked:
        c = PALETTE[t.id % len(PALETTE)]
        out[t.mask] = (out[t.mask] * 0.5 + np.array(c) * 0.5).astype(np.uint8)
    return out


def run_baseline(processor, cap, writer, limit, timer, height, width):
    tracker, prev_hist, id_counter = Tracker(), None, Counter()
    frames = raw_total = kept_total = 0
    while frames < limit:
        with timer.stage("decode"):
            ok, frame = cap.read()
        if not ok:
            break
        tic = time.perf_counter()
        with timer.stage("shotcut"):
            h = hist_of(frame)
            cut = prev_hist is not None and cv2.compareHist(prev_hist, h, cv2.HISTCMP_BHATTACHARYYA) > CUT_HIST_DIST
            prev_hist = h
            if cut:
                tracker.reset()
        with timer.stage("backbone"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        with timer.stage("prompt+decoder"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = processor.set_text_prompt(prompt=PROMPT, state=state)
        with timer.stage("to_cpu"):
            masks = to_np(out["masks"])
            scores = to_np(out["scores"]).reshape(-1)
            masks = masks.reshape(-1, *masks.shape[-2:]) > 0.5 if masks.size else np.zeros((0, height, width), bool)
        with timer.stage("dedupe"):
            dets = clean_detections(masks, scores, height)
        with timer.stage("track"):
            tracked = tracker.update(dets)
        ms = (time.perf_counter() - tic) * 1000
        with timer.stage("draw"):
            img = draw_numpy(frame, tracked)
            img = hud_and_text(img, [(t.id, t.box, t.score, held) for t, held in tracked], frames, ms, cut, len(scores))
        with timer.stage("encode"):
            writer.write(img)
        id_counter.update(t.id for t, _ in tracked)
        raw_total += len(scores)
        kept_total += len(dets)
        frames += 1
        if frames == WARMUP_FRAMES:
            timer.totals.clear()
        if frames > WARMUP_FRAMES:
            timer.frames += 1
        if frames % 25 == 0:
            print(f"  frame {frames}: raw {len(scores)} -> {len(dets)} -> {len(tracked)} tracks  {ms:.0f} ms", flush=True)
    return frames, id_counter, raw_total, kept_total


# ============================================================================ fast path
class FastDetector:
    """SAM3 grounding with a cached text embedding; returns low-resolution mask logits."""

    def __init__(self, processor, prompt):
        from sam3.model import box_ops
        self.p = processor
        self.box_ops = box_ops
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.text = processor.model.backbone.forward_text([prompt], device=processor.device)

    @torch.inference_mode()
    def backbone(self, frame_bgr_gpu):
        rgb = frame_bgr_gpu.flip(-1).permute(2, 0, 1).contiguous()      # HWC BGR -> CHW RGB, on the GPU
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.p.set_image(rgb)

    @torch.inference_mode()
    def detect(self, state):
        state["backbone_out"].update(self.text)
        state["geometric_prompt"] = self.p.model._get_dummy_prompt()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.p.model.forward_grounding(backbone_out=state["backbone_out"], find_input=self.p.find_stage,
                                                 geometric_prompt=state["geometric_prompt"], find_target=None)
        probs = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
        keep = probs > self.p.confidence_threshold
        probs = probs[keep].float()
        logits = out["pred_masks"][keep].float()                         # (K, h, w) low resolution
        boxes = self.box_ops.box_cxcywh_to_xyxy(out["pred_boxes"][keep].float())   # normalised xyxy
        return logits, probs, boxes


def dedupe_gpu(logits, probs, boxes, full_pixels):
    """Same rules as clean_detections, computed on low-res masks with one matmul."""
    if logits.shape[0] == 0:
        return logits, probs, boxes, logits.flatten(1)
    low = (logits > 0).flatten(1).float()                                # (K, P); sigmoid > 0.5 <=> logit > 0
    area = low.sum(1)
    scale = full_pixels / low.shape[1]
    ok = (area * scale >= MIN_AREA_PX) & (boxes[:, 3] >= HUD_TOP_FRACTION)
    low, area, probs, logits, boxes = low[ok], area[ok], probs[ok], logits[ok], boxes[ok]
    if low.shape[0] == 0:
        return logits, probs, boxes, low
    order = torch.argsort(probs, descending=True)
    low, area, probs, logits, boxes = low[order], area[order], probs[order], logits[order], boxes[order]
    inter = low @ low.T
    iou = inter / (area[:, None] + area[None, :] - inter).clamp(min=1)
    contain = inter / area[:, None].clamp(min=1)                         # fraction of i inside j
    iou_c, contain_c = iou.cpu().numpy(), contain.cpu().numpy()
    kept = []
    for i in range(low.shape[0]):
        if all(iou_c[i, j] <= DEDUPE_IOU and contain_c[i, j] <= CONTAIN_RATIO for j in kept):
            kept.append(i)
    idx = torch.tensor(kept, device=low.device, dtype=torch.long)
    return logits[idx], probs[idx], boxes[idx], low[idx]


class FastTracker:
    """IoU association on the GPU; each track keeps its low-res vector and its full-res mask."""

    def __init__(self):
        self.tracks, self.next_id = [], 0        # track = dict(id, low, full, box, score, misses)

    def reset(self):
        self.tracks = []

    def update(self, low, full, boxes, scores):
        n_det, n_trk = low.shape[0], len(self.tracks)
        if n_det and n_trk:
            trk = torch.stack([t["low"] for t in self.tracks])
            inter = low @ trk.T
            iou = (inter / (low.sum(1)[:, None] + trk.sum(1)[None, :] - inter).clamp(min=1)).cpu().numpy()
        else:
            iou = np.zeros((n_det, n_trk), np.float32)
        matched, out = set(), []
        for d in range(n_det):
            best, best_iou = None, MATCH_IOU
            for i in range(n_trk):
                if i not in matched and iou[d, i] > best_iou:
                    best, best_iou = i, iou[d, i]
            rec = {"low": low[d], "full": full[d], "box": boxes[d], "score": float(scores[d]), "misses": 0}
            if best is None:
                rec["id"] = self.next_id
                self.next_id += 1
                self.tracks.append(rec)
                matched.add(len(self.tracks) - 1)
            else:
                rec["id"] = self.tracks[best]["id"]
                self.tracks[best] = rec
                matched.add(best)
        survivors = []
        for i, t in enumerate(self.tracks):
            if i in matched:
                survivors.append(t)
                out.append((t, False))
            else:
                t["misses"] += 1
                if t["misses"] <= HOLD_FRAMES:
                    survivors.append(t)
                    out.append((t, True))
        self.tracks = survivors
        return out


def draw_gpu(frame_gpu, tracked):
    out = frame_gpu.float()
    for t, _held in tracked:
        c = torch.tensor(PALETTE[t["id"] % len(PALETTE)], device=out.device, dtype=torch.float32)
        m = t["full"]
        out[m] = out[m] * 0.5 + c * 0.5
    return out.to(torch.uint8).cpu().numpy()


def run_fast(processor, cap, writer, limit, timer, height, width, detect_every=1):
    det = FastDetector(processor, PROMPT)
    tracker, prev_hist, id_counter = FastTracker(), None, Counter()
    frames = raw_total = kept_total = 0
    full_pixels = float(height * width)
    scale = torch.tensor([width, height, width, height], device="cuda", dtype=torch.float32)
    empty_low = torch.zeros((0, 1), device="cuda")
    while frames < limit:
        with timer.stage("decode"):
            ok, frame = cap.read()
        if not ok:
            break
        tic = time.perf_counter()
        with timer.stage("shotcut"):
            h = hist_of(frame)
            cut = prev_hist is not None and cv2.compareHist(prev_hist, h, cv2.HISTCMP_BHATTACHARYYA) > CUT_HIST_DIST
            prev_hist = h
            if cut:
                tracker.reset()
        with timer.stage("upload"):
            frame_gpu = torch.from_numpy(frame).cuda()
        n_raw, n_kept = -1, 0
        if frames % detect_every == 0 or cut:
            with timer.stage("backbone"):
                state = det.backbone(frame_gpu)
            with timer.stage("prompt+decoder"):
                logits, probs, boxes = det.detect(state)
                n_raw = int(logits.shape[0])
            with timer.stage("dedupe"):
                logits, probs, boxes, low = dedupe_gpu(logits, probs, boxes, full_pixels)
                n_kept = int(logits.shape[0])
            with timer.stage("upsample"):
                if n_kept:
                    full = F.interpolate(logits.unsqueeze(1), (height, width), mode="bilinear", align_corners=False)[:, 0] > 0
                    px_boxes = (boxes * scale).round().long().cpu().numpy()
                else:
                    full = torch.zeros((0, height, width), dtype=torch.bool, device="cuda")
                    px_boxes = np.zeros((0, 4), np.int64)
            with timer.stage("track"):
                tracked = tracker.update(low, full, px_boxes, probs.cpu().numpy())
        else:
            with timer.stage("track"):
                tracked = tracker.update(empty_low, None, np.zeros((0, 4), np.int64), np.zeros(0))
        ms = (time.perf_counter() - tic) * 1000
        with timer.stage("draw"):
            img = draw_gpu(frame_gpu, tracked)
            img = hud_and_text(img, [(t["id"], [int(v) for v in t["box"]], t["score"], held) for t, held in tracked],
                               frames, ms, cut, n_raw)
        with timer.stage("encode"):
            writer.write(img)
        id_counter.update(t["id"] for t, _ in tracked)
        raw_total += max(n_raw, 0)
        kept_total += n_kept
        frames += 1
        if frames == WARMUP_FRAMES:
            timer.totals.clear()
        if frames > WARMUP_FRAMES:
            timer.frames += 1
        if frames % 25 == 0:
            print(f"  frame {frames}: raw {n_raw} -> {n_kept} -> {len(tracked)} tracks  {ms:.0f} ms", flush=True)
    return frames, id_counter, raw_total, kept_total


# ============================================================================ driver
def build(compile_model):
    from sam3.model_builder import build_sam3_image_model
    t = time.time()
    model = build_sam3_image_model(compile=compile_model)
    print(f"model ready in {time.time() - t:.1f}s (compile={compile_model})", flush=True)
    return model


def run_variant(name, model):
    from sam3.model.sam3_image_processor import Sam3Processor
    resolution = 768 if "res768" in name else 640 if "res640" in name else 1008
    detect_every = 2 if "every2" in name else 1
    processor = Sam3Processor(model, resolution=resolution, confidence_threshold=CONFIDENCE)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, MAX_FRAMES) if MAX_FRAMES else total
    stem = Path(VIDEO).stem
    writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f"{stem}_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    timer = StageTimer()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        if name == "baseline":
            frames, ids, raw_total, kept_total = run_baseline(processor, cap, writer, limit, timer, height, width)
        else:
            frames, ids, raw_total, kept_total = run_fast(processor, cap, writer, limit, timer, height, width, detect_every)
    finally:
        writer.release()
        cap.release()
    elapsed = time.time() - t0
    timed = sum(timer.totals.values())
    result = {
        "variant": name, "resolution": resolution, "detect_every": detect_every, "frames": frames,
        "elapsed_s": round(elapsed, 1), "fps_overall": round(frames / elapsed, 2),
        "fps_steady": round(timer.frames / timed, 2) if timed else None,   # after warm-up, timed stages only
        "ms_per_frame": round(timed * 1000 / max(timer.frames, 1), 1),
        "stage_ms": timer.ms_per_frame(),
        "raw_per_frame": round(raw_total / max(frames, 1), 1), "kept_per_frame": round(kept_total / max(frames, 1), 1),
        "unique_ids": len(ids), "avg_tracks_per_frame": round(sum(ids.values()) / max(frames, 1), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(json.dumps(result), flush=True)
    return result


STAGES = ["decode", "shotcut", "upload", "backbone", "prompt+decoder", "to_cpu", "dedupe", "upsample", "track", "draw", "encode"]


def print_table(results):
    print("\n=== summary (ms per frame after warm-up) ===")
    head = f"{'variant':<14}{'fps':>6}{'ms':>7} " + "".join(f"{s[:9]:>10}" for s in STAGES)
    print(head + f"{'raw':>6}{'kept':>6}{'ids':>5}{'VRAM':>6}")
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<14} ERROR {r['error'][:120]}")
            continue
        sm = r["stage_ms"]
        line = f"{r['variant']:<14}{r['fps_steady']:>6}{r['ms_per_frame']:>7} " + "".join(f"{sm.get(s, 0):>10}" for s in STAGES)
        print(line + f"{r['raw_per_frame']:>6}{r['kept_per_frame']:>6}{r['unique_ids']:>5}{r['peak_vram_gb']:>6}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}  clip {VIDEO}  prompt '{PROMPT}'  thr {CONFIDENCE}", flush=True)
    results = []
    model = build(compile_model=False)
    for name in VARIANTS:
        if name == "fast_compile":
            continue
        print(f"\n=== {name} ===", flush=True)
        try:
            results.append(run_variant(name, model))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} failed: {exc!r}", flush=True)
            results.append({"variant": name, "error": repr(exc)})
    if "fast_compile" in VARIANTS:
        print("\n=== fast_compile ===", flush=True)
        del model
        torch.cuda.empty_cache()
        try:
            model = build(compile_model=True)
            results.append(run_variant("fast_compile", model))
        except Exception as exc:  # noqa: BLE001
            print(f"  fast_compile failed: {exc!r}", flush=True)
            results.append({"variant": "fast_compile", "error": repr(exc)})

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print_table(results)


if __name__ == "__main__":
    main()
