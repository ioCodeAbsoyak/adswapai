"""AdSwapAI R&D, 2026-09-02: hybrid tracking = SAM3 image detection every frame + our own association.

Why: SAM3's video predictor gives stable masks but runs at about 0.2-1 fps
with many objects, while the SAM3 image model finds every board in ~200 ms.
This script runs the image model on every frame (or every DETECT_EVERY
frames), de-duplicates overlapping masks, drops HUD false positives, links
detections across frames by mask IoU with a short hold, and resets ids on a
shot cut. Output: diagnostic video (colour per track id), stats JSON, fps.
"""
import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------------------------------- settings
VIDEO = "data/2.mp4"
PROMPT = "sponsor banner"
CONFIDENCE = 0.35            # SAM3 score threshold
MAX_FRAMES = 150             # None = whole clip
DETECT_EVERY = 1             # run SAM3 on every N-th frame (1 = every frame)
DEDUPE_IOU = 0.6             # masks overlapping more than this are merged (higher score wins)
CONTAIN_RATIO = 0.8          # a mask lying 80 % inside a bigger one is a duplicate part -> dropped
HUD_TOP_FRACTION = 0.08      # masks entirely inside the top 8 % of the frame are HUD, dropped
MIN_AREA_PX = 400            # tiny fragments dropped
MATCH_IOU = 0.3              # association threshold between frames
HOLD_FRAMES = 5              # keep a track alive this many frames without a match
CUT_HIST_DIST = 0.5          # HSV histogram distance above this = shot cut (Bhattacharyya)
OUTPUT_DIR = "output/hybrid_track"
# --------------------------------------------------------------------------------------

PALETTE = [
    (0, 200, 255), (0, 255, 120), (255, 80, 80), (255, 200, 0), (200, 0, 255),
    (0, 120, 255), (120, 255, 0), (255, 0, 180), (80, 255, 255), (255, 140, 0),
    (160, 160, 255), (255, 255, 120), (120, 255, 200), (255, 120, 200), (200, 255, 0),
]


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
    """Drop HUD / tiny masks, then remove duplicates and contained parts (higher score wins)."""
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


def hist_of(frame):
    hsv = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def draw(frame, tracked, fi, ms, cut, n_raw):
    out = frame.copy()
    h, w = out.shape[:2]
    for t, held in tracked:
        c = PALETTE[t.id % len(PALETTE)]
        out[t.mask] = (out[t.mask] * 0.5 + np.array(c) * 0.5).astype(np.uint8)
        x0, y0, x1, y1 = t.box
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 1 if held else 2)
        cv2.putText(out, f"id{t.id}{'*' if held else ''} {t.score:.2f}", (x0, max(y0 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(out, f"frame {fi}  raw {n_raw} -> tracks {len(tracked)}  {ms:.0f} ms{'  SHOT CUT' if cut else ''}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, MAX_FRAMES) if MAX_FRAMES else total
    stem = Path(VIDEO).stem
    tag = f"{stem}_{PROMPT.replace(' ', '_')}_hybrid"
    writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f"{tag}_track.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = Tracker()
    prev_hist = None
    stats, id_counter, sheet = [], Counter(), {}
    cuts = []
    t_start = time.time()
    fi = 0
    while fi < limit:
        ok, frame = cap.read()
        if not ok:
            break
        tic = time.time()
        h = hist_of(frame)
        cut = prev_hist is not None and cv2.compareHist(prev_hist, h, cv2.HISTCMP_BHATTACHARYYA) > CUT_HIST_DIST
        prev_hist = h
        if cut:
            tracker.reset()
            cuts.append(fi)

        dets = None
        if fi % DETECT_EVERY == 0 or cut:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                out = processor.set_text_prompt(prompt=PROMPT, state=state)
            masks = to_np(out["masks"])
            scores = to_np(out["scores"]).reshape(-1)
            masks = masks.reshape(-1, *masks.shape[-2:]) > 0.5 if masks.size else np.zeros((0, height, width), bool)
            n_raw = len(scores)
            dets = clean_detections(masks, scores, height)
        else:
            n_raw = -1
            dets = []
        tracked = tracker.update(dets) if dets is not None else []
        ms = (time.time() - tic) * 1000

        img = draw(frame, tracked, fi, ms, cut, n_raw)
        writer.write(img)
        ids = [t.id for t, _ in tracked]
        id_counter.update(ids)
        stats.append({"frame": fi, "raw": n_raw, "kept": len(dets), "ids": ids, "held": [t.id for t, held in tracked if held], "ms": round(ms), "cut": bool(cut)})
        if fi % 25 == 0:
            sheet[fi] = img
            print(f"frame {fi}: raw {n_raw} -> {len(dets)} kept -> {len(tracked)} tracks {ids}  {ms:.0f} ms{'  CUT' if cut else ''}")
        fi += 1

    writer.release()
    cap.release()
    elapsed = time.time() - t_start
    summary = {
        "video": VIDEO, "prompt": PROMPT, "confidence": CONFIDENCE, "frames": fi,
        "elapsed_s": round(elapsed, 1), "fps": round(fi / elapsed, 2),
        "shot_cuts": cuts, "unique_ids": len(id_counter),
        "frames_per_id": dict(sorted(id_counter.items())),
        "avg_tracks_per_frame": round(sum(len(s["ids"]) for s in stats) / max(fi, 1), 2),
        "median_ms": int(np.median([s["ms"] for s in stats])),
    }
    with open(os.path.join(OUTPUT_DIR, f"{tag}_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": stats}, f, indent=2)
    keys = sorted(sheet)[:6]
    cells = [cv2.resize(sheet[k], (640, int(height * 640 / width))) for k in keys]
    if len(cells) % 2:
        cells.append(np.zeros_like(cells[0]))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{tag}_sheet.jpg"), np.vstack([np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
