"""AdSwapAI R&D, 2026-09-02: sweep text prompts (and a zoom crop) for SAM3 board detection.

For every clip x timestamp the backbone runs once, then each prompt is queried
(about 55 ms each). Optionally the same frame is also processed as a zoomed
crop (the horizontal band where the boards are) so thin far-away strips get
more pixels. Outputs: one contact sheet per prompt (all frames side by side),
a JSON summary, and a ranking table printed at the end.
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------------------------------- settings
VIDEOS = ["data/1.mp4", "data/2.mp4", "data/3.mp4"]
TIMESTAMPS = [1.0, 4.0, 8.0, 11.0]
PROMPTS = [
    "billboard", "advertisement", "advertising banner", "sponsor banner", "banner",
    "advertising hoarding", "perimeter advertising", "sign board", "LED screen",
    "advertising sign", "sponsor logo board", "pitch side advertising",
]
CONFIDENCE = 0.2                 # low on purpose: we look at the score distribution
ZOOM_BAND = (0.05, 0.60)         # (top, bottom) fraction of the frame height used for the zoom crop; None = off
OUTPUT_DIR = "output/prompt_sweep"
# --------------------------------------------------------------------------------------

PALETTE = [
    (0, 200, 255), (0, 255, 120), (255, 80, 80), (255, 200, 0), (200, 0, 255),
    (0, 120, 255), (120, 255, 0), (255, 0, 180), (80, 255, 255), (255, 140, 0),
]


def read_frame(video_path, t):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def to_np(x):
    return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def run_prompt(processor, state, prompt):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = processor.set_text_prompt(prompt=prompt, state=state)
    masks = to_np(out["masks"])
    boxes = to_np(out["boxes"]).reshape(-1, 4)
    scores = to_np(out["scores"]).reshape(-1)
    if masks.size == 0:
        masks = np.zeros((0, 1, 1), bool)
    masks = masks.reshape(-1, *masks.shape[-2:]) > 0.5
    processor.reset_all_prompts(state)
    return masks, boxes, scores


def overlay(frame, masks, boxes, scores, title):
    out = frame.copy()
    h, w = out.shape[:2]
    for i, (m, b, s) in enumerate(zip(masks, boxes, scores)):
        c = PALETTE[i % len(PALETTE)]
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[m] = (out[m] * 0.45 + np.array(c) * 0.55).astype(np.uint8)
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 2)
        cv2.putText(out, f"{s:.2f}", (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(out, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def sheet(images, cols, cell_w=480):
    cells = [cv2.resize(im, (cell_w, int(im.shape[0] * cell_w / im.shape[1]))) for im in images]
    ch = max(c.shape[0] for c in cells)
    rows = (len(cells) + cols - 1) // cols
    canvas = np.zeros((rows * ch, cols * cell_w, 3), np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        canvas[r * ch:r * ch + c.shape[0], k * cell_w:k * cell_w + c.shape[1]] = c
    return canvas


def main():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE)

    frames = []                                  # (clip, t, frame)
    for video in VIDEOS:
        for t in TIMESTAMPS:
            f = read_frame(video, t)
            if f is not None:
                frames.append((Path(video).stem, t, f))
    print(f"{len(frames)} frames, {len(PROMPTS)} prompts, zoom={'on' if ZOOM_BAND else 'off'}")

    per_prompt_images = {p: [] for p in PROMPTS}
    per_prompt_zoom = {p: [] for p in PROMPTS}
    records = []
    for clip, t, frame in frames:
        h, w = frame.shape[:2]
        variants = [("full", frame, 0)]
        if ZOOM_BAND:
            y0, y1 = int(ZOOM_BAND[0] * h), int(ZOOM_BAND[1] * h)
            variants.append(("zoom", frame[y0:y1].copy(), y0))
        for variant, img, y_off in variants:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            for prompt in PROMPTS:
                tic = time.time()
                masks, boxes, scores = run_prompt(processor, state, prompt)
                ms = (time.time() - tic) * 1000
                # map zoom results back into the full frame for the overlay
                if variant == "zoom":
                    full_masks = np.zeros((len(masks), h, w), bool)
                    for i, m in enumerate(masks):
                        mm = m if m.shape == img.shape[:2] else cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                        full_masks[i, y_off:y_off + mm.shape[0]] = mm
                    masks = full_masks
                    boxes = boxes + np.array([0, y_off, 0, y_off])
                title = f"{clip} {t:g}s {variant} '{prompt}' n={len(scores)}"
                img_out = overlay(frame, masks, boxes, scores, title)
                (per_prompt_images if variant == "full" else per_prompt_zoom)[prompt].append(img_out)
                records.append({
                    "clip": clip, "t": t, "variant": variant, "prompt": prompt, "count": int(len(scores)),
                    "scores": [round(float(s), 3) for s in scores],
                    "area_pct": [round(float(m.sum()) * 100 / m.size, 2) for m in masks], "ms": round(ms),
                })
                print(f"{title:55s} scores={[round(float(s), 2) for s in scores]}")

    for prompt in PROMPTS:
        tag = prompt.replace(" ", "_")
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"full_{tag}.jpg"), sheet(per_prompt_images[prompt], cols=4), [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ZOOM_BAND:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"zoom_{tag}.jpg"), sheet(per_prompt_zoom[prompt], cols=4), [cv2.IMWRITE_JPEG_QUALITY, 78])
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # ranking: frames with at least one hit, mean score, mean count, huge-mask rate (area > 25 % = probably crowd/pitch)
    print("\nprompt ranking (per variant):")
    print(f"{'variant':6s} {'prompt':26s} {'frames_hit':>10s} {'mean_n':>7s} {'mean_score':>10s} {'huge_masks':>10s}")
    for variant in ("full", "zoom") if ZOOM_BAND else ("full",):
        rows = []
        for prompt in PROMPTS:
            rs = [r for r in records if r["prompt"] == prompt and r["variant"] == variant]
            hit = sum(1 for r in rs if r["count"] > 0)
            allscores = [s for r in rs for s in r["scores"]]
            huge = sum(1 for r in rs for a in r["area_pct"] if a > 25)
            rows.append((prompt, hit, np.mean([r["count"] for r in rs]), np.mean(allscores) if allscores else 0, huge))
        for prompt, hit, mean_n, mean_s, huge in sorted(rows, key=lambda x: (-x[1], -x[3])):
            print(f"{variant:6s} {prompt:26s} {hit:>6d}/{len(frames):<3d} {mean_n:7.1f} {mean_s:10.2f} {huge:10d}")
    print(f"done: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
