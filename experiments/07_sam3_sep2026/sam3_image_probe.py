"""AdSwapAI R&D, 2026-09-02: does SAM3 find pitch-side advertising boards from a text prompt?

Plain script, no web app. Edit the settings block, run it, look at output/image_probe/.
For every clip x timestamp x prompt it writes an overlay image (masks, boxes, scores)
and a contact sheet per clip, plus a JSON summary with the detection counts and scores.
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
VIDEOS = ["data/1.mp4", "data/2.mp4", "data/3.mp4"]   # sample clips (see docs/assets.md)
TIMESTAMPS = [1.0, 4.0, 8.0, 12.0]                    # seconds into each clip
PROMPTS = [                                          # text prompts to compare
    "advertising board",
    "billboard",
    "LED perimeter board",
]
CONFIDENCE = 0.3          # SAM3 confidence threshold (0.5 is the default; lower = more candidates)
RESOLUTION = 1008         # SAM3 input resolution (default 1008)
OUTPUT_DIR = "output/image_probe"
# --------------------------------------------------------------------------------------

PALETTE = [
    (0, 200, 255), (0, 255, 120), (255, 80, 80), (255, 200, 0), (200, 0, 255),
    (0, 120, 255), (120, 255, 0), (255, 0, 180), (80, 255, 255), (255, 140, 0),
]


def read_frame(video_path: str, t: float):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def overlay(frame_bgr, masks, boxes, scores, title):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for i, (m, b, s) in enumerate(zip(masks, boxes, scores)):
        color = PALETTE[i % len(PALETTE)]
        m = m.astype(bool)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[m] = (out[m] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, f"{i} {s:.2f}", (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def contact_sheet(images, cols, cell_width=640):
    cells = []
    for img in images:
        h, w = img.shape[:2]
        cells.append(cv2.resize(img, (cell_width, int(h * cell_width / w))))
    cell_h = max(c.shape[0] for c in cells)
    rows = (len(cells) + cols - 1) // cols
    sheet = np.zeros((rows * cell_h, cols * cell_width, 3), np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        sheet[r * cell_h:r * cell_h + c.shape[0], k * cell_width:k * cell_width + c.shape[1]] = c
    return sheet


def main():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    model = build_sam3_image_model()                      # downloads facebook/sam3 on first run
    processor = Sam3Processor(model, resolution=RESOLUTION, confidence_threshold=CONFIDENCE)
    print(f"model ready in {time.time() - t0:.1f}s")

    summary = []
    for video in VIDEOS:
        clip = Path(video).stem
        sheet_images = []
        for t in TIMESTAMPS:
            frame = read_frame(video, t)
            if frame is None:
                print(f"{clip} t={t}s: no frame")
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # SAM3 runs its backbone in bf16 (the video predictor wraps inference in
            # torch.autocast); without autocast the fp32 weights hit bf16 activations.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(image)        # backbone runs once per frame
            for prompt in PROMPTS:
                tic = time.time()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = processor.set_text_prompt(prompt=prompt, state=state)
                dt = time.time() - tic
                masks = to_numpy(out["masks"])
                boxes = to_numpy(out["boxes"]).reshape(-1, 4)
                scores = to_numpy(out["scores"]).reshape(-1)
                masks = masks.reshape(-1, *masks.shape[-2:]) if masks.size else np.zeros((0, *frame.shape[:2]))
                title = f"{clip} t={t:g}s  '{prompt}'  n={len(scores)}  {dt * 1000:.0f} ms"
                img = overlay(frame, masks, boxes, scores, title)
                name = f"{clip}_{t:g}s_{prompt.replace(' ', '_')}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                sheet_images.append(img)
                summary.append({
                    "clip": clip, "t": t, "prompt": prompt, "count": int(len(scores)),
                    "scores": [round(float(s), 3) for s in scores],
                    "mask_area_pct": [round(float(m.sum()) * 100 / m.size, 2) for m in masks],
                    "ms": round(dt * 1000),
                })
                print(f"{title}  scores={[round(float(s), 2) for s in scores]}")
                processor.reset_all_prompts(state)
        if sheet_images:
            sheet = contact_sheet(sheet_images, cols=len(PROMPTS))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{clip}_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"done: {OUTPUT_DIR} ({len(summary)} runs)")


if __name__ == "__main__":
    main()
