"""AdSwapAI R&D, 2026-09-02: SAM3 with a visual exemplar ("find the boards that look like this one").

SAM3 accepts box exemplars on top of (or instead of) the text prompt. This
script compares, on the same frames:
  A. text only
  B. text + one positive exemplar box (a board)
  C. text + positive box + one negative box (crowd / pitch), if given
Boxes are normalized [x0, y0, x1, y1] in [0, 1]. Leave EXEMPLAR_BOX empty to
pick one interactively with the mouse (a window opens on the first frame),
or set AUTO_EXEMPLAR = True to use the highest-scoring text detection.
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
VIDEO = "data/1.mp4"
TIMESTAMPS = [1.0, 4.0, 8.0, 11.0]
PROMPT = "billboard"
EXEMPLAR_FRAME_T = 1.0            # the exemplar box is drawn on this frame ...
EXEMPLAR_BOX = []                 # ... as normalized [x0, y0, x1, y1]; empty = interactive selection
NEGATIVE_BOX = []                 # optional normalized [x0, y0, x1, y1] on something that is NOT a board
AUTO_EXEMPLAR = False             # True: use the best text detection as the exemplar instead
CONFIDENCE = 0.3
OUTPUT_DIR = "output/exemplar_probe"
# --------------------------------------------------------------------------------------

PALETTE = [
    (0, 200, 255), (0, 255, 120), (255, 80, 80), (255, 200, 0), (200, 0, 255),
    (0, 120, 255), (120, 255, 0), (255, 0, 180), (80, 255, 255), (255, 140, 0),
]


def read_frame(video, t):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def to_np(x):
    return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def unpack(out):
    masks = to_np(out["masks"])
    boxes = to_np(out["boxes"]).reshape(-1, 4)
    scores = to_np(out["scores"]).reshape(-1)
    masks = masks.reshape(-1, *masks.shape[-2:]) > 0.5 if masks.size else np.zeros((0, 1, 1), bool)
    return masks, boxes, scores


def xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def overlay(frame, masks, boxes, scores, title, exemplar=None, negative=None):
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
    for box, color in ((exemplar, (255, 255, 255)), (negative, (0, 0, 255))):
        if box:
            cv2.rectangle(out, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)), color, 3)
    cv2.rectangle(out, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(out, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def pick_box_interactively(frame):
    print("Draw a box around ONE advertising board, then press ENTER (c to cancel).")
    x, y, w, h = cv2.selectROI("exemplar (draw a board, press ENTER)", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        raise SystemExit("no box selected")
    H, W = frame.shape[:2]
    return [x / W, y / H, (x + w) / W, (y + h) / H]


def main():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE)

    exemplar_frame = read_frame(VIDEO, EXEMPLAR_FRAME_T)
    exemplar = list(EXEMPLAR_BOX)
    if not exemplar and not AUTO_EXEMPLAR:
        exemplar = pick_box_interactively(exemplar_frame)
    print(f"exemplar box (normalized xyxy): {[round(v, 3) for v in exemplar] if exemplar else 'auto'}")

    rows, records = [], []
    for t in TIMESTAMPS:
        frame = read_frame(VIDEO, t)
        if frame is None:
            continue
        H, W = frame.shape[:2]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        # A. text only
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = processor.set_text_prompt(prompt=PROMPT, state=state)
        mA, bA, sA = unpack(out)
        ex = exemplar
        if AUTO_EXEMPLAR:
            if len(sA) == 0:
                print(f"t={t}s: no text detection to use as exemplar, skipping")
                processor.reset_all_prompts(state)
                continue
            b = bA[int(np.argmax(sA))]
            ex = [b[0] / W, b[1] / H, b[2] / W, b[3] / H]

        # B. text + positive exemplar
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = processor.add_geometric_prompt(box=xyxy_to_cxcywh(ex), label=True, state=state)
        mB, bB, sB = unpack(out)

        # C. + negative box
        mC = bC = sC = None
        if NEGATIVE_BOX:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = processor.add_geometric_prompt(box=xyxy_to_cxcywh(NEGATIVE_BOX), label=False, state=state)
            mC, bC, sC = unpack(out)
        processor.reset_all_prompts(state)

        imgs = [overlay(frame, mA, bA, sA, f"A text '{PROMPT}' t={t:g}s n={len(sA)}"),
                overlay(frame, mB, bB, sB, f"B text+exemplar t={t:g}s n={len(sB)}", exemplar=ex)]
        if mC is not None:
            imgs.append(overlay(frame, mC, bC, sC, f"C +negative t={t:g}s n={len(sC)}", exemplar=ex, negative=NEGATIVE_BOX))
        cells = [cv2.resize(im, (640, int(H * 640 / W))) for im in imgs]
        rows.append(np.hstack(cells))
        rec = {"t": t, "A_scores": [round(float(s), 2) for s in sA], "B_scores": [round(float(s), 2) for s in sB]}
        if sC is not None:
            rec["C_scores"] = [round(float(s), 2) for s in sC]
        records.append(rec)
        print(f"t={t:g}s  A n={len(sA)} {rec['A_scores']}  |  B n={len(sB)} {rec['B_scores']}" + (f"  |  C n={len(sC)} {rec['C_scores']}" if sC is not None else ""))

    stem = Path(VIDEO).stem
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{stem}_{PROMPT.replace(' ', '_')}_exemplar_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    with open(os.path.join(OUTPUT_DIR, f"{stem}_{PROMPT.replace(' ', '_')}_exemplar.json"), "w", encoding="utf-8") as f:
        json.dump({"exemplar_box": exemplar, "negative_box": NEGATIVE_BOX, "auto": AUTO_EXEMPLAR, "frames": records}, f, indent=2)
    print(f"done: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
