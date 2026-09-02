"""AdSwapAI R&D, 2026-09-02: SAM3 video tracking probe.

Prompt SAM3 once with a text prompt on one frame, let its video predictor
propagate the masks through the clip, and write a diagnostic video: one
colour per object id, id + probability labels, per-frame timing. No
per-frame detection. Edit the settings block and run.
"""
import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------- settings
VIDEO = "data/2.mp4"                 # sample clip (see docs/assets.md)
PROMPT = "sponsor banner"            # text prompt given on the first frame
START_FRAME = 0                      # frame index that receives the prompt
MAX_FRAMES = 150                     # frames to propagate (None = whole clip)
PROB_THRESH = 0.3                    # output probability threshold
OFFLOAD_VIDEO_TO_CPU = True          # keep decoded frames in RAM instead of VRAM
OUTPUT_DIR = "output/video_probe"
# --------------------------------------------------------------------------------------

PALETTE = [
    (0, 200, 255), (0, 255, 120), (255, 80, 80), (255, 200, 0), (200, 0, 255),
    (0, 120, 255), (120, 255, 0), (255, 0, 180), (80, 255, 255), (255, 140, 0),
    (160, 160, 255), (255, 255, 120), (120, 255, 200), (255, 120, 200), (200, 255, 0),
]


def color_for(obj_id: int):
    return PALETTE[int(obj_id) % len(PALETTE)]


def draw(frame, outputs, frame_idx, dt_ms):
    out = frame.copy()
    h, w = out.shape[:2]
    ids = outputs.get("out_obj_ids", [])
    probs = outputs.get("out_probs", [])
    masks = outputs.get("out_binary_masks", np.zeros((0, h, w), bool))
    boxes = outputs.get("out_boxes_xywh", np.zeros((0, 4)))
    for obj_id, p, m, b in zip(ids, probs, masks, boxes):
        c = color_for(obj_id)
        m = np.asarray(m).astype(bool)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[m] = (out[m] * 0.5 + np.array(c) * 0.5).astype(np.uint8)
        x, y, bw, bh = b
        x0, y0 = int(x * w), int(y * h)
        cv2.rectangle(out, (x0, y0), (int((x + bw) * w), int((y + bh) * h)), c, 2)
        cv2.putText(out, f"id{int(obj_id)} {float(p):.2f}", (x0, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(out, f"frame {frame_idx}  objects {len(ids)}  {dt_ms:.0f} ms  prompt '{PROMPT}'",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    from sam3.model_builder import build_sam3_video_predictor

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{VIDEO}: {width}x{height} @ {fps:.2f} fps, {total} frames")

    t0 = time.time()
    predictor = build_sam3_video_predictor()          # downloads facebook/sam3 on first run
    print(f"video predictor ready in {time.time() - t0:.1f}s")

    session = predictor.handle_request({
        "type": "start_session", "resource_path": VIDEO,
        "offload_video_to_cpu": OFFLOAD_VIDEO_TO_CPU,
    })
    session_id = session["session_id"]

    t1 = time.time()
    first = predictor.handle_request({
        "type": "add_prompt", "session_id": session_id, "frame_index": START_FRAME,
        "text": PROMPT, "output_prob_thresh": PROB_THRESH,
    })
    print(f"prompt on frame {START_FRAME}: {len(first['outputs']['out_obj_ids'])} objects "
          f"in {(time.time() - t1) * 1000:.0f} ms, ids={first['outputs']['out_obj_ids'].tolist()}")

    stem = Path(VIDEO).stem
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_{PROMPT.replace(' ', '_')}_track.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    stats = []
    id_counter = Counter()
    sheet_frames = {}
    last = time.time()
    n = 0
    for out in predictor.handle_stream_request({
        "type": "propagate_in_video", "session_id": session_id,
        "propagation_direction": "forward", "start_frame_index": START_FRAME,
        "max_frame_num_to_track": MAX_FRAMES, "output_prob_thresh": PROB_THRESH,
    }):
        now = time.time()
        dt_ms = (now - last) * 1000
        last = now
        fi = out["frame_index"]
        o = out["outputs"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        img = draw(frame, o, fi, dt_ms)
        writer.write(img)
        ids = [int(i) for i in o["out_obj_ids"]]
        id_counter.update(ids)
        stats.append({"frame": int(fi), "ids": ids, "probs": [round(float(p), 3) for p in o["out_probs"]],
                      "mask_area_pct": [round(float(m.sum()) * 100 / m.size, 2) for m in o["out_binary_masks"]],
                      "ms": round(dt_ms)})
        if n % 25 == 0 or fi == START_FRAME:
            sheet_frames[fi] = img
            print(f"frame {fi}: {len(ids)} objects {ids}  {dt_ms:.0f} ms")
        n += 1

    writer.release()
    cap.release()
    predictor.handle_request({"type": "close_session", "session_id": session_id})

    elapsed = time.time() - t1
    summary = {
        "video": VIDEO, "prompt": PROMPT, "frames": n, "elapsed_s": round(elapsed, 1),
        "fps": round(n / elapsed, 2) if elapsed else None,
        "unique_ids": len(id_counter),
        "frames_per_id": dict(sorted(id_counter.items())),
        "avg_objects_per_frame": round(sum(len(s["ids"]) for s in stats) / max(n, 1), 2),
    }
    with open(os.path.join(OUTPUT_DIR, f"{stem}_{PROMPT.replace(' ', '_')}_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": stats}, f, indent=2)

    if sheet_frames:
        keys = sorted(sheet_frames)[:6]
        cells = [cv2.resize(sheet_frames[k], (640, int(height * 640 / width))) for k in keys]
        rows = [np.hstack(cells[i:i + 2]) if i + 1 < len(cells) else np.hstack([cells[i], np.zeros_like(cells[i])])
                for i in range(0, len(cells), 2)]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{stem}_{PROMPT.replace(' ', '_')}_sheet.jpg"), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(json.dumps(summary, indent=2))
    print(f"video: {out_path}")


if __name__ == "__main__":
    main()
