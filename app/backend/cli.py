#!/usr/bin/env python3
"""
Command line runner for the AdSwapAI pipeline (no web server needed).

Examples (inside the backend container or any env with the dependencies):
  python3 cli.py input.mp4 output.mp4 --replacement ad.jpg
  python3 cli.py input.mp4 output.mp4 --mode mask --mask-color 00ff00 --mask-alpha 0.5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import cv2

from pipeline import JobParams, Models, process_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdSwapAI billboard replacement CLI")
    p.add_argument("input", help="input video path")
    p.add_argument("output", help="output video path (.mp4)")
    p.add_argument("--mode", choices=["image", "mask"], default="image")
    p.add_argument("--replacement", help="replacement image (required for image mode)")
    p.add_argument("--conf", type=float, default=0.5, help="billboard confidence threshold")
    p.add_argument("--human-conf", type=float, default=0.5, help="person/ball confidence threshold")
    p.add_argument("--no-human-filter", action="store_true", help="disable person/ball protection")
    p.add_argument("--min-mask-size", type=float, default=0.0, help="min mask area fraction of frame")
    p.add_argument("--hold-frames", type=int, default=3, help="temporal smoothing frames (0 = off)")
    p.add_argument("--feather", type=float, default=2.0, help="edge feather in pixels")
    p.add_argument("--mask-color", default="00ff00", help="hex RGB for mask mode")
    p.add_argument("--mask-alpha", type=float, default=0.5)
    p.add_argument("--billboard-weights", default=os.environ.get("BILLBOARD_WEIGHTS", os.path.join(BASE_DIR, "model_final.pth")))
    p.add_argument("--coco-weights", default=os.environ.get("COCO_WEIGHTS", os.path.join(BASE_DIR, "models", "mask_rcnn_R_50_FPN_3x.pkl")))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    replacement = None
    if args.mode == "image":
        if not args.replacement:
            print("error: --replacement is required in image mode", file=sys.stderr)
            return 2
        replacement = cv2.imread(args.replacement, cv2.IMREAD_COLOR)
        if replacement is None:
            print(f"error: cannot read replacement image {args.replacement}", file=sys.stderr)
            return 2

    color = args.mask_color.lstrip("#")
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    params = JobParams(
        mode=args.mode, conf_threshold=args.conf, human_conf_threshold=args.human_conf,
        min_mask_size=args.min_mask_size, enable_human_filter=not args.no_human_filter,
        mask_color_bgr=(b, g, r), mask_alpha=args.mask_alpha,
        hold_frames=args.hold_frames, feather=args.feather,
    )

    models = Models(args.billboard_weights, args.coco_weights)
    started = time.time()
    last = {"t": 0.0}

    def on_progress(current: int, total: int) -> None:
        now = time.time()
        if now - last["t"] >= 1.0 or current == total:
            last["t"] = now
            pct = (100.0 * current / total) if total else 0.0
            fps = current / (now - started) if now > started else 0.0
            print(f"\r  frame {current}/{total} ({pct:5.1f}%)  {fps:4.1f} fps", end="", flush=True)

    stats = process_video(args.input, args.output, params, models, replacement=replacement, progress_cb=on_progress)
    print()
    print(f"done: {args.output}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
