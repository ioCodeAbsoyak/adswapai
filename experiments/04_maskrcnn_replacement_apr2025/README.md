# 04 · Mask R-CNN, RAFT and the replacement pipeline (18 Apr – 8 May 2025)

With the YOLOv8-seg masks judged too coarse, the detector became a
**Detectron2 Mask R-CNN (R50-FPN-3x)** fine-tuned on 153 frames labelled with
the VGG Image Annotator (see [`training/`](../../training)). The resulting
`model_final.pth` (29 Apr) is still the model in `app/`. The rest of the
chapter is about what to do with its masks: how to paste the ad, and how to
keep the paste stable across frames.

| Step | Date | What it does | Outcome |
|------|------|--------------|---------|
| `01_perspective_replacement/` | 20 Apr | Per-frame detection; mask contour → quadrilateral → `getPerspectiveTransform` → ad warped into each board. | Perspective-correct paste, but flicker between frames. |
| `02_two_stage_pipeline/` | 22 Apr | Stage 1 detects every N frames, assigns ids and smooths, writes JSON; stage 2 replays the replacement from the JSON with a different ad per track. | Clean separation of detection and rendering; the JSON intermediate made experiments cheap. |
| `03_tracked_replacement/` | 20–24 Apr | `csrt_tiled_replacement.py`: CSRT tracking with re-detection every 30 frames, seam-blended horizontal tiling for wide boards, blurred-alpha homography paste (the one script that shipped with a matching input/output video pair). `deepsort_style_billboard_tracker.py`: per-board Kalman filter, MobileNetV2 appearance features, Hungarian matching. | First "release" snapshot; tracks get stable ids. |
| `04_raft_optical_flow/` | 26 Apr | RAFT optical flow (torchvision) warps masks between detections so the detector can run less often and masks stop jittering; IoU-matched tracks, exponential mask smoothing, COCO people/ball masks excluded from the paste. `raft_billboard_replacement.py` is the compact reference version, `raft_billboard_pipeline_full.py` the full-featured one (CLI flags, mask mode, flow visualisation). | Works, but slow at 1080p and fragile on fast pans; complexity peaked here (1 270 lines). Foreground exclusion becomes standard. |
| `05_back_to_per_frame/` | 8 May | Deliberate rollback: one predictor, no tracking, per-frame mask replacement with the ad resized into the mask's bounding box. 150 lines. | Reliable. Became the core of the web app (`experiments/05_web_app_may2025`). |

## What was dropped and why

* A RAFT script using the original `raft` package that ran the detector on
  the previous frame and warped flow with a per-pixel Python loop.
* Three foreground/background variants of 22 Apr (debug logging only),
  static bounding-box replacers, `replace_video` v1/v2, mask-QA viewers.
* Intermediate RAFT patches (`fix`, `fix-warnings`), a flow visualisation
  tool that never called a replace function, a byte-identical duplicate of
  the full pipeline, and a variant that imported `deep_sort_realtime`
  without it being installed anywhere.
* `billboard_improved.py` (608 lines) and friends from early May: the
  complexity peak that the 8 May rollback replaced.

## Data and weights

`model_final.pth` (351 MB) and the clips are outside git (`docs/assets.md`).
Scripts default to `model_final.pth`, `data/adVideo1.mp4`, `data/replace.jpg`
or `data/replace/` next to them. The training container is
`training/detectron2/Dockerfile` (CUDA 12.8, nightly PyTorch, Detectron2
from source, RAFT extras).
