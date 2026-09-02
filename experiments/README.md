# Experiments

The R&D history, one folder per phase, in chronological order. Each folder
has a README that explains the step, lists the scripts that were kept and
says what was dropped from the archive and why. Every script starts with a
one-line header `AdSwapAI R&D, <date>: <purpose>`.

| Folder | Period | Theme |
|--------|--------|-------|
| `01_pretrained_detectors_jan2025/` | 17 Jan – 1 Feb 2025 | Pretrained Faster/Mask R-CNN in Docker: click-to-detect on stills, then video frames, then a server-side MJPEG pipeline |
| `02_cars_to_adboards_mar2025/` | 1 – 17 Mar 2025 | Car tracking GUI, MOG2 masking, car removal by inpainting; pivot to ad boards, hand-drawn polygons, first replacement |
| `03_polygon_tracking_apr2025/` | 1 – 17 Apr 2025 | LK / SIFT / CSRT / SuperGlue on user polygons; custom YOLOv8-seg model; automatic replacement; Kalman "known ads" tracker |
| `04_maskrcnn_replacement_apr2025/` | 18 Apr – 8 May 2025 | Detectron2 Mask R-CNN, perspective paste, two-stage pipeline, DeepSORT-style and RAFT tracking, rollback to per-frame |
| `05_web_app_may2025/` | 8 May – 5 Jun 2025 | The Flask + nginx application used for the investor demo |
| `06_sam2_baseline_aug2025/` | 25 Aug 2025 | SAM2 video predictor baseline, last experiment before the shelf |
| `07_sam3_sep2026/` | 2 Sep 2026 → | **Active.** SAM3 text-prompted board detection (works), SAM3 / SAM 3.1 / hybrid tracking probes; next: board-space replacement |

These are historical snapshots: they were cleaned (English comments, relative
paths, a few one-line bug fixes noted in the READMEs) but not modernised.
Dependencies are the ones of spring 2025: PyTorch 2.x with CUDA, OpenCV
(contrib for CSRT), Ultralytics, Detectron2, PyQt5 for the desktop tools.
Inputs go in a `data/` folder next to each script (see `docs/assets.md`).
