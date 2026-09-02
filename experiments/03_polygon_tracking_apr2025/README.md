# 03 · From hand-drawn polygons to detector-driven replacement (1 – 17 Apr 2025)

April is the tracking chapter. It starts by asking the user to outline one
board on the first frame and trying every classical way to follow that
polygon; within three days it is clear that manually initialised trackers fail
as soon as boards leave the frame, multiply or lose texture. The answer was a
custom one-class **YOLOv8s-seg "billboard"** model (trained from the
`training/dataset_v1` Roboflow export): its segmentation polygon supplies the
homography directly, every frame, with no manual initialisation.

| Step | Date | What it does | Outcome |
|------|------|--------------|---------|
| `01_manual_polygon_trackers/` | 1 Apr | `lk_four_corners.py`: click 4 corners, Lucas-Kanade tracks them, perspective warp + alpha blend. `lk_with_sift_reacquisition.py`: SIFT re-detects the board when LK loses corners. | Cheap and accurate while it holds; drifts and dies under occlusion. |
| `02_tracker_comparison/` | 2 Apr | The same `VideoPolygonMapper` class with three interchangeable trackers: **SIFT + RANSAC homography**, **Lucas-Kanade** on the polygon corners, **CSRT** on the bounding box. | SIFT re-acquires and is perspective-correct but needs texture; LK is fragile; CSRT is robust but only gives translation/scale. |
| `03_occlusion_multiboard_superglue/` | 3 Apr | `csrt_hog_occlusion.py`: HOG pedestrian boxes cut out of the overlay (first occlusion handling). `csrt_multi_polygon.py`: several boards, one CSRT each. `superglue_probe.py`: SuperPoint/SuperGlue matching probe. | Occlusion and multi-board handled crudely; SuperGlue abandoned within the evening. |
| `04_detector_tracking/` | 3 Apr | `yolov8_deepsort_benchmark.py`: YOLOv8 + DeepSORT on COCO classes (people, ball). `yolov8seg_billboard_masks.py`: **first run of the custom billboard model** (`best.pt`), masks visualised. | The board itself becomes detectable every frame. |
| `05_automatic_replacement/` | 8 Apr | `first_automatic_replacement.py` (bounding box warp), `seg_polygon_perspective_replacement.py` (mask contour to 4-point quad to perspective warp), `smoothed_replacement.py` (orientation check + exponential smoothing), `botsort_track_masks.py` (Ultralytics `model.track`, BoT-SORT, no DeepSORT dependency). | Fully automatic replacement. Remaining problem: jitter between independent per-frame detections. |
| `06_stable_tracking/` | 13 Apr | `known_ads_memory.py`: remember boards across frames by polygon IoU, EMA smoothing. `yolo_init_sift_tracker.py`: YOLO picks the board on frame 1, SIFT tracks it. `stable_ad_tracker_kalman.py`: per-corner 16-state Kalman filter, IoU matching to the predicted polygon, coasting through misses, detector run only every 10 frames. | The most complete tracker of the project. Debug drawing is behind `DRAW_DEBUG`. |

17 April: environment check for nightly PyTorch + CUDA 12.8 + TensorRT (not
kept, one print statement), which opens the "make it fast" work of the next chapter.

## What was dropped and why

* A kitchen-sink tracker running SIFT, ORB and LK at once with arbitrary weights.
* A "GPU SIFT" that wrapped CPU SIFT in torch tensors and swapped the R/B channels.
* The SuperGlue tracker class (`KeyError` on the first frame, homography never applied).
* YOLO scripts filtering on a class COCO does not have, and a `best.pt` script
  that labelled boards as "Football".
* Intermediate steps fully contained in the kept files, ChatGPT/Claude/other
  variants of the same script (`_GPT`, `_Cl`, `_BB` suffixes), a person/ball
  exclusion that used class 0 of a single-class model (it erased the boards).

## Data and weights

`best.pt` (YOLOv8s-seg, one class `billboard`, Ultralytics 8.3.105, retrained
8 and 16 April) is not in git, see `docs/assets.md`. Scripts default to
`data/adVideo1.mp4`, `data/replace.jpg` and `best.pt` next to the script.
`superglue_probe.py` needs a clone of
https://github.com/magicleap/SuperGluePretrainedNetwork next to it.
