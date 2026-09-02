# 02 · From tracking cars to replacing ad boards (1 – 17 Mar 2025)

March starts with a generic computer-vision problem (follow a car, then make
it disappear) and ends with the first advertisement replaced in a football
clip. The target switched from COCO vehicles to pitch-side boards on 14 March.

| Step | Date | What it does | Outcome |
|------|------|--------------|---------|
| `01_car_tracking_gui/` | 1 Mar | PyQt5 desktop app. `car_tracking_csrt.py`: drag a box, OpenCV CSRT follows it. `car_tracking_fasterrcnn_deepsort.py`: Faster R-CNN finds vehicles, a hand-written Kalman + ResNet18-appearance tracker (DeepSORT-style) keeps identities. | Tracking works on road footage; the GUI is the first interactive tool. |
| `02_motion_masking/` | 3 Mar | `mog2_car_masking.py`: classical MOG2 background subtraction, contour filters, centroid tracker, per-car mask video. `yolov5_predict.py`: the one-liner that produced the YOLOv5n weights later exported to ONNX/TensorRT. | The TensorRT 10 attempt on the new Blackwell GPU stalled on decoding the `(1, 84, 8400)` output tensor and was dropped; the classical pipeline worked. |
| `03_car_removal_inpainting/` | 10–13 Mar | `inpaint_cars_minimal.py`: Mask R-CNN masks + `cv2.inpaint`, the first "remove the object" result. `car_remover.py` + `gpu_monitor.py` + `main.py`: batched removal with mask dilation, shadow extension, temporal blending and NVML-driven batch size, as a CLI. | Object removal works; the temporal-blend formula in `car_remover.py` was corrected while curating (it darkened unmasked pixels). |
| `04_adboard_heuristics/` | 14 Mar | `football_adboard_detector.py`: find the pitch edge (green mask, Sobel, Hough), take the strip above it, split it into board segments by colour variance. | Only the best of several heuristics is kept. None was reliable on real broadcast stills, and COCO has no "ad board" class. |
| `05_manual_polygons/` | 16 Mar | `polygon_adboard_selector.py`: annotate boards by hand as polygons (pickle + mask export). `precise_mask_selector.py`: subtract people from the polygon with Mask R-CNN person masks. | The pivot: automatic detection abandoned for user-drawn polygons, occluders handled with a second model. |
| `06_polygon_tracking_replacement/` | 16–17 Mar | `polygon_tracker.py`: SIFT keypoints inside the polygon, per-frame matching + RANSAC homography, YOLOv8n-seg person masks excluded. `polygon_replacement.py`: several polygons, each tracked the same way, an image perspective-warped into every polygon minus people. | **First replaced advertisement in video.** A debugging `waitKey(0)` was changed to `waitKey(1)` while curating. |

## What was dropped and why

* Two "GPU template matching" trackers (`conv2d` cross-correlation with a fixed
  threshold, placeholder batch processor).
* The raw TensorRT inference loops (`test.py`, `test2.py` of 3 Mar): output
  parsing was never finished.
* A dozen `testN.py` iterations, ChatGPT/Claude variants of the same file
  (`_C`, `_GPT`, `_claude` suffixes), a DeepLabV3 dead end, an optical-flow
  warp with swapped axes, and a PyQt car app that mixed fp16 batches into an
  fp32 model.
* Heuristic detectors superseded by the kept one, a YOLOv4 branch without
  weights, and a fixed "strip at 22–28 % of the height" viewer.

## Data and weights

Scripts default to `data/…` next to them (edit the constants at the top).
The clips used were `adVideo1.mp4` / `adVideo2.mp4` (the same ones later
bundled with the web app as sample videos) and `replace.jpg`. YOLO weights are
downloaded by `ultralytics`/`torch.hub` on first run.
