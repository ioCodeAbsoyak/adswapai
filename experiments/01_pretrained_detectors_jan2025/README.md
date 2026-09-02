# 01 · Pretrained detectors in Docker (17 Jan – 1 Feb 2025)

The project starts from zero: is the GPU usable, can a pretrained detector run
behind a web UI, and can it run on video frames? Everything here uses stock
torchvision models (COCO classes, cars as the test object) inside Docker
Compose. No ad boards yet.

| Step | Date | What it does | Outcome |
|------|------|--------------|---------|
| `01_cuda_check/` | 17 Jan | Allocate a tensor on `cuda:0`, print memory stats. | GPU works from PyTorch. |
| `02_click_detect_fasterrcnn/` | 18 Jan | Two containers: nginx serves `index.html`, Flask loads **Faster R-CNN R50-FPN**. Upload a still image, click on it, the backend returns the COCO box that contains the click. | First end-to-end app. Ran on CPU inside Docker (no GPU reservation in compose yet). |
| `03_maskrcnn_overlay/` | 19 Jan | Detector swapped for **Mask R-CNN**; the clicked instance's mask is returned as a byte array and composited on a second canvas with a colour picker and opacity slider. | First pixel-accurate region, the seed of the "paint over this area" idea. |
| `04_video_frames/` | 24–28 Jan | A third container serves MP4 files. The browser plays a video into a `<canvas>`, a click sends a JPEG snapshot plus coordinates, the backend returns the box. Compose gains the GPU reservation and the `numpy<2` pin (28 Jan). | Detection on video frames, GPU active in the container. |
| `05_mjpeg_pipeline/` | 1 Feb | Design flipped: the server reads the video itself, runs Mask R-CNN per frame, draws car boxes and streams MJPEG at `/video_feed`. Code split into `modules/detection.py` and `modules/streaming.py`. | The server-side per-frame pipeline that every later version builds on. |

## What was dropped and why

* `RD/20250120`, `RD/20250125` were byte-identical copies of the previous day.
* `RD/20250130` bolted the SORT tracker onto step 04: the browser pushed every
  frame to a `/update_frame` route that did not exist, `/track` returned the
  seed box plus random jitter, and the vendored `sort.py` used the `TkAgg`
  backend in a headless container. It never worked and is only mentioned here.
* `backup.txt` copies, runtime `uploads/` snapshots and a 510 MB test video.

## Running a step

Each step folder is a self-contained Compose project (`docker compose up --build`
in that folder). They pin `torch 2.1 / CUDA 12.1` era wheels; treat them as
historical snapshots rather than something to deploy today. Step 05 expects the
input clip at `/app/videos/raw/CarsMoving.mp4` inside the container (mount a
folder there).
