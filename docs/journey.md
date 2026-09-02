# The journey: how AdSwapAI was built

A chronological account of the research and development, written from the
dated experiment folders (`RD/20250117` … `RD/20260902`) of the private
archive. Each section says what the goal was, what was tried, what failed,
and what carried over. Code for every step is under `experiments/`; the
current application is under `app/`.

Timeline at a glance:

```
Jan 2025   pretrained detectors in Docker, click-to-detect, MJPEG stream
Mar 2025   car tracking -> car removal -> pivot to ad boards -> first replacement
Apr 2025   tracker comparison -> custom YOLOv8-seg -> Detectron2 Mask R-CNN -> RAFT
May 2025   Flask web app, job queue, admin page, smart tiling, AdSwap AI brand
Jun 2025   investor search (landing page published, demo online)
Aug 2025   SAM2 baseline experiment, project shelved
Sep 2026   clean-up, bug fixes, this repository
```

---

## 1 · Pretrained detectors in Docker (17 Jan – 1 Feb 2025)

**Goal.** Get a GPU-backed detector running behind a browser UI and see what
"detect the thing the user clicked on" looks like.

**What was built.** A CUDA smoke test, then a two-container Compose project:
nginx serving a page with a `<canvas>`, Flask loading torchvision's
Faster R-CNN. The user uploads a still image, clicks, and gets the COCO box
that contains the click. A day later the detector became Mask R-CNN and the
clicked instance's *mask* came back as a byte array that the page composited
with a colour picker and opacity slider. That was the first pixel-accurate
region and, in hindsight, the seed of the whole product: "paint something else
over exactly this area".

Video came next. A third container served MP4 files, the browser painted the
playing video onto the canvas and sent a JPEG snapshot with the click
coordinates. The compose file finally got a GPU reservation on 28 January
(until then the backend had been running on CPU inside Docker without anyone
noticing) and a `numpy<2` pin.

**What failed.** An attempt on 30 January to bolt the SORT tracker onto this
design: the browser pushed every frame to a route that did not exist, the
tracking endpoint returned the seed box plus random jitter, and the vendored
SORT used a Tk backend in a headless container. Browser-driven frame pushing
was abandoned.

**What carried over.** On 1 February the design flipped to what every later
version uses: the server reads the video itself, runs the model per frame,
and streams the result (MJPEG at the time). Detection and streaming were
split into modules.

Code: `experiments/01_pretrained_detectors_jan2025/`

---

## 2 · From tracking cars to replacing ad boards (1 – 17 Mar 2025)

**Goal.** Learn tracking and object removal on easy footage (cars on a road),
then apply it to advertising boards.

**Tracking (1 Mar).** A PyQt5 desktop app: drag a box around a car and OpenCV
CSRT follows it; then Faster R-CNN finds vehicles automatically and a
hand-written Kalman + ResNet18-appearance tracker (DeepSORT-style) keeps
identities. Two "GPU template matching" trackers built on raw `conv2d`
cross-correlation were dropped: an un-normalised correlation with a fixed
threshold locks onto the brightest patch.

**TensorRT detour (3 Mar).** YOLOv5n was exported to ONNX and TensorRT and
run with the raw TensorRT 10 API on the new Blackwell GPU. The engine ran, but
its `(1, 84, 8400)` output was parsed as rows of `[x, y, w, h, conf, …]` and
never decoded correctly. Cars were found with classical MOG2 background
subtraction instead.

**Removal (10 – 13 Mar).** "Can we make an object disappear?" Mask R-CNN
masks, dilation plus a shadow extension under each car, two-stage OpenCV
inpainting (Navier-Stokes then Telea), temporal blending with the previous
frame, batching, and an NVML-driven batch-size controller, packaged as a CLI.
Along the way: a DeepLabV3 semantic-segmentation dead end, an optical-flow
warp with swapped axes, and two assistants' variants of the same file kept
side by side (`_claude`, `_chatgpt` suffixes in the archive).

**The pivot (14 – 16 Mar).** Target changed to pitch-side boards in stills
from real broadcasts. COCO has no such class (the `tv` class was tried as a
proxy) and none of the colour / field-edge / Hough heuristics was reliable.
Within 48 hours the boards were being annotated **by hand as polygons**, and
people in front of them were cut out with Mask R-CNN person masks.

**First replacement (16 – 17 Mar).** SIFT keypoints inside the user polygon,
per-frame matching with a ratio test and RANSAC homography, YOLOv8n-seg person
masks excluded from the overlay, and finally an ad image perspective-warped
into every tracked polygon. The first video with a swapped advertisement.

Code: `experiments/02_cars_to_adboards_mar2025/`

---

## 3 · From hand-drawn polygons to detector-driven replacement (1 – 17 Apr 2025)

**Goal.** Make the polygon follow the board reliably, then remove the need for
a human to draw it.

**Classical trackers (1 – 3 Apr).** With one hand-drawn polygon on frame 1:

* *Lucas-Kanade on the four corners*: cheapest and sub-pixel accurate, but
  isolated points drift and vanish under occlusion; SIFT re-acquisition was
  added as a fallback.
* *SIFT + RANSAC homography against the first frame*: re-acquires after loss
  and is perspective-correct, but needs texture (flat-colour boards give few
  matches) and costs a full-frame SIFT per frame; a "GPU SIFT" that wrapped
  CPU SIFT in torch tensors sped nothing up and swapped the colour channels.
* *CSRT on the bounding box*: most robust to partial occlusion, but only
  outputs a box, so the polygon cannot follow perspective changes. It hosted
  the first occlusion handling (HOG pedestrian boxes cut out of the overlay)
  and the first multi-board experiment.
* *SuperPoint/SuperGlue*: tried the same evening as a learned matcher,
  abandoned within the hour (integration errors, no visible gain).

The conclusion: any manually initialised tracker fails the moment boards
leave the frame, multiply or lose texture.

**A detector for the board itself (3 – 8 Apr).** YOLOv5/v8 on COCO were first
used to find the *occluders* (people, ball). Then a single-class
**YOLOv8s-seg "billboard"** model was trained on a Roboflow-labelled set
(`training/dataset_v1`), and its segmentation polygon fed the homography
directly: contour → `approxPolyDP` → four corners → `warpPerspective`.
Manual initialisation and drift were gone. DeepSORT and Ultralytics'
built-in BoT-SORT were tried for identities; neither turned out to be
necessary for replacement.

**Stability (13 Apr).** Per-frame detections jitter. Three answers were
written the same day: a "known ads" memory that matches detections to
remembered boards by polygon IoU and smooths them; a hybrid where YOLO picks
the board on frame 1 and SIFT tracks it; and a proper multi-object tracker
with a 16-state Kalman filter per board (four corners plus velocities), IoU
matching to the *predicted* polygon, coasting through missed detections and
the detector run only every tenth frame. The last one is the most complete
tracker of the project.

**17 Apr.** An environment check for nightly PyTorch + CUDA 12.8 + TensorRT:
the "make it fast" thread that was never finished.

Code: `experiments/03_polygon_tracking_apr2025/`

---

## 4 · Mask R-CNN, RAFT and the replacement pipeline (18 Apr – 8 May 2025)

**Goal.** Better masks than YOLOv8-seg gave, and a replacement that survives
occlusion and camera motion.

**Dataset.** The Roboflow set (9 source frames, three augmented copies each)
was too small. The frames were relabelled as polygons with the VGG Image
Annotator and the set grew to 153 frames (`training/dataset_v3`, one class
"pitch side billboards"). The VIA csv was converted to COCO json and merged
with Roboflow's COCO export into `annotationsFinal.json` (153 images, 675
board annotations). A CVAT instance was also set up and evaluated for
labelling at scale, but the small set was finished in VIA.

**Training.** Detectron2's Mask R-CNN R50-FPN-3x fine-tuned from the COCO
checkpoint for 5 000 iterations in a CUDA 12.8 container; the run of
29 April produced the `model_final.pth` the web app still uses. A longer
fine-tune schedule was started and abandoned at 7 500 of 10 000 iterations.

**Replacement (20 – 22 Apr).** First a per-frame perspective paste (mask
contour → quadrilateral → homography), then a two-stage design: stage 1
detects every N frames, assigns ids, smooths and writes JSON; stage 2 renders
from the JSON with a different ad per board. A self-contained snapshot of
20 April, CSRT tracking with re-detection every 30 frames, seam-blended
horizontal tiling for wide boards and a blurred-alpha homography paste, is the
one script of the period that shipped together with its input and output
video.

**Tracking (24 – 26 Apr).** A DeepSORT-style tracker (per-board Kalman
filter, MobileNetV2 appearance features, Hungarian matching), then RAFT
optical flow from torchvision to warp masks between detections so the
detector could run less often and masks would stop jittering, with COCO
people/ball masks excluded from the paste. The full RAFT pipeline reached 1 270 lines with CLI flags, a mask
mode and flow visualisation. It worked, but it was slow at 1080p and fragile
on fast pans.

**The rollback (8 May).** The last file of the period is 150 lines: one
predictor, no tracking, the ad resized into each mask's bounding box, every
frame. It was more reliable than everything above it and became the core of
the web app. The lesson, that per-frame detection with occluder subtraction
beats elaborate tracking on this footage, held until the temporal smoothing
added in 2026.

Code: `experiments/04_maskrcnn_replacement_apr2025/`, `training/`

---

## 5 · The web application (8 May – 5 Jun 2025)

**Goal.** Something an investor can click on.

The Detectron2 pipeline was wrapped in a Flask API behind an nginx site. In
four weeks: a single blocking `/process` endpoint (8 May); job ids, a status
endpoint and an admin page (14 May); processing moved to a background thread,
`minAreaRect` + perspective transform instead of an axis-aligned paste, boards
wider than 60 % of the frame split off as "big" (16 May); "smart tiling" that
lays a tiled ad across the full-width LED strip (24 May); and on 5 June the
rebrand from Altervision to **AdSwap AI**, a landing page with a before/after
slider and a 301 from the old domain. The Dockerfile grew a two-stage build
that compiles Detectron2 into a wheel, and, prepared for a tracking layer
(DeepSORT) that the code never used.

The demo went online and was used in the investor conversations of June and
July 2025. No investment was found, and the code stayed as it was.

Code: `experiments/05_web_app_may2025/`

---

## 6 · SAM2 baseline (25 Aug 2025)

*(see the chapter README for the file-level detail)*

One last experiment before shelving the project: Meta's Segment Anything 2
video predictor, prompted with a grid of points on the first frame and
propagated through the clip, to see whether prompt-based segmentation could
replace the trained detector. It produced masks that follow objects well, but
without a board-specific prompt strategy it segments everything; the
experiment stayed a baseline.

Code: `experiments/06_sam2_baseline_aug2025/`

---

## 7 · Clean-up (2 Sep 2026)

Fifteen months later the May 2025 build still ran (after a 23-minute Docker
build). The clean-up kept the architecture and the model and fixed what the
demo had been hiding:

* wide boards in far shots were silently skipped because the corner
  heuristic (`x+y` min/max) collapsed on thin, slanted strips;
* human/ball protection did not apply to small boards in image mode;
* black areas in an ad punched holes in the replacement;
* every threshold change rebuilt a Mask R-CNN; ffmpeg failures left jobs
  spinning forever; temp uploads were never deleted; two of eight API routes
  were proxied.

It added a single GPU worker queue, an IoU tracker for temporal hold,
feathered edges, aspect-correct tiling, one-pass ffmpeg encoding with the
original audio, a CLI, and health checks. Verified on the three sample clips
at about 6 fps (1080p, RTX 5070 Ti).

Code: `app/`

---

## 8 · SAM3: detection without a dataset (2 Sep 2026, ongoing)

**Goal.** Make the POC good on the existing clips without labelling more
data: replace per-frame Mask R-CNN with a prompt-based foundation model,
then track instead of re-detecting.

**Environment.** SAM3 on Windows needs a few things the package does not
declare (einops, pycocotools, psutil, a community Triton build), bf16
autocast around inference, and gated Hugging Face weights (manual approval
by Meta). All of it is in `experiments/07_sam3_sep2026/README.md`.

**Detection.** The prompt decides everything. "billboard" (the obvious
choice) misses wide shots; "sponsor banner" and "advertisement" find every
board in every sampled frame of the three clips, far LED strips included,
with sharp masks, at ~200 ms per 1080p frame. No trained model involved.

**Tracking, three ways on the same 150 frames.** SAM3's video predictor
tracks well and survives a camera cut, but at 0.24 fps with 15 GB of VRAM;
a hybrid (SAM3 image detection every frame + IoU association + histogram
shot-cut detection) runs at 3.4 fps but fragments ids when a board is found
as a whole in one frame and as its sponsor panels in the next; the SAM 3.1
multiplex predictor prompts ten times faster than SAM3 but its propagation
crashed at the VRAM limit and is not measured yet.

**Decision pending.** The architecture is settled: detect on the first
frame and on events (shot cuts, new boards), track in between, render each
board in its own board-space so a half-visible board shows half an ad. Open
question for the next session: whether SAM 3.1 or our own camera-motion
propagation fills the frames between detections.

## Lessons

* **Detection beat tracking.** Every manually initialised tracker (LK, SIFT,
  CSRT, SuperGlue) failed on real footage; a task-specific segmentation model
  run every frame, plus a little temporal smoothing, was simpler and better.
* **Occluders need their own model.** Subtracting COCO person/ball masks from
  the board mask was the single most convincing feature in the demo.
* **Geometry is the remaining problem.** A per-frame homography is fine for
  flat boards; curved LED strips and fast pans need a board model that lives
  across frames.
* **Data was the ceiling.** ~150 labelled frames from three matches produce a
  model that is excellent on those matches and modest elsewhere. The business
  plan's 3 million frames were the right idea.
* **Keep the archive dated.** The `RD/YYYYMMDD` folders made this write-up
  possible; the `_claude` / `_chatgpt` file suffixes show how much of 2025's
  iteration was assistant-driven.
