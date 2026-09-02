# 07 · SAM3 for board detection and tracking (Sep 2026)

Goal of this chapter: replace per-frame Mask R-CNN detection with a
prompt-based foundation model. Plain Python scripts with paths at the top of
each file, no web app, no Docker. When something works here it moves into
`app/`.

Plan, one step at a time:

1. **Environment** — venv with PyTorch (CUDA 12.8) and the `sam3` package (this page).
2. **Does SAM3 find the boards?** — `sam3_image_probe.py`: text prompts on
   sampled frames of the three clips, overlays + contact sheets, tune the
   prompt and the confidence threshold.
3. **Tracking** — SAM3 video predictor: prompt once, propagate through the
   clip, stable ids, no per-frame detection.
4. **Replacement** — board-space rendering on top of the tracked masks
   (an ad half visible when the board is half visible), then occluders.

## 1. Environment (Windows, local venv outside OneDrive)

SAM3 needs Python 3.12+ and PyTorch 2.7+ with CUDA 12.6+. The venv lives in
`C:\Users\<you>\venvs\adswapai` so that OneDrive does not sync gigabytes of
packages.

```powershell
python -m venv C:\Users\$env:USERNAME\venvs\adswapai
C:\Users\$env:USERNAME\venvs\adswapai\Scripts\python.exe -m pip install --upgrade pip
C:\Users\$env:USERNAME\venvs\adswapai\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
C:\Users\$env:USERNAME\venvs\adswapai\Scripts\python.exe -m pip install "git+https://github.com/facebookresearch/sam3.git" "huggingface_hub[cli]"
C:\Users\$env:USERNAME\venvs\adswapai\Scripts\python.exe -m pip install einops pycocotools psutil triton-windows "numpy<2" "opencv-python-headless<5" pillow
```

Notes from the first install on Windows (Sep 2026):

* `sam3` imports `einops`, `pycocotools`, `psutil` and `triton` at package
  import time without declaring them. Triton has no official Windows build;
  the community `triton-windows` wheel (3.8) imports fine. Its kernels are
  compiled at first use and need the MSVC Build Tools (VS 2022) installed.
* `sam3` pins `numpy<2`, so OpenCV must stay on a 4.x wheel (OpenCV 5 pulls
  numpy 2).
* `pip install` of the git URL takes several minutes (clone + timm).

Checkpoints are gated on Hugging Face (SAM License):

1. Open https://huggingface.co/facebook/sam3 and accept the terms.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Log in once from your terminal (the token is stored in your user profile):
   `C:\Users\$env:USERNAME\venvs\adswapai\Scripts\hf.exe auth login`

Check everything (add `--download` to fetch the 3.4 GB checkpoint right away):

```powershell
cd experiments\07_sam3_sep2026
C:\Users\$env:USERNAME\venvs\adswapai\Scripts\python.exe check_env.py --download
```

Put the sample clips in `data/` (`1.mp4`, `2.mp4`, `3.mp4`, see `docs/assets.md`).

## 2. Image probe

```powershell
C:\Users\$env:USERNAME\venvs\adswapai\Scripts\python.exe sam3_image_probe.py
```

Look at `output/image_probe/<clip>_sheet.jpg` (rows = timestamps, columns =
prompts) and `summary.json`. Things to play with at the top of the script:
`PROMPTS`, `CONFIDENCE` (SAM3 default 0.5), `TIMESTAMPS`. What we want to
see before moving on: every visible board found, the far LED strip included,
no crowd/pitch/player false positives, stable scores across the four
timestamps.

### Findings (2 Sep 2026, 11 frames from the 3 clips, RTX 5070 Ti)

* Speed: backbone about 140 ms per 1080p frame at 1008 px, each extra text
  prompt about 55 ms. Model load 6 s, weights 3.4 GB, bf16 autocast needed.
* Prompt matters more than anything else. `sam3_prompt_sweep.py` ranking
  (frames with at least one hit / mean score at threshold 0.2):

  | prompt | hit frames | mean score | remarks |
  |--------|-----------|------------|---------|
  | sponsor banner | 11/11 | 0.46 | every board incl. far LED strips, some crowd flags |
  | advertisement | 11/11 | 0.43 | same coverage |
  | banner | 11/11 | 0.44 | slightly fewer boards |
  | advertising banner | 11/11 | 0.39 | misses far strips |
  | billboard | 9/11 | 0.30 | first choice, weak: misses wide shots |
  | LED perimeter board, perimeter advertising, advertising hoarding | 0/11 | - | nothing |

* Threshold: with "sponsor banner" the number of detections per frame is
  about 40 at 0.2, 27 at 0.3, 20 at 0.4, 15 at 0.5 (min 10). No mask larger
  than 8 % of the frame: the huge boxes in the overlays are the bounding boxes
  of long slanted strips, not crowd masks.
* The zoom crop (upper 5-60 % band) did not add recall; not needed.

<p align="center"><img src="../../docs/images/sam3_prompt_sponsor_banner_sheet.jpg" width="900"><br>
<sub>"sponsor banner", threshold 0.2: rows = clips 1/2/3, columns = 1/4/8/11 s. Every board is found, including the far LED strips in the wide shots.</sub></p>
* Remaining precision problems: overlapping duplicates (a board found as a
  whole and as its sponsor panels), the broadcaster score bug, crowd flags.
  Plan: threshold 0.3-0.4, mask-IoU de-duplication, a HUD exclusion zone.

## 3. Video tracking probe

```powershell
C:\Users\$env:USERNAME\venvs\adswapai\Scripts\python.exe sam3_video_probe.py
```

One text prompt on `START_FRAME`, then SAM3's video predictor propagates the
masks (`MAX_FRAMES` frames, forward). Output in `output/video_probe/`: a
diagnostic video (one colour per object id, id + probability, ms per frame),
a contact sheet and a stats JSON (unique ids, frames per id, fps). What we
want to see: each board keeps one id for the whole clip, ids do not swap or
multiply, masks stay tight when players pass in front, and a usable fps.

### Findings (clip 2, 150 frames, "sponsor banner", threshold 0.3, SAM3 video predictor)

* Clip 2 contains a camera cut at about frame 45 (close-up of a player until
  frame ~120, then back to the wide shot). The tracker correctly holds 11
  boards before the cut, tracks nothing during the close-up and re-detects
  the boards after it, re-using most of the old ids (ids 1, 3, 5-9 appear in
  both wide segments). Any pipeline needs a shot-cut detector anyway.
* One persistent false object: the broadcaster's top band (id 4, present in
  136 of 151 frames). A HUD exclusion zone fixes it.
* Speed is the problem: 0.24 fps overall. Median 0.9 s per frame, but the
  periodic re-detection frames take 10-60 s (p90 = 10.8 s) and VRAM peaks at
  15 GB with `offload_video_to_cpu=True`. The image model finds the same
  boards in 140 ms per frame.
* Next: the SAM 3.1 "multiplex" predictor (built for many objects), then, if
  still too slow, a hybrid: SAM3 image detection every frame + our own
  IoU/Kalman association.

### Hybrid tracker (`sam3_hybrid_track.py`, same clip and frames)

SAM3 image model on every frame + mask de-duplication (IoU > 0.6 or 80 %
containment), HUD exclusion (top 8 %), IoU association with a 5-frame hold,
histogram-based shot-cut detection.

| | SAM3 video predictor | hybrid |
|---|---|---|
| speed | 0.24 fps, spikes of 10-60 s | 3.4 fps, median 251 ms, no spikes |
| shot cut | survives it, re-uses most ids | detected at frames 46 and 112 (correct), ids reset |
| close-up segment | one persistent false object (top band) | 0 objects |
| id stability | 11 ids stable in the first segment | 9 boards stable for all 46 frames, but 56 ids in total: some panels flicker between "whole board" and "single sponsor panel" detections and spawn short-lived ids |

Speed favours the hybrid, mask/id consistency favours the video predictor.

<p align="center"><img src="../../docs/images/sam3_hybrid_tracking_clip2.jpg" width="640"><br>
<sub>Hybrid tracker on clip 2: frames 0, 25 (wide shot, 13-16 tracks), 50-100 (close-up after the cut, nothing), 125 (back to the wide shot, new ids).</sub></p>

### SAM 3.1 multiplex predictor (`VERSION = "sam3.1"`)

Two Windows-specific problems had to be worked around in `sam3_video_probe.py`:
the base predictor passes `offload_state_to_cpu` to the 3.1 model's
`init_state`, which rejects it (wrapped to drop the argument), and the 3.1
decoder pins scaled-dot-product attention to the flash kernel, which raises
"No available kernel" on this torch/Blackwell build (`sdpa_kernel` is
replaced by a permissive one). With that, the text prompt on frame 0 took
**0.85 s for 9 objects** (SAM3: 8.6 s), but the propagation then died
silently (no Python traceback, VRAM at 15.8 GB). Not measured yet; retry
with `max_num_objects` lowered and the video offloaded to CPU.

### Where we are (end of 2 Sep 2026)

* Detection: solved by prompt choice ("sponsor banner" / "advertisement",
  threshold 0.3-0.4), sharp masks, ~200 ms per frame.
* Tracking: the architecture is "detect on the first frame and on events
  (shot cut, new object), track in between". What fills the gap between
  detections is open: SAM 3.1 (quality, if it runs in 16 GB) or our own
  propagation (camera-motion homography + IoU association, fast). The
  homography is needed for step 4 anyway.
* Step 4 (board-space replacement) starts once the tracker choice is made.

## Access

The checkpoint repo is gated *with manual approval*: after accepting the terms
the request shows "awaiting a review from the repo authors" until Meta
approves it. `check_env.py` reports a 403 until then.
