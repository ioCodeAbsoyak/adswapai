# Large assets (not in git)

Model weights, datasets with images and videos are excluded by `.gitignore`
(GitHub rejects files above 100 MB and the repo would be several GB
otherwise). This page lists what exists, where it lives in the private archive
and where it must be placed to run the code.

| Asset | Size | Archive location | Place at |
|-------|------|------------------|----------|
| Custom billboard Mask R-CNN (`model_final.pth`, Detectron2 R50-FPN, 1 class) | 351 MB | `RD/20250430/model_final.pth` (same file as in `RD/2025052x/backend`) | `app/backend/model_final.pth` |
| COCO Mask R-CNN weights (`model_final_f10217.pkl`) | 170 MB | downloaded at Docker build time from the Detectron2 model zoo | `app/backend/models/` (automatic) |
| Sample clips `1.mp4`, `2.mp4`, `3.mp4` (1080p30, 14-20 s) | 35 / 24 / 54 MB | `RD/20260902/frontend/static/sampleVideos/` | `app/frontend/static/sampleVideos/` |
| Detectron2 training images (158 frames + COCO json) | 170 MB | `RD/20250418/dataset/` | `training/detectron2/dataset/images/` |
| VIA labelling images (dataset v3, 153 frames) | 170 MB | `RD/dataset_v3/images/` | `training/dataset_v3/images/` |
| YOLO-seg experiments weights (`yolov8*-seg.pt`, `best.pt`) | 7-144 MB each | `RD/20250403/`, `RD/dataset_v1/` | next to the script that loads them |
| Training checkpoints (`output*/model_*.pth`, tfevents) | 350 MB each | `RD/20250418/output*/` | not needed |
| Extracted ad frames (722 jpg) | 194 MB | `RD/Ads/` | not needed |
| Raw source clips and stills | 62 MB | `RD/Orjinal/` | not needed |

Convention for the experiment scripts: each script expects its inputs in a
`data/` folder next to it (`data/adVideo1.mp4`, `data/replace.jpg`,
`data/ads/*.jpg`, …) and writes to `output/`. Both folders are ignored by
git. The clips named `adVideo1.mp4` / `adVideo2.mp4` in the 2025 scripts are
the same files as the web app's `2.mp4` / `1.mp4`.

The small assets that *are* in git: YOLO dataset v1 (31 images, 4 MB), all
annotation files (VIA csv, COCO json), the ad images used by the demo
(`app/frontend/static/images/`), thumbnails, concept art and the before/after
frames in `docs/images/`.

If you want the model in the repository anyway, use Git LFS:

```bash
git lfs install
git lfs track "app/backend/model_final.pth"
git add .gitattributes app/backend/model_final.pth
```
