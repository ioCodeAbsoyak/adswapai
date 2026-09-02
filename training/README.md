# Training data and models

Three labelling rounds and two trained models. All frames come from three
football clips (Turkish top-flight broadcasts, 1080p), extracted with OpenCV.

```
training/
├── dataset_v1/        Roboflow export, YOLO-seg format (in git)
│   ├── data.yaml, train/ valid/ test/       1 class "billboard", 21 images + 21 label files
│   ├── train_yolov8_seg.py                  YOLOv8s-seg fine-tune -> best.pt (Apr 2025)
│   └── predict_yolov8_seg.py                visual check on a clip
├── dataset_v3/        VGG Image Annotator (VIA) polygons (csv in git, frames in the archive)
│   ├── via_annotations.csv                  153 frames, one polyline per board group
│   └── csv_to_cocojson.py                   VIA csv -> COCO json
└── detectron2/        Mask R-CNN training (json in git, frames and weights in the archive)
    ├── dataset/_annotations.coco.json       Roboflow COCO export: 100 images, 523 boards
    ├── dataset/converted_dataset_real.json  VIA converted: 150 images, 152 polygons
    ├── dataset/annotationsFinal.json        merged training set: 153 images, 675 annotations
    ├── merge_coco_jsons.py                  merges two COCO files (remaps ids)
    ├── train_mask_rcnn.py                   the run that produced model_final.pth
    ├── finetune_mask_rcnn.py                a longer fine-tune attempt (not used)
    └── Dockerfile                           CUDA 12.8 nightly-torch training container
```

## Round 1: Roboflow, YOLO-seg (early April 2025)

Frames were uploaded to a Roboflow workspace ("altervision") and boards were
drawn as polygons; the export in `dataset_v1` is the YOLO segmentation format
(`class x1 y1 x2 y2 …` normalised). `dataset_v2` in the archive was a
re-export with the same script and is not duplicated here.
`train_yolov8_seg.py` fine-tunes `yolov8s-seg.pt` for 50 epochs at 640 px and
gives `best.pt` (24 MB, Ultralytics 8.3.105), the model behind
`experiments/03_polygon_tracking_apr2025`.

Result: boards are found, but masks are coarse and the model flickers between
frames, which is what pushed the project to Mask R-CNN.

## Round 2: VIA polygons and Detectron2 (18 – 29 April 2025)

153 frames were annotated again with the VGG Image Annotator as polylines
(`dataset_v3/via_annotations.csv`, category "pitch side billboards"). The csv
was converted to COCO (`csv_to_cocojson.py`), merged with Roboflow's COCO
export (`merge_coco_jsons.py`) and the union, `annotationsFinal.json`,
became the training set. CVAT was also installed (the archive has a clone)
and evaluated for labelling at scale, but the small set was finished in VIA.

`train_mask_rcnn.py` fine-tunes Detectron2's `mask_rcnn_R_50_FPN_3x` from
the COCO checkpoint: 5000 iterations, batch 4, LR 0.001, AMP, flips, colour
jitter and multi-scale resize, one class. The run of 29 April 2025 (folder
`outputNew` in the archive, MD5 `7820c792…`) produced the
`model_final.pth` that `app/` still uses. `finetune_mask_rcnn.py` was a
second, longer schedule (10 000 iterations with LR steps and random crops)
started from an earlier checkpoint; it was interrupted at 7 500 iterations
and its weights were not adopted.

Result: clean instance masks on the three clips, good enough for the demo.
With ~150 frames the model is tuned to those matches; the business plan's
3 million labelled frames were the intended next step.

## Reproducing

```bash
# Detectron2 (needs the frames from the archive in detectron2/dataset/images/)
cd training/detectron2
docker build -t adswapai-train .
docker run --gpus all -v "$PWD:/workspace" adswapai-train python3 train_mask_rcnn.py

# YOLOv8-seg
cd training/dataset_v1
pip install ultralytics && python train_yolov8_seg.py
```

Where the frames and weights live: [`../docs/assets.md`](../docs/assets.md).
