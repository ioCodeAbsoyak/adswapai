"""AdSwapAI R&D, 2025-04-28: train the single-class billboard Mask R-CNN (Detectron2, R50-FPN-3x).

This is the script that produced model_final.pth, the model used by the web app.
Dataset: dataset/annotationsFinal.json (COCO format, 153 frames, 675 polygons,
one category) with the frames in dataset/images/ (see docs/assets.md).
"""
import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T

# 1. Register the dataset
register_coco_instances(
    "ads_train", {},
    "dataset/annotationsFinal.json",  # merged COCO annotations
    "dataset/images"                   # frames
)

# 2. Config
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)

cfg.DATASETS.TRAIN = ("ads_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # start from COCO
cfg.SOLVER.IMS_PER_BATCH = 4   # fits a 16 GB GPU
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 5000     # enough for ~150 images
cfg.SOLVER.STEPS = []          # no LR decay steps
cfg.SOLVER.AMP.ENABLED = True  # mixed precision

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # one class: billboard

cfg.OUTPUT_DIR = "outputNew"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 3. Augmentation
def custom_mapper(dataset_dict):
    mapper = DatasetMapper(
        is_train=True,
        augmentations=[
            T.RandomFlip(horizontal=True),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomLighting(0.7),
            T.ResizeShortestEdge(short_edge_length=(512, 640, 720), max_size=1280, sample_style='choice'),
        ],
        image_format="BGR",
        use_instance_mask=True
    )
    return mapper(dataset_dict)

# 4. Trainer with the custom loader
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

# 5. Train
def main():
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)  # start from the COCO weights
    trainer.train()

if __name__ == "__main__":
    main()
