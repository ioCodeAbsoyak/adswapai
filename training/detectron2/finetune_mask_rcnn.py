import os
import random
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator

setup_logger()

# 1. Register dataset
register_coco_instances(
    "billboard_train", {}, 
    "dataset/converted_dataset_real.json",
    "dataset/images"
)

# 2. Configure training settings
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)

# Dataset configuration
cfg.DATASETS.TRAIN = ("billboard_train",)
cfg.DATASETS.TEST = ("billboard_val",)  # Set to empty list if no validation set
cfg.DATALOADER.NUM_WORKERS = 4

# Load pre-trained model
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (billboard)

# Training parameters
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001  # Lower learning rate for fine-tuning
cfg.SOLVER.MAX_ITER = 10000  # Adjust based on dataset size
cfg.SOLVER.CHECKPOINT_PERIOD = 500

# Use step learning rate schedule
cfg.SOLVER.STEPS = [6000, 8000]  # Reduce LR at these iterations
cfg.SOLVER.GAMMA = 0.1  # LR reduction factor

# Specify output directory
cfg.OUTPUT_DIR = "output_improved_finetune"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Mixed precision training for speed
cfg.SOLVER.AMP.ENABLED = True

# Add more robust augmentations
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            is_train=True,
            augmentations=[
                T.RandomFlip(prob=0.5, horizontal=True),
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
                # Resize with multiple scales to improve detection at different distances
                T.ResizeShortestEdge(
                    short_edge_length=(640, 672, 704, 736, 768, 800),
                    max_size=1333, 
                    sample_style='choice'
                ),
                # Random crop to help focus on detail
                T.RandomCrop("relative_range", (0.8, 1.0)),
            ],
            image_format="BGR",
            use_instance_mask=True
        )
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def main():
    # Set seeds for reproducibility
    random.seed(42)
    
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()