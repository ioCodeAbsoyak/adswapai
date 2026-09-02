"""AdSwapAI R&D, 2025-02-01: Mask R-CNN detection helper for the MJPEG pipeline."""
# modules/detection.py
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np

# Set up CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights).to(device)
model.eval()




# COCO classes list (index corresponds to label id)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def detect_objects(frame, threshold=0.7):
    """
    Runs object detection on a frame.
    Returns a list of detections with bounding box, score, and label.
    """
    # Convert OpenCV BGR image to PIL RGB image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor_img = F.to_tensor(image).to(device)
    with torch.no_grad():
        predictions = model([tensor_img])[0]
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    results = []
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            results.append({
                'box': box.astype(int).tolist(),
                'score': float(score),
                'label': COCO_CLASSES[label]
            })
    return results
