"""AdSwapAI R&D, 2025-03-10: first attempt at removing cars - Mask R-CNN masks + cv2.inpaint, writes a video."""

import torch
import torchvision
import cv2
import numpy as np
import time

# put the input files next to this script or pass a path
DEFAULT_VIDEO = "data/CarsMoving.mp4"

# Load pre-trained Mask R-CNN model
weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights).cuda()
model.eval()

# Load video
video_path = DEFAULT_VIDEO
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file.")
    exit()

# COCO labels (Cars = 3)
CAR_CLASS_ID = 3

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_no_cars.mp4", fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torchvision.transforms.functional.to_tensor(frame_rgb).cuda()

    with torch.no_grad():
        detections = model([frame_tensor])[0]

    car_indices = [idx for idx, label in enumerate(detections["labels"]) if label.item() == CAR_CLASS_ID and detections["scores"][idx] > 0.7]

    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for idx in car_indices:
        mask = detections["masks"][idx, 0].cpu().numpy()
        mask_binary = (mask > 0.5).astype(np.uint8)
        combined_mask = cv2.bitwise_or(combined_mask, mask_binary)

    # Inpaint cars out of the frame
    frame_inpainted = cv2.inpaint(frame, combined_mask, 3, cv2.INPAINT_TELEA)

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    cv2.putText(frame_inpainted, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("No Cars (Inpainted)", frame_inpainted)
    out.write(frame_inpainted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
