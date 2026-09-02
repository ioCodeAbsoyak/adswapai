"""AdSwapAI R&D, 2025-04-03: SuperPoint/SuperGlue frame-to-frame matching probe for board
tracking, kept as a documented dead end. Requires the SuperGluePretrainedNetwork repo
(see SUPERGLUE_REPO below) to be cloned next to this script."""

import torch
import cv2
import numpy as np

# Local clone of https://github.com/magicleap/SuperGluePretrainedNetwork, expected next to this script
SUPERGLUE_REPO = "SuperGluePretrainedNetwork"

from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue

VIDEO_PATH = "data/adVideo1.mp4"   # sample clip, see repo docs/assets.md

# Load the video
video_path = VIDEO_PATH
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read frame from video")
    exit()

# Load SuperPoint and SuperGlue models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

superpoint_config = {
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 1024
}
superpoint = SuperPoint(config=superpoint_config).to(device).eval()

superglue_config = {
    'weights': 'outdoor',
    'sinkhorn_iterations': 20,
    'match_threshold': 0.2,
}
superglue = SuperGlue(config=superglue_config).to(device).eval()

# Define a polygon for the object you want to track
polygon = np.array([[584, 314], [722, 348], [722, 390], [584, 348]], dtype=np.int32)

# Initialize object tracking variables
tracker = cv2.TrackerCSRT_create()
init_bbox = cv2.boundingRect(polygon)
ok = tracker.init(frame, init_bbox)

# Function to properly preprocess image for SuperPoint
def preprocess_image(img):
    # Convert to grayscale
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Normalize to [0,1] and convert to tensor
    img_normalized = gray.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).float()
    
    # Add batch and channel dimensions
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    
    # Move to device
    img_tensor = img_tensor.to(device)
    
    return {'image': img_tensor}, img_normalized

prev_frame = None
prev_frame_tensor = None
prev_keypoints = None
prev_descriptors = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Prepare the image for SuperPoint
    input_data, frame_normalized = preprocess_image(frame)
    frame_tensor = input_data['image']
    
    # Extract features with SuperPoint
    with torch.no_grad():
        pred = superpoint(input_data)
        
    # Get keypoints and descriptors
    keypoints = pred['keypoints'][0].cpu().numpy()
    scores = pred['scores'][0].cpu().numpy()
    descriptors = pred['descriptors'][0].cpu().numpy()
    
    # Prepare inputs for SuperGlue
    if prev_frame is not None and prev_keypoints is not None:
        # Create SuperGlue input
        superglue_input = {
            'image0': prev_frame_tensor,  # Previous frame tensor
            'image1': frame_tensor,        # Current frame tensor
            'keypoints0': torch.from_numpy(prev_keypoints)[None].to(device),
            'keypoints1': torch.from_numpy(keypoints)[None].to(device),
            'descriptors0': torch.from_numpy(prev_descriptors)[None].to(device),
            'descriptors1': torch.from_numpy(descriptors)[None].to(device),
            'scores0': torch.from_numpy(scores)[None].to(device),
            'scores1': torch.from_numpy(scores)[None].to(device),
        }
        
        # Match features with SuperGlue
        with torch.no_grad():
            matches = superglue(superglue_input)
            
        # Extract matches
        matches0 = matches['matches0'][0].cpu().numpy()
        
        # Get valid matches
        valid = matches0 > -1
        
        if np.sum(valid) > 0:
            mkpts0 = prev_keypoints[valid]
            mkpts1 = keypoints[matches0[valid]]
            
            # Check if any matched points are inside the polygon
            points_inside = []
            for pt in mkpts1:
                if cv2.pointPolygonTest(polygon, tuple(pt), False) >= 0:
                    points_inside.append(pt)
                    
            # Update polygon if enough points are found
            if len(points_inside) >= 4:
                points_inside = np.array(points_inside, dtype=np.int32)
                # Get new bounding box
                init_bbox = cv2.boundingRect(points_inside)
                tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(frame, init_bbox)
            
    # Update tracker
    ok, bbox = tracker.update(frame)
    
    # Draw results
    if ok:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Draw keypoints
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
    
    # Draw polygon
    cv2.polylines(frame, [polygon], True, (255, 0, 0), 2)
    
    # Store current features for next iteration
    prev_frame = frame.copy()
    prev_frame_tensor = frame_tensor
    prev_keypoints = keypoints
    prev_descriptors = descriptors
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()