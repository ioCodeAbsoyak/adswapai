"""AdSwapAI R&D, 2025-01-19: Mask R-CNN instance-mask overlay (Flask backend)."""
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import uuid
import logging
import os
import base64
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

# COCO class list
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

last_uploaded_image = None

# Flask app
app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    global last_uploaded_image
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    last_uploaded_image = file_path  # Save the path of the last uploaded file

    return jsonify({'message': 'Image uploaded successfully', 'path': file_path})

# Load the Mask R-CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
model.eval()

@app.route('/select', methods=['POST'])
def select_object():
    global last_uploaded_image
    logging.info("Processing object selection...")
    data = request.get_json()
    x, y = data.get('x'), data.get('y')  # Coordinates the user clicked

    if x is None or y is None:
        return jsonify({'error': 'Invalid coordinates provided'}), 400

    if last_uploaded_image is None:
        return jsonify({'error': 'No image uploaded yet'}), 400

    try:
        image = Image.open(last_uploaded_image).convert("RGB")
    except FileNotFoundError:
        logging.error(f"File not found: {last_uploaded_image}")
        return jsonify({'error': f'Image file {last_uploaded_image} not found'}), 404
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

    try:
        img_tensor = [F.to_tensor(image).to(device)]  # Wrapped in a list
        with torch.no_grad():
            predictions = model(img_tensor)[0]

        # Get the masks and boxes
        masks = predictions['masks'] > 0.5  # Boolean mask format
        boxes = predictions['boxes']
        scores = predictions['scores']
        labels = predictions['labels']

        # Find the mask closest to the point the user clicked
        selected_mask = None
        for mask, box, score, label in zip(masks, boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            if x1 <= x <= x2 and y1 <= y <= y2 and score > 0.5:
                selected_mask = base64.b64encode(np.array(mask.cpu().numpy(), dtype=np.uint8)).decode('utf-8')
                selected_label = int(label)
                selected_score = float(score)
                break

        if selected_mask:
            response = {
                'mask': selected_mask,
                'label': selected_label,
                'score': selected_score
            }
            logResponse = {
                'label': selected_label,
                'score': selected_score
            }
            logging.info(f"Response: {logResponse}")
            return jsonify(response)
        else:
            return jsonify({'error': 'No object found at this location'}), 404
    except Exception as e:
        logging.error(f"Error during object selection: {e}")
        return jsonify({'error': 'Failed to process the object selection'}), 500


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Backend is working'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

