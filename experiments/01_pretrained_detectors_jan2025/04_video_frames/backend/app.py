"""AdSwapAI R&D, 2025-01-24: click-to-detect on video-frame snapshots (Flask backend, Mask R-CNN)."""
from flask import Flask, request, jsonify, send_from_directory
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

# COCO Classes List
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

# Flask App Setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Pretrained Mask R-CNN Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
model.eval()

@app.route('/upload', methods=['POST'])
def upload_snapshot():
    if 'image' not in request.files:
        logging.error("No image file found in request")
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    x = request.form.get('x')
    y = request.form.get('y')

    if not x or not y:
        logging.error("Coordinates not provided in the request")
        return jsonify({'error': 'Coordinates not provided'}), 400

    try:
        filename = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(file_path)

        logging.info(f"Image saved to {file_path} with coordinates: x={x}, y={y}")
        return jsonify({
            'message': 'Image saved successfully',
            'path': f"http://localhost:5000/uploads/{filename}",
            'coordinates': {'x': int(x), 'y': int(y)}
        })
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        return jsonify({'error': 'Failed to save the image'}), 500

@app.route('/select', methods=['POST'])
def select_object():
    logging.info("Processing object selection...")

    try:
        x = int(request.form.get('x'))
        y = int(request.form.get('y'))
        image_file = request.files.get('image')

        if not image_file:
            logging.error("No image file provided in the request")
            return jsonify({'error': 'No image provided'}), 400

        filename = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(file_path)

        logging.info(f"Processing uploaded image: {file_path} with coordinates x={x}, y={y}")

        image = Image.open(file_path).convert("RGB")
        img_tensor = [F.to_tensor(image).to(device)]

        with torch.no_grad():
            predictions = model(img_tensor)[0]

        masks = predictions['masks'] > 0.5
        boxes = predictions['boxes']
        scores = predictions['scores']
        labels = predictions['labels']

        selected_box = None
        selected_label = None

        for mask, box, score, label in zip(masks, boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            if x1 <= x <= x2 and y1 <= y <= y2 and score > 0.5:
                selected_box = [int(x1), int(y1), int(x2), int(y2)]
                selected_label = COCO_CLASSES[label]
                break

        if selected_box:
            logging.info(f"Object detected: {selected_label} at {selected_box}")
            return jsonify({
                'box': selected_box,
                'label': selected_label,
                'score': float(scores[0])
            })
        else:
            logging.warning("No object found at the clicked location.")
            return jsonify({'error': 'No object found at this location'}), 404
    except Exception as e:
        logging.error(f"Error processing object selection: {e}")
        return jsonify({'error': 'Failed to process object selection'}), 500

@app.route('/uploads/<filename>', methods=['GET'])
def serve_file(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
