"""AdSwapAI R&D, 2025-01-18: click-to-detect with a pretrained Faster R-CNN (Flask backend)."""
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import uuid
import logging
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Load the Faster R-CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1  # Weights pretrained on the COCO dataset
model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
model.eval()  # Switch the model to evaluation mode

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
        return jsonify({'error': f'Image file {last_uploaded_image} not found'}), 404

    img_tensor = F.to_tensor(image).to(device)

    # Run inference with the model
    predictions = model([img_tensor])[0]

    # Get the bounding boxes and classes
    boxes = predictions['boxes'].tolist()  # Bounding boxes
    labels = predictions['labels'].tolist()  # Class labels
    scores = predictions['scores'].tolist()  # Confidence scores

    # Find the box closest to the point the user clicked
    selected_box = None
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        if x1 <= x <= x2 and y1 <= y <= y2 and score > 0.5:  # Confidence score > 0.5
            selected_box = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": COCO_CLASSES[int(label)],  # Human-readable class name
                "score": float(score)
            }
            break

    # Return the results
    if selected_box:
        return jsonify({'bounding_box': selected_box})
    else:
        return jsonify({'error': 'No object found at this location'}), 404


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Backend is working'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

