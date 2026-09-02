"""AdSwapAI R&D, 2025-02-01: server-side per-frame Mask R-CNN detection streamed as MJPEG."""
# app.py
from flask import Flask, Response
from modules import detection, streaming
import cv2

app = Flask(__name__)

# Path to the raw video file (ensure you mount the videos folder correctly in docker-compose)
VIDEO_PATH = '/app/videos/raw/CarsMoving.mp4'

def process_frame(frame):
    """
    Process a frame: detect objects, filter for 'car' detections, and paint over the first car found.
    """
    detections = detection.detect_objects(frame)
    # Filter detections for 'car'
    car_detections = [d for d in detections if d['label'] == 'car']

    # Draw a rectangle for each car
    for car in car_detections:
        x1, y1, x2, y2 = car['box']
        # Draw the rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add the car label
        cv2.putText(frame, "Car", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route. Returns an MJPEG stream.
    """
    return Response(streaming.generate_video_stream(VIDEO_PATH, process_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start Flask on host 0.0.0.0 port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
