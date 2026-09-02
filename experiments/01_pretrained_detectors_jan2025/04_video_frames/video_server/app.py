"""AdSwapAI R&D, 2025-01-24: small Flask server serving video files for the frontend player."""
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Directory holding the video files
VIDEO_FOLDER = './videos'
os.makedirs(VIDEO_FOLDER, exist_ok=True)

@app.route('/videos', methods=['GET'])
def list_videos():
    """Return all video files in the directory as JSON."""
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.mkv', '.avi'))]
    return jsonify({'videos': videos})

@app.route('/videos/<filename>', methods=['GET'])
def stream_video(filename):
    """Stream the given video file."""
    return send_from_directory(VIDEO_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload a new video file."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video = request.files['video']
    filepath = os.path.join(VIDEO_FOLDER, video.filename)
    video.save(filepath)
    return jsonify({'message': f'{video.filename} uploaded successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
