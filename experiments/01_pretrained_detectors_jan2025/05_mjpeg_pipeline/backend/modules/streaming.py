"""AdSwapAI R&D, 2025-02-01: MJPEG frame generator with FPS overlay."""
# modules/streaming.py
import cv2
import time

def generate_video_stream(video_path, process_frame_callback):
    """
    Read the given video file, process each frame with the process_frame_callback
    function, overlay FPS info on the frame, and yield it as a JPEG MJPEG stream.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file: " + video_path)

    prev_time = time.time()  # Time before the first frame

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop back to the start when the video ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed_frame = process_frame_callback(frame)

        # FPS calculation:
        current_time = time.time()
        elapsed = current_time - prev_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        prev_time = current_time

        # Write the FPS info in the top-left corner:
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
