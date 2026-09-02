# AdSwapAI R&D, 2025-02-01: MJPEG pipeline backend image (Mask R-CNN, OpenCV).
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libopencv-dev \
    cmake \
    git \
    tzdata \
    && apt-get clean

WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the entire application code
COPY . .

# Start the Flask application
CMD ["python3", "app.py"]
