# AdSwapAI R&D, 2025-01-28: video-frame detection backend image (Mask R-CNN, GPU).
# CUDA supported base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean

# Set working directory
WORKDIR /app

# Install PyTorch and specify compatible NumPy version
RUN pip install "numpy<2" \
    torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Run the Flask application
CMD ["python3", "app.py"]
