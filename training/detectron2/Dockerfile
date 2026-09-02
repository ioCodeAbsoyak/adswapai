# AdSwapAI R&D, 2025-04-18: training container for the Detectron2 billboard model (also used for the RAFT experiments)
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# 1) System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip python3-dev git build-essential cmake \
      libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# 2) Nightly PyTorch with CUDA 12.8 (the only build that supported the RTX 50xx GPU in April 2025)
RUN pip3 install --upgrade pip
RUN pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 3) Detectron2 dependencies
RUN pip3 install pyyaml>=5.1 shapely fvcore iopath opencv-python-headless

# 4) Extra dependencies for RAFT optical flow
RUN pip3 install scipy imageio matplotlib Pillow

# 5) Detectron2 from source
RUN git clone https://github.com/facebookresearch/detectron2.git /detectron2
WORKDIR /detectron2
RUN pip3 install -e .

# 6) Back to the working directory (mount this folder there)
WORKDIR /workspace
