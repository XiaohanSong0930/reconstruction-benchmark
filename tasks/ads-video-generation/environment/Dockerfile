FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 \
    torchvision==0.18.1
RUN python -m pip install \
    openai==1.84.0 \
    google-genai==1.16.1 \
    pillow==10.4.0 \
    timm==1.0.15 \
    ftfy==6.3.1 \
    regex==2024.11.6 \
    transformers==4.46.3 \
    pytest==8.3.5

RUN git clone https://github.com/microsoft/VideoX.git /opt/VideoX \
    && mkdir -p /opt/VideoX/checkpoints \
    && curl -L https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/zero.pth -o /opt/VideoX/checkpoints/zero.pth

ENV PYTHONPATH=/workspace \
    XCLIP_REPO=/opt/VideoX \
    XCLIP_CHECKPOINT=/opt/VideoX/checkpoints/zero.pth \
    XCLIP_ARCH=ViT-B/16
