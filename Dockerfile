# ─────────────────────────────────────────────────────────────────────────
# CineCut AI Studio — RunPod Serverless image
# CUDA-enabled base (not python:slim) so Real-ESRGAN / faster-whisper /
# Demucs / rembg all get real GPU acceleration on RunPod's workers.
# ─────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.10 + ffmpeg + build tools
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    ffmpeg git build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install GPU-enabled PyTorch first (CUDA 12.1 wheels), including
# torchaudio in the SAME command — installing it separately via
# requirements.txt would pull a mismatched/CPU build from PyPI's default
# index and silently downgrade torch to satisfy it.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ────────────────────────────────────────────────────
COPY server.py .
COPY ai_engine.py .
COPY esrgan_engine.py .
COPY bg_removal_engine.py .
COPY caption_engine.py .
COPY text_correction.py .
COPY handler.py .

# Model weights (Real-ESRGAN checkpoint) + RIFE code (kept for future use)
COPY weights/ ./weights/
COPY rife/ ./rife/

# Frontend files (kept in the image so the same container can still be run
# as a normal FastAPI server locally/on Render if ever needed; RunPod
# Serverless itself only calls handler.py, not these).
COPY index.html .
COPY app.js .
COPY styles.css .

# RunPod Serverless invokes this file directly as the worker entrypoint.
# (For local/Render use, run: uvicorn server:app --host 0.0.0.0 --port 10000)
CMD ["python", "-u", "handler.py"]
