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

# Install GPU-enabled PyTorch first (CUDA 12.8 wheels), including
# torchaudio in the SAME command — installing it separately via
# requirements.txt would pull a mismatched/CPU build from PyPI's default
# index and silently downgrade torch to satisfy it.
# NOTE: cu121 wheels only ship kernels for sm_50..sm_90 (no Blackwell
# support), which crashes with "no kernel image is available for
# execution on the device" on RunPod GPUs like the RTX PRO 6000 Blackwell
# (CUDA capability sm_120). cu128 wheels include Blackwell kernels while
# still running fine on older Ampere/Ada/Hopper cards.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download every AI model's weights at BUILD time so RunPod workers never
# fetch them over the network at runtime. Without this, each model is lazy-
# loaded on first use, and since this endpoint scales workers down to ZERO
# when idle (that's the whole point of Serverless), every single cold start
# would have to re-download rembg's ~176MB matting model, faster-whisper's
# ~1.5GB medium model, and Demucs's htdemucs_6s + htdemucs_ft checkpoints
# (~500MB+ each) from GitHub/HuggingFace before doing any real work, turning
# "scales up in seconds" into "waits several minutes per cold start" and
# making every request depend on those external hosts being fast and up.
# Baking the weights into the image means a cold start only pays for
# container boot; the actual model files ship as image layers.
RUN python -c "from rembg import new_session; new_session('isnet-general-use')" \
    && python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')" \
        && python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s'); get_model('htdemucs_ft')"

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
