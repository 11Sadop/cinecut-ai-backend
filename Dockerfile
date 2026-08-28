# ─────────────────────────────────────────────────────────────────────────
# CineCut AI Studio — RunPod Serverless image
# CUDA-enabled base (not python:slim) so Real-ESRGAN / faster-whisper /
# Demucs / rembg all get real GPU acceleration on RunPod's workers.
# ─────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ROOT CAUSE FOUND (real RunPod job logs): every AI-upscale job was
# silently falling back to CPU libx264 encoding at full 4K instead of
# hardware h264_nvenc -- _test_nvenc_available() in ai_engine.py kept
# returning False on every worker. The nvidia/cuda base image only
# advertises "compute,utility" driver capabilities by default, which is
# enough for PyTorch/CUDA compute (that's why ESRGAN itself correctly
# loaded on CUDA) but NOT enough to expose the NVENC hardware video
# encode block inside the container -- that needs the "video"
# capability explicitly requested, or ffmpeg's nvenc init call fails
# even though the GPU has working NVENC hardware. CPU-encoding a full
# real-time 4K stream is expensive on its own, and was very likely the
# actual dominant cost this whole time (not the minterpolate filter
# mode, which barely moved observed per-frame timing across 3 rounds of
# tuning it -- 3.1-3.4s/frame stayed roughly constant regardless).
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Python 3.10 + build tools (ffmpeg installed separately below --
# ROUND 24 ROOT CAUSE: apt's ffmpeg package is a stock Ubuntu build
# with NO nvenc encoder compiled into it at all -- no env var or
# NVIDIA_DRIVER_CAPABILITIES setting could ever have made -c:v
# h264_nvenc work through it. This is the actual reason every
# upscale job silently, permanently fell back to slow CPU libx264:
# the hardware encoder never existed in the binary in the first
# place, confirmed live via a real RunPod job's NVENC probe log
# ("Real-ESRGAN loaded on CUDA" -- proving the GPU/driver itself is
# fine -- immediately followed by "NVENC probe failed").
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    git build-essential curl xz-utils \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Static NVENC/CUDA-enabled ffmpeg build (BtbN/FFmpeg-Builds, GPL
# linux64-gpl release), installed AT /usr/bin/ffmpeg -- the exact
# path _find_ffmpeg() in ai_engine.py already checks first, so no
# Python code change is needed; this just makes that path finally
# point at a binary that actually has h264_nvenc built in.
RUN curl -L -o /tmp/ffmpeg.tar.xz \
    https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linux64-gpl.tar.xz \
    && mkdir -p /tmp/ffmpeg-extract \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg-extract --strip-components=1 \
    && cp /tmp/ffmpeg-extract/bin/ffmpeg /usr/bin/ffmpeg \
    && cp /tmp/ffmpeg-extract/bin/ffprobe /usr/bin/ffprobe \
    && chmod +x /usr/bin/ffmpeg /usr/bin/ffprobe \
    && rm -rf /tmp/ffmpeg.tar.xz /tmp/ffmpeg-extract

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
# large-v3 model (matches what get_whisper_model() loads at runtime on CUDA), and Demucs's htdemucs_6s + htdemucs_ft checkpoints
# (~500MB+ each) from GitHub/HuggingFace before doing any real work, turning
# "scales up in seconds" into "waits several minutes per cold start" and
# making every request depend on those external hosts being fast and up.
# Baking the weights into the image means a cold start only pays for
# container boot; the actual model files ship as image layers.
RUN python -c "from rembg import new_session; new_session('isnet-general-use')" \
    && python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')" \
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
