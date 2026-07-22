FROM python:3.10-slim

# Install FFmpeg and system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server.py .

# Expose Hugging Face Space port 7860
EXPOSE 7860

# Run FastAPI server on port 7860
CMD ["uvicorn", "server.py:app", "--host", "0.0.0.0", "--port", "7860"]
