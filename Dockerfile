FROM python:3.13-slim

# System libraries needed by opencv-python-headless and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install CPU-only PyTorch first to avoid pulling ~2.5 GB of CUDA libraries,
# then install the rest of the project's dependencies.
COPY pyproject.toml ./
RUN uv pip install --system --no-cache \
      --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision \
    && uv pip install --system --no-cache \
      pyiqa flask numpy piexif pillow requests scikit-learn \
      transformers opencv-python-headless mediapipe

# Pre-bake the MediaPipe face landmarker model (~3.6 MB) so startup works offline
RUN curl -fsSL -o /app/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

COPY config.py daily_highlights.py sync_album_to_pixtale.py sync_immich_album_previews.py upload_to_pixtale.py ./
COPY templates/ ./templates/
COPY static/ ./static/

ENV DATA_DIR=/data \
    PORT=5724 \
    FLASK_RELOADER=0 \
    FLASK_DEBUG=0 \
    HF_HOME=/data/.cache/huggingface \
    TORCH_HOME=/data/.cache/torch \
    PYTHONUNBUFFERED=1

EXPOSE 5724
VOLUME ["/data"]

CMD ["python", "daily_highlights.py"]
