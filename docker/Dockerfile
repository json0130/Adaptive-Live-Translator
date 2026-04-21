# ── Build stage ──────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

# ── Dependencies ─────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── App ──────────────────────────────────────────────────────────────────────
COPY . .
RUN pip install -e . --no-deps

# Create dirs expected at runtime
RUN mkdir -p models data/glossaries data/translation_memory \
             data/speaker_profiles data/lora_adapters \
             logs reports

# ── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8000

ENV CONFIG_PATH=configs/default.yaml \
    MODEL_CACHE_DIR=/app/models \
    LOG_LEVEL=INFO

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
