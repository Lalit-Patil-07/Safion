# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --omit=dev

COPY frontend/ ./
RUN npm run build


# Stage 2: Backend runtime (CUDA)
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/backend

# System libraries required by OpenCV and InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source (layer cache)
COPY backend/requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Copy application source
COPY backend/ ./

# Copy compiled frontend into the location Flask serves static files from
COPY --from=frontend-builder /app/frontend/build /app/frontend/build

# Model weights — placed at /app/best.pt, consistent with MODEL_PATH=../best.pt
COPY best.pt /app/best.pt

# Database configuration is injected at runtime via environment variables:
#   DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
# No .env file is baked into the image.
# No PostgreSQL or pgvector is installed here — DB is external.

EXPOSE 5000

CMD ["gunicorn", "app:create_app()", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120"]