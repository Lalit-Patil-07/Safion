# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --omit=dev

COPY frontend/ ./
RUN npm run build


# Stage 2: Backend (CUDA runtime)
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

# Install Python dependencies before copying source so this layer is cached
COPY backend/requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy application source
COPY backend/ ./

# Copy compiled frontend into the location Flask serves static files from
COPY --from=frontend-builder /app/frontend/build ./static

# Copy model weights
COPY best.pt ../

EXPOSE 5000

CMD ["python3", "run.py"]