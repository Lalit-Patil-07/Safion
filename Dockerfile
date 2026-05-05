# ==========================================
# STAGE 1: Frontend Builder
# ==========================================
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy package files and install strictly from the synced lockfile
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --omit=dev

# Copy the rest of the frontend source and build the static files
COPY frontend/ ./
RUN npm run build


# ==========================================
# STAGE 2: Backend Builder (The Heavy Lifter)
# ==========================================
FROM python:3.12-slim-bookworm AS backend-builder
WORKDIR /app/backend

# Install C++ compilers and Python dev headers required to compile InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment (Fixes the Ubuntu 24.04 PEP 668 installation block)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies into the venv
# Note: PyTorch pulls its own CUDA runtime, so we don't need a massive Nvidia base image
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128


# ==========================================
# STAGE 3: Final Runtime Image (Slim & Clean)
# ==========================================
FROM python:3.12-slim-bookworm
WORKDIR /app/backend

# Install ONLY the bare-minimum runtime libraries required by OpenCV and InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled Python virtual environment from the backend-builder stage
COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---- NEW LINES ADDED HERE ----
# Create the scripts directory and copy ONLY the setup_db.sh script
RUN mkdir -p /app/scripts
COPY scripts/setup_db.sh /app/scripts/
RUN chmod +x /app/scripts/setup_db.sh
# ------------------------------

# Copy the backend source code
COPY backend/ ./

# Copy the compiled static frontend files from the frontend-builder stage
COPY --from=frontend-builder /app/frontend /app/frontend

# Expose the port your backend runs on
EXPOSE 5000

# Start the application
CMD ["python3", "run.py"]