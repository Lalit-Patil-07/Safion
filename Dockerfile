# ==========================================
# STAGE 1: Frontend Builder
# ==========================================
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --omit=dev

COPY frontend/ ./
RUN npm run build


# ==========================================
# STAGE 2: Backend Builder
# ==========================================
FROM python:3.12-slim-bookworm AS backend-builder
WORKDIR /app/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128


# ==========================================
# STAGE 3: Final Runtime Image
# ==========================================
FROM python:3.12-slim-bookworm
WORKDIR /app/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/ ./
COPY --from=frontend-builder /app/frontend /app/frontend
COPY best.pt /app/best.pt

# ADDED: entrypoint handles .env loading, DB readiness, and Gunicorn startup
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# CHANGED: was `CMD ["python3", "run.py"]`
CMD ["/app/entrypoint.sh"]