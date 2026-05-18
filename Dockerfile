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

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files — uv needs both to resolve the lock
COPY pyproject.toml uv.lock ./

# Install into a virtual env at /opt/venv
# --frozen            → respect uv.lock exactly, no re-resolution
# --no-install-project → skip installing the project package itself (no src layout needed)
RUN uv sync --frozen --no-install-project --no-dev \
    --python /usr/local/bin/python3 \
    --link-mode copy \
    --compile-bytecode \
    --venv /opt/venv


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

COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]