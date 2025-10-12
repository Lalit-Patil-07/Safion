# Stage 1: Build the React frontend
FROM node:18-alpine AS build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Setup the Python backend with NVIDIA CUDA 12.9.0 support
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and essential libraries
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY backend/requirements.txt .

# Install PyTorch, Torchvision, and Torchaudio from the specific CUDA 12.9 index
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Install the rest of the Python dependencies from the requirements file
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the backend application and model
COPY backend/app_server.py .
COPY best.pt .

# Copy the built frontend from the 'build' stage
COPY --from=build /app/frontend/build ./frontend/build

# Expose the port
EXPOSE 5000

# Command to run the application
CMD ["python3", "app_server.py"]
