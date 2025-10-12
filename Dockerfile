# Stage 1: Build the React frontend
FROM node:18-alpine AS build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Setup the Python backend and serve the frontend
FROM python:3.9-slim
WORKDIR /app

# Install system dependencies required for building and running
# - build-essential & cmake: For building dlib
# - libgl1: Headless OpenGL dependency for OpenCV
# - libglib2.0-0: Provides libgthread, a runtime dependency for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the backend code and requirements
# Note: We copy into a 'backend' subdirectory to keep paths consistent
COPY backend/ ./backend/
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# Copy the built frontend from the build stage
# The static folder in app_server.py is '../build', so this path is correct
COPY --from=build /app/frontend/build ../build

# Copy the model into the backend directory where app_server.py expects it
COPY best.pt .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app_server.py"]