# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker (Recommended)
- Copy environment: `cp .env.example .env` (edit secrets if needed)
- Start services: `docker compose up -d`
- Access: http://localhost:5000

### Local Development
#### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 18 (with pgvector extension)
- NVIDIA CUDA Toolkit 12.8 (for GPU acceleration)

#### Setup
1. Install PostgreSQL 18 and pgvector:
   ```bash
   sudo bash scripts/install_postgres.sh
   ```
   Verify extension: `sudo -u postgres psql -c "SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';"`

2. Configure environment:
   ```bash
   cp .env.example .env
   ```

3. Create database:
   ```bash
   bash scripts/setup_db.sh
   ```

4. Install Python dependencies:
   - CPU-only: `python -m venv venv && source venv/bin/activate && pip install -r backend/requirements.txt`
   - GPU (CUDA 12.8): `python -m venv venv && source venv/bin/activate && pip install -r backend/requirements.txt --index-url https://download.pytorch.org/whl/cu128`

5. Install frontend dependencies:
   ```bash
   cd frontend && npm install
   ```

#### Running
- Backend: `python backend/run.py` (from project root)
- Frontend: `cd frontend && npm start` (separate terminal)

#### Testing
- Run backend tests: `cd backend && pytest`
- Run frontend tests: `cd frontend && npm test`

## Code Architecture

### Backend (Python/Flask)
- `app.py`: Flask application factory, registers extensions, blueprints, and services.
- `config.py`: Configuration loaded from environment variables.
- `extensions.py`: Flask extensions initialization (db, bcrypt, jwt).
- `auth/`: JWT authentication routes and models.
- `detection/`: YOLOv11m object detection service (`yolo_service.py`) and detection-association logic.
- `face/`: InsightFace pipeline for face recognition, identity clustering, and routes.
- `streams/`: Stream management (RTSP/webcam/video files), worker threads, and IoU tracking.
- `tasks/`: Async task queue for violation processing.
- `violations/`: Violation logging and retrieval routes.
- `version.py`: Version information.

### Frontend (React)
- `src/App.js`: Main React application.
- Uses Tailwind CSS for styling, Lucide React for icons.
- Communicates with backend via REST APIs and Socket.IO for real-time updates.

### Database
- PostgreSQL 18 with pgvector extension for vector similarity search (used in face recognition).
- Tables managed via Flask-SQLAlchemy models in respective modules (auth, face, violations, etc.).

### Key Services
- `YOLOService`: Loads YOLOv11m model, performs object detection on video frames.
- `InsightFacePipeline`: Handles face detection and embedding generation using InsightFace (buffalo_l model).
- `StreamManager`: Manages video stream workers, each processing frames from a source.
- `TaskQueue`: Async queue (using Redis or similar?) for offloading violation processing (face recognition, logging).

Note: The actual task queue implementation may be in `tasks/queue.py`.

## Common Tasks
- Adding a new video stream: Use the frontend settings page or interact with `/streams` endpoints.
- Reviewing violations: Visit `/violations` endpoint or frontend violation log.
- Managing known violators: Use identity confirmation features in the face module.

This guide should help Claude Code quickly understand and work with the Safion codebase.