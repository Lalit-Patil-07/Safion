<div align="center">
  <img src="frontend/public/logo.svg" alt="Safion Logo" width="200" />
  <h1>Safion - PPE Detection System</h1>
</div>

Safion is a real-time Personal Protective Equipment (PPE) detection system designed to enhance workplace safety. It uses a deep learning model to monitor video streams from various sources, detect PPE compliance (hardhats, masks, safety vests), and log violations for review.

## ✨ Features

- **Live Multi-stream Detection**: Monitor multiple video feeds simultaneously from webcams, RTSP streams, or pre-recorded video files.
- **Real-time Violation Alerts**: Get instant visual feedback when a safety violation (e.g., "NO-Hardhat", "NO-Mask") is detected.
- **Comprehensive Violation Log**: Automatically records every violation with a timestamp, violation type, and a cropped image of the individual for evidence.
- **Identity Recognition**: Anonymously groups images of unknown violators. You can then assign a name to a group of images to track repeat offenders.
- **Zoom & Theater Mode**: Focus on a single stream for detailed monitoring or view a thumbnail grid of all active streams.
- **Easy Configuration**: A simple settings page to add, name, and manage RTSP video streams.

## 💻 Technology Stack

- **Backend**: Python, Flask, PyTorch
- **AI Model**: YOLOv11m for object detection and InsightFace (ArcFace) for identity clustering
- **GPU Acceleration**: NVIDIA CUDA
- **Real-time Video Processing**: OpenCV
- **Frontend**: React, Tailwind CSS, Lucide React
- **Database**: PostgreSQL 18 with pgvector
- **Containerization**: Docker

## 🏗️ Architecture Overview

Safion can be run in two different ways depending on your needs:

- **Docker Deployment (Recommended/Production):** Uses a multi-stage Docker build. The React frontend is compiled into static assets and served directly by the Flask backend running under Gunicorn on port `5000`. This requires only a single entry point.
- **Local Development:** Runs as two separate servers. The Flask API runs on port `5000`, and a Node.js React development server runs on port `3000`. This provides hot-reloading for frontend development.

## ⚙️ Environment Setup & Configuration

Before running the application, you must configure your environment variables.

```bash
cp .env.example .env
```

### Key Environment Variables

- **Database Connection**: Do not use a single `DATABASE_URL`. Ensure the discrete variables are set: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD`.
- **First-Run Bootstrap Credentials**: The first time the backend starts, it will attempt to create an initial admin user. You **must** set `DEFAULT_ADMIN_USERNAME` and `DEFAULT_ADMIN_PASSWORD` in your `.env` to log in for the first time.
- **Frontend API URL**:
  - For **Docker**, leave `REACT_APP_API_URL` empty (it defaults to the same host).
  - For **Local Development**, you **must set** `REACT_APP_API_URL=http://localhost:5000` to allow the React dev server (port 3000) to communicate with the Flask backend.

## 🚀 Getting Started

### Option 1: Docker Deployment (Recommended)

**Prerequisites:** Docker with **NVIDIA Container Toolkit** installed.
> ⚠️ **Strict Requirement:** The `docker-compose.yml` uses GPU resource reservations. Without the NVIDIA Container Toolkit installed and configured on your host, the container will fail to start.

> Docker deployment uses a pgvector-enabled PostgreSQL container (`pgvector/pgvector:pg18`). No manual PostgreSQL installation is required. Database initialization is handled automatically via `scripts/init_db.sh` mapped to `/docker-entrypoint-initdb.d/`. `setup_db.sh` is strictly for local development and should NOT be used with Docker.
> **Note on AI Model:** The `best.pt` YOLO model weights are baked directly into the Docker image. If you update the model file, you must rebuild the image.

```bash
cp .env.example .env          # ensure DB_* and DEFAULT_ADMIN_* vars are set
docker compose up -d
```

Open `http://localhost:5000`.

---

### Option 2: Local Development Setup

#### Script responsibilities

| Script | What it does |
|---|---|
| `scripts/install_postgres.sh` | Installs PostgreSQL 18 and the pgvector system extension |
| `scripts/setup_db.sh` | Creates the database user, database, grants, and enables the extension |
| `backend/run.py` | Starts the Flask development server and checks database readiness |

#### System Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| PostgreSQL | **18 exactly** |
| pgvector | system extension (see step 1 — **not a pip package**) |
| NVIDIA CUDA Toolkit | **12.8** (must match torch and onnxruntime-gpu) |

---

#### 1. Install PostgreSQL 18 and pgvector

> **`pgvector` is a PostgreSQL server extension — it is not installed via pip.**
> The `psycopg2-binary` pip package handles the Python DB driver.
> The extension itself must be present on the PostgreSQL server.

```bash
sudo bash scripts/install_postgres.sh
```

This installs PostgreSQL 18, attempts `postgresql-18-pgvector` via apt, and falls back to a source build if the package is unavailable.

**Verify the extension is available:**
```bash
sudo -u postgres psql -c \
  "SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';"
```
A row must appear. If it does not, the extension install failed — do not proceed.

**Run the setup script manually:**
```bash
bash scripts/setup_db.sh
```
This is required for local development to create the database user, database, grants privileges, and enable the `vector` extension.

---

#### 2. Configure environment

```bash
cp .env.example .env
```

**Crucial for Local Dev**: Ensure `REACT_APP_API_URL=http://localhost:5000` is set in your `.env`, and provide a `DB_PASSWORD` and `DEFAULT_ADMIN_PASSWORD`.

---

#### 3. Create the database

```bash
bash scripts/setup_db.sh
```

Creates the PostgreSQL user, database, grants privileges, and enables the `vector` extension. Fully idempotent. This step must be run manually for local development.

If the `postgres` superuser has a password:
```bash
PGPASSWORD=your_postgres_password bash scripts/setup_db.sh
```

---

#### 4. Install Python dependencies

`torch` is declared in `backend/requirements.txt` — no separate install step is needed.

**CPU-only:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

**GPU (CUDA 12.8 Requirement):**
```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt --index-url https://download.pytorch.org/whl/cu128
```

> The `--index-url` flag resolves `torch`, `torchvision`, and `torchaudio` from the
> cu128 wheel index. All other packages are fetched from PyPI as normal.

---

#### ONNXRuntime Setup (CPU vs GPU)

> ⚠️ **`torch` and `onnxruntime-gpu` must both target the same CUDA major version.**
> A mismatch (e.g. torch cu128 + ort built for cu11) causes `libcublasLt.so.12 not found`
> and silent fallback to CPU with no error message.

**This project explicitly targets CUDA 12.8.** Both torch (cu128) and onnxruntime-gpu (CUDA 12.x build
from PyPI) are aligned to CUDA 12. Install CUDA 12.8 from the
[NVIDIA CUDA Archive](https://developer.nvidia.com/cuda-12-8-0-download-archive) before
setting up the GPU environment.

The CPU and GPU `onnxruntime` builds install under the same module name. **Never install
both** in the same environment — uninstall one before switching.

**Default (CPU):** `backend/requirements.txt` ships with `onnxruntime` (CPU) active.

**To switch to GPU:**

1. Open `backend/requirements.txt` and swap the active line:
   ```
   # comment this out:
   onnxruntime>=1.24.0

   # uncomment this:
   onnxruntime-gpu>=1.24.0
   ```

2. Reinstall cleanly:
   ```bash
   pip uninstall -y onnxruntime onnxruntime-gpu
   pip install -r backend/requirements.txt --index-url https://download.pytorch.org/whl/cu128
   ```

3. Verify GPU is active:
   ```bash
   python -c "import torch; import onnxruntime; print(onnxruntime.get_available_providers())"
   # Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
   ```
   Or check the app startup log for:
   ```
   InsightFace loaded — active session providers: ['CUDAExecutionProvider']
   ```

Also ensure `PREFER_GPU=true` is set in your `.env`.

---

#### 5. Install frontend dependencies

```bash
cd frontend && npm install
```

---

#### 6. Run the application

**Backend** (from project root):
```bash
python backend/run.py
```

**Frontend** (separate terminal):
```bash
cd frontend && npm start
```

Opens at `http://localhost:3000`.

---

#### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `could not connect to server` | PostgreSQL not running | `sudo systemctl start postgresql` |
| `extension "vector" is not available` | pgvector not installed at system level | Re-run `sudo bash scripts/install_postgres.sh` |
| `role "safion" does not exist` | setup_db.sh not run | `bash scripts/setup_db.sh` |
| `password authentication failed` | Wrong password in `.env` | Check `DB_PASSWORD` and `DB_USER` in `.env` |
| `could not load library "vector.so"` | pgvector built for wrong PG version | Reinstall pgvector against PG 18 |

---

## 📂 Project Structure

```
ppe-detection-system/
├── backend/
│   ├── app.py                  # Flask application factory
│   ├── run.py                  # Development entry point
│   ├── config.py               # Configuration (reads from .env)
│   ├── auth/                   # JWT authentication
│   ├── detection/              # YOLO inference + violation association
│   ├── face/                   # InsightFace pipeline, identity models, clustering
│   ├── streams/                # Stream manager, worker, IoU tracker
│   ├── tasks/                  # Async violation queue
│   ├── violations/             # Violation log routes
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── src/App.js              # React application
│   └── package.json
├── scripts/
│   ├── install_postgres.sh     # Installs PostgreSQL 18 + pgvector (run once, as root)
│   └── setup_db.sh             # Creates DB user, database, enables extension
├── best.pt                     # Trained YOLOv11m model weights
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

## 🧠 The AI Model

The detection model is a YOLOv11m model trained on a custom dataset for PPE detection. The model detects the following classes:

- Hardhat / NO-Hardhat
- Mask / NO-Mask
- Safety Vest / NO-Safety Vest
- Person, Safety Cone, Machinery, Vehicle

## 📋 Future Improvements

- **Email/SMS Notifications**: Send real-time alerts for critical violations
- **Advanced Analytics**: A dashboard with charts and statistics on violation trends
- **Model Optimization**: Further training and optimization of the YOLO model for higher accuracy