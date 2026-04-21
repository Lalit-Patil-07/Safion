# Safion - PPE Detection System

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

## 🚀 Getting Started

### Option 1: Docker Deployment (Recommended)

**Prerequisites:** Docker with NVIDIA Container Toolkit installed.

> Docker deployment uses a pgvector-enabled PostgreSQL container (`pgvector/pgvector:pg16`).
> No manual PostgreSQL or pgvector installation is required.

```bash
cp .env.example .env          # edit secrets if needed
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
| `backend/run.py` | Auto-triggers `setup_db.sh` on first start if DB is not ready |

---

#### System Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| PostgreSQL | **18 exactly** |
| pgvector | system extension (see step 1 — **not a pip package**) |
| NVIDIA CUDA Toolkit | 12.4+ (CUDA 13 requires compatibility workaround — see below) |

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

---

#### 2. Configure environment

```bash
cp .env.example .env
```

Default values work for local development without changes.

---

#### 3. Create the database

```bash
bash scripts/setup_db.sh
```

Creates the PostgreSQL user, database, grants privileges, and enables the `vector` extension. Fully idempotent.

If the `postgres` superuser has a password:
```bash
PGPASSWORD=your_postgres_password bash scripts/setup_db.sh
```

---

#### 4. Install Python dependencies

**CPU-only:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

**GPU (CUDA 12.4 / CUDA 13):**
```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

> `torch` is declared in `requirements.txt` — no separate install step is needed.
> The `--index-url` flag makes pip resolve `torch`, `torchvision`, and `torchaudio`
> from the cu124 wheel index while installing all other packages from PyPI normally.
---

#### ONNXRuntime Setup (CPU vs GPU)

> ⚠️ **Mismatched CUDA versions will cause silent CPU fallback** — no error, just no GPU.

**Compatibility matrix:**

| System CUDA | `torch` wheel | `onnxruntime-gpu` | Works? |
|---|---|---|---|
| 12.x | cu124 | `>=1.24.0` (PyPI) | ✅ |
| 13.x | cu124 | `>=1.24.0` (PyPI) | ✅ via PyTorch DLL preload |
| 13.x | cu124 | `<1.24.0` | ❌ missing `libcublasLt.so.12` |
| 13.x | cu130 | `>=1.24.0` (PyPI) | ❌ DLL version mismatch |

**Why CUDA 13 works with CUDA 12 packages:** PyTorch cu124 bundles `libcublasLt.so.12` inside
its wheel. The app imports `torch` before ONNXRuntime, which preloads those CUDA 12 DLLs into
the process. ONNXRuntime then finds them without needing a system-level CUDA 12 install.
NVIDIA drivers guarantee forward compatibility for CUDA 12 binaries running on CUDA 13 hardware.

The CPU and GPU `onnxruntime` builds install under the same package name and **must never both
be installed** in the same environment.

**Default (CPU):** `requirements.txt` ships with `onnxruntime` active. No changes needed.

**To switch to GPU:**

1. Open `backend/requirements.txt` and swap the active line:
   ```
   # comment this out:
   onnxruntime>=1.24.0

   # uncomment this:
   onnxruntime-gpu>=1.24.0
   ```

2. Reinstall cleanly (the uninstall step is required):
   ```bash
   pip uninstall -y onnxruntime onnxruntime-gpu
   pip install -r backend/requirements.txt --index-url https://download.pytorch.org/whl/cu124
   ```

3. Verify the session is actually using GPU (not just that the provider is *available*):
   ```bash
   python -c "import torch; import onnxruntime; print(onnxruntime.get_available_providers())"
   # Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
   # Note: import torch FIRST — this preloads the CUDA DLLs
   ```
   Or check the app startup log for:
   ```
   InsightFace loaded — active session providers: ['CUDAExecutionProvider']
   ```
   If instead you see an ERROR log mentioning `libcublasLt.so.12`, `onnxruntime-gpu`
   predates DLL preloading support — ensure version `>=1.24.0` is installed.

Also ensure `PREFER_GPU=true` is set in your `.env`.


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

On first run, `run.py` detects if the database or `vector` extension is missing and runs `setup_db.sh` automatically. On subsequent starts it performs a single fast check and skips setup.

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
| `password authentication failed` | Wrong password in `.env` | Check `DATABASE_URL` in `.env` |
| `could not load library "vector.so"` | pgvector built for wrong PG version | Reinstall pgvector against PG 18 |

---

## 📂 Project Structure

```
ppe-detection-system/
├── backend/
│   ├── app.py                  # Flask application factory
│   ├── run.py                  # Development entry point (auto-setup)
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