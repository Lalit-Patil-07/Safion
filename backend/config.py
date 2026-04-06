import os
from datetime import timedelta


class Config:
    # ── Core ──────────────────────────────────────────────────────────────────
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"

    # ── Database ──────────────────────────────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    SQLALCHEMY_DATABASE_URI: str = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'safion.db')}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        "pool_pre_ping": True,
        "connect_args": {"check_same_thread": False},  # SQLite only; Postgres ignores
    }

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "jwt-secret-change-in-production")
    JWT_ACCESS_TOKEN_EXPIRES: timedelta = timedelta(
        hours=int(os.environ.get("JWT_ACCESS_HOURS", "8"))
    )
    JWT_REFRESH_TOKEN_EXPIRES: timedelta = timedelta(
        days=int(os.environ.get("JWT_REFRESH_DAYS", "30"))
    )
    JWT_ALGORITHM: str = "HS256"

    # ── File Storage ──────────────────────────────────────────────────────────
    VIOLATIONS_IMAGE_DIR: str = os.environ.get(
        "VIOLATIONS_IMAGE_DIR",
        os.path.join(BASE_DIR, "violations_images"),
    )
    KNOWN_FACES_DIR: str = os.environ.get(
        "KNOWN_FACES_DIR",
        os.path.join(BASE_DIR, "known_faces"),
    )
    TEMP_DIR: str = os.environ.get(
        "TEMP_DIR",
        os.path.join(BASE_DIR, "temp"),
    )

    # ── YOLO ──────────────────────────────────────────────────────────────────
    MODEL_PATH: str = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "..", "best.pt"))
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.4"))
    YOLO_DEVICE: str = os.environ.get("YOLO_DEVICE", "auto")  # auto | cpu | cuda

    # ── Face Recognition ──────────────────────────────────────────────────────
    FACE_MATCH_THRESHOLD: float = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.50"))
    FACE_HEAD_CROP_RATIO: float = float(os.environ.get("FACE_HEAD_CROP_RATIO", "0.35"))
    FACE_HEAD_CROP_PADDING: float = float(os.environ.get("FACE_HEAD_CROP_PADDING", "0.15"))
    FACE_MIN_CROP_SIZE: int = int(os.environ.get("FACE_MIN_CROP_SIZE", "120"))
    FACE_MIN_QUALITY_SCORE: float = float(os.environ.get("FACE_MIN_QUALITY_SCORE", "0.40"))

    # ── Stream / Violations ───────────────────────────────────────────────────
    VIOLATION_COOLDOWN_SECONDS: int = int(os.environ.get("VIOLATION_COOLDOWN", "10"))
    STREAM_JPEG_QUALITY: int = int(os.environ.get("STREAM_JPEG_QUALITY", "80"))
    MAX_CONCURRENT_STREAMS: int = int(os.environ.get("MAX_CONCURRENT_STREAMS", "8"))
    FRAME_RATE_LIMIT: int = int(os.environ.get("FRAME_RATE_LIMIT", "30"))  # max fps per stream

    # ── Task Queue ────────────────────────────────────────────────────────────
    TASK_WORKER_THREADS: int = int(os.environ.get("TASK_WORKER_THREADS", "2"))
    TASK_QUEUE_MAXSIZE: int = int(os.environ.get("TASK_QUEUE_MAXSIZE", "200"))

    # ── PPE Classes ───────────────────────────────────────────────────────────
    PPE_CLASSES: dict = {
        0: {"name": "Hardhat",        "color": "#3B82F6", "safe": True},
        1: {"name": "Mask",           "color": "#10B981", "safe": True},
        2: {"name": "NO-Hardhat",     "color": "#EF4444", "safe": False},
        3: {"name": "NO-Mask",        "color": "#F59E0B", "safe": False},
        4: {"name": "NO-Safety Vest", "color": "#EC4899", "safe": False},
        5: {"name": "Person",         "color": "#FBBF24", "safe": True},
        6: {"name": "Safety Cone",    "color": "#8B5CF6", "safe": True},
        7: {"name": "Safety Vest",    "color": "#059669", "safe": True},
        8: {"name": "Machinery",      "color": "#6366F1", "safe": True},
        9: {"name": "Vehicle",        "color": "#14B8A6", "safe": True},
    }
