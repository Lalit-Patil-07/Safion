import os
from datetime import timedelta


def _require(key: str) -> str:
    """Return the environment variable or raise RuntimeError immediately."""
    val = os.environ.get(key, "")
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


class Config:
    # ── Core ──────────────────────────────────────────────────────────────────
    SECRET_KEY: str = _require("SECRET_KEY")
    DEBUG: bool     = _require("DEBUG").lower() == "true"

    # ── Database ──────────────────────────────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

    _DB_USER:     str = _require("DB_USER")
    _DB_PASSWORD: str = _require("DB_PASSWORD")
    _DB_HOST:     str = _require("DB_HOST")
    _DB_PORT:     str = _require("DB_PORT")
    _DB_NAME:     str = _require("DB_NAME")

    SQLALCHEMY_DATABASE_URI: str = (
        f"postgresql://{_DB_USER}:{_DB_PASSWORD}@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        "pool_pre_ping": True,
    }

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str                  = _require("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES: timedelta  = timedelta(hours=int(_require("JWT_ACCESS_HOURS")))
    JWT_REFRESH_TOKEN_EXPIRES: timedelta = timedelta(days=int(_require("JWT_REFRESH_DAYS")))
    JWT_ALGORITHM: str                   = "HS256"

    # ── File Storage ──────────────────────────────────────────────────────────
    VIOLATIONS_IMAGE_DIR: str = os.environ.get(
        "VIOLATIONS_IMAGE_DIR", os.path.join(BASE_DIR, "violations_images")
    )
    TEMP_DIR: str = os.environ.get(
        "TEMP_DIR", os.path.join(BASE_DIR, "temp")
    )

    # ── YOLO ──────────────────────────────────────────────────────────────────
    MODEL_PATH: str             = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "..", "best.pt"))
    CONFIDENCE_THRESHOLD: float = float(_require("CONFIDENCE_THRESHOLD"))
    YOLO_DEVICE: str            = _require("YOLO_DEVICE")
    YOLO_BATCH_SIZE: int        = int(os.environ.get("YOLO_BATCH_SIZE", "4"))
    YOLO_BATCH_TIMEOUT_MS: int  = int(os.environ.get("YOLO_BATCH_TIMEOUT_MS", "20"))

    # ── InsightFace ───────────────────────────────────────────────────────────
    INSIGHTFACE_MODEL: str          = _require("INSIGHTFACE_MODEL")
    PREFER_GPU: bool                = _require("PREFER_GPU").lower() == "true"
    IDENTITY_MATCH_THRESHOLD: float = float(_require("IDENTITY_MATCH_THRESHOLD"))
    EMBEDDING_QUALITY_MIN: float    = float(_require("EMBEDDING_QUALITY_MIN"))

    # ── Multi-prototype identity model ────────────────────────────────────────
    MAX_PROTOTYPES: int          = int(_require("MAX_PROTOTYPES"))
    PROTO_MERGE_THRESHOLD: float = float(_require("PROTO_MERGE_THRESHOLD"))

    # ── Face tracker ─────────────────────────────────────────────────────────
    TRACK_MIN_FRAMES: int      = int(_require("TRACK_MIN_FRAMES"))
    TRACK_MAX_LOST: int        = int(_require("TRACK_MAX_LOST"))
    TRACK_IOU_THRESHOLD: float = float(_require("TRACK_IOU_THRESHOLD"))
    TRACK_MIN_EMBEDDINGS: int  = int(_require("TRACK_MIN_EMBEDDINGS"))

    # ── Merge suggestion engine ───────────────────────────────────────────────
    SUGGESTION_THRESHOLD: float      = float(_require("SUGGESTION_THRESHOLD"))
    SIMILARITY_SOFT_THRESHOLD: float = float(_require("SIMILARITY_SOFT_THRESHOLD"))
    SUGGESTION_MAX_RESULTS: int      = int(_require("SUGGESTION_MAX_RESULTS"))
    OUTLIER_MIN_SIMILARITY: float    = float(_require("OUTLIER_MIN_SIMILARITY"))
    FACE_DET_SCORE_MIN: float        = float(_require("FACE_DET_SCORE_MIN"))

    # ── Clustering ────────────────────────────────────────────────────────────
    CLUSTER_EPS: float              = float(_require("CLUSTER_EPS"))
    CLUSTER_MIN_SAMPLES: int        = int(_require("CLUSTER_MIN_SAMPLES"))
    CLUSTER_EVERY_N_VIOLATIONS: int = int(_require("CLUSTER_EVERY_N_VIOLATIONS"))

    # ── Stream / Violations ───────────────────────────────────────────────────
    VIOLATION_COOLDOWN_SECONDS: int  = int(_require("VIOLATION_COOLDOWN"))
    IDENTITY_VIOLATION_COOLDOWN: int = int(_require("IDENTITY_VIOLATION_COOLDOWN"))
    STREAM_JPEG_QUALITY: int         = int(_require("STREAM_JPEG_QUALITY"))
    MAX_CONCURRENT_STREAMS: int      = int(_require("MAX_CONCURRENT_STREAMS"))
    FRAME_RATE_LIMIT: int            = int(_require("FRAME_RATE_LIMIT"))
    FACE_EMBED_EVERY_N_FRAMES: int   = int(_require("FACE_EMBED_EVERY_N_FRAMES"))
    IDENTITY_RECHECK_SECONDS: float = float(_require("IDENTITY_RECHECK_SECONDS"))
    MAX_PENDING_EMBEDDINGS: int     = int(_require("MAX_PENDING_EMBEDDINGS"))
    PROCESS_WIDTH: int               = int(_require("PROCESS_WIDTH"))
    STREAM_OUTPUT_FPS: int           = int(_require("STREAM_OUTPUT_FPS"))

    # ── Task Queue ────────────────────────────────────────────────────────────
    TASK_WORKER_THREADS: int = int(_require("TASK_WORKER_THREADS"))
    TASK_QUEUE_MAXSIZE: int  = int(_require("TASK_QUEUE_MAXSIZE"))

    # ── Identity temporal bias ───────────────────────────────────────────────
    TEMPORAL_BOOST: float         = float(_require("TEMPORAL_BOOST"))
    EMA_ALPHA: float              = float(_require("EMA_ALPHA"))
    STRONG_MATCH_THRESHOLD: float = float(_require("STRONG_MATCH_THRESHOLD"))
    RECENT_WINDOW: float          = float(_require("RECENT_WINDOW"))

    # ── Pipeline queues ───────────────────────────────────────────────────────
    PROCESS_QUEUE_SIZE: int     = int(os.environ.get("PROCESS_QUEUE_SIZE", "2"))
    OUTPUT_QUEUE_SIZE: int      = int(os.environ.get("OUTPUT_QUEUE_SIZE", "2"))
    FACE_QUEUE_SIZE: int        = int(os.environ.get("FACE_QUEUE_SIZE", "4"))

    # ── Adaptive frame skip ───────────────────────────────────────────────────
    MAX_FRAME_SKIP: int          = int(os.environ.get("MAX_FRAME_SKIP", "4"))
    LOAD_HIGH_THRESHOLD: float   = float(os.environ.get("LOAD_HIGH_THRESHOLD", "0.8"))
    LOAD_LOW_THRESHOLD: float    = float(os.environ.get("LOAD_LOW_THRESHOLD", "0.3"))
    TRACK_STALE_TIMEOUT_S: float        = float(os.environ.get("TRACK_STALE_TIMEOUT_S", "30.0"))
    VIDEO_CAPTURE_BACKEND: str          = os.environ.get("VIDEO_CAPTURE_BACKEND", "OPENCV").upper()
    EMBED_QUALITY_IMPROVE_MARGIN: float  = float(os.environ.get("EMBED_QUALITY_IMPROVE_MARGIN", "0.12"))
    IDENTITY_MIN_EMBEDDINGS_DELTA: int   = int(os.environ.get("IDENTITY_MIN_EMBEDDINGS_DELTA", "3"))

    # ── Violation write batching ───────────────────────────────────────────────
    VIOLATION_BATCH_SIZE: int        = int(os.environ.get("VIOLATION_BATCH_SIZE", "8"))
    VIOLATION_BATCH_TIMEOUT_MS: int  = int(os.environ.get("VIOLATION_BATCH_TIMEOUT_MS", "400"))
    VIOLATION_COALESCE_WINDOW_S: float = float(os.environ.get("VIOLATION_COALESCE_WINDOW_S", "2.0"))

    # ── PPE Classes (internal constant — not configurable via env) ────────────
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