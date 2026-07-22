"""
Application configuration from environment variables.

All required settings use ``_require()`` which raises RuntimeError on
missing values, failing fast at startup rather than at first use.
"""
import os
from datetime import timedelta


def _require(key: str) -> str:
    """Return the environment variable or raise RuntimeError immediately."""
    val = os.environ.get(key, "")
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


def _require_float(key: str) -> float:
    """Return the env var as a float, or raise with a readable message."""
    raw = _require(key)
    try:
        return float(raw)
    except (ValueError, TypeError):
        raise RuntimeError(
            f"{key} must be a number (float), got: {raw!r}"
        ) from None


def _require_int(key: str) -> int:
    """Return the env var as an int, or raise with a readable message."""
    raw = _require(key)
    try:
        return int(raw)
    except (ValueError, TypeError):
        raise RuntimeError(
            f"{key} must be an integer, got: {raw!r}"
        ) from None


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
    JWT_SECRET_KEY: str = _require("JWT_SECRET_KEY")
    _access_hours       = int(os.environ.get("JWT_ACCESS_HOURS",   "0"))
    _access_minutes     = int(os.environ.get("JWT_ACCESS_MINUTES", "15"))
    JWT_ACCESS_TOKEN_EXPIRES: timedelta = (
        timedelta(hours=_access_hours) if _access_hours > 0
        else timedelta(minutes=_access_minutes)
    )
    JWT_REFRESH_TOKEN_EXPIRES: timedelta = timedelta(
        days=int(os.environ.get("JWT_REFRESH_DAYS", "7"))
    )
    JWT_ALGORITHM: str                   = "HS256"

    # ── JWT Cookie ───────────────────────────────────────────────────────────
    JWT_TOKEN_LOCATION: list = ["cookies"]
    JWT_ACCESS_COOKIE_NAME: str = "access_token_cookie"
    JWT_REFRESH_COOKIE_NAME: str = "refresh_token_cookie"
    JWT_COOKIE_SECURE: bool = os.environ.get("JWT_COOKIE_SECURE", "false").lower() == "true"
    JWT_COOKIE_SAMESITE: str = os.environ.get("JWT_COOKIE_SAMESITE", "Lax")
    JWT_COOKIE_DOMAIN: str = os.environ.get("JWT_COOKIE_DOMAIN", "") or None
    JWT_SESSION_COOKIE: bool = False

    # ── CSRF (double-submit cookie via flask-jwt-extended) ────────────────────
    JWT_COOKIE_CSRF_PROTECT: bool = True
    JWT_CSRF_IN_COOKIES: bool = True
    JWT_ACCESS_CSRF_HEADER_NAME: str = "X-CSRF-Token"
    JWT_REFRESH_CSRF_HEADER_NAME: str = "X-CSRF-Token"

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list = (
        os.environ.get("CORS_ORIGINS", "").split(",")
        if os.environ.get("CORS_ORIGINS")
        else ["*"]
    )
    CORS_SUPPORTS_CREDENTIALS: bool = True

    # ── Rate Limiting ────────────────────────────────────────────────────────
    RATELIMIT_DEFAULT: str = os.environ.get("RATELIMIT_DEFAULT", "200/minute")
    RATELIMIT_STORAGE_URI: str = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")

    # ── File Storage ──────────────────────────────────────────────────────────
    VIOLATIONS_IMAGE_DIR: str = os.environ.get(
        "VIOLATIONS_IMAGE_DIR", os.path.join(BASE_DIR, "violations_images")
    )
    TEMP_DIR: str = os.environ.get(
        "TEMP_DIR", os.path.join(BASE_DIR, "temp")
    )

    # ── Identity Storage ──────────────────────────────────────────────────────────
    IDENTITY_STORAGE_PATH = os.environ.get(
        "IDENTITY_STORAGE_PATH", os.path.join(BASE_DIR, "identity_storage")
    )

    IMPORT_MAX_ZIP_MB: int = int(os.environ.get("IMPORT_MAX_ZIP_MB", "100"))

    # ── YOLO ──────────────────────────────────────────────────────────────────
    MODEL_PATH: str             = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "..", "best.pt"))
    CONFIDENCE_THRESHOLD: float = _require_float("CONFIDENCE_THRESHOLD")
    YOLO_DEVICE: str            = _require("YOLO_DEVICE")
    YOLO_BATCH_SIZE: int        = int(os.environ.get("YOLO_BATCH_SIZE", "4"))
    YOLO_BATCH_TIMEOUT_MS: int  = int(os.environ.get("YOLO_BATCH_TIMEOUT_MS", "20"))

    # ── InsightFace ───────────────────────────────────────────────────────────
    INSIGHTFACE_MODEL: str          = _require("INSIGHTFACE_MODEL")
    PREFER_GPU: bool                = _require("PREFER_GPU").lower() == "true"
    IDENTITY_MATCH_THRESHOLD: float = _require_float("IDENTITY_MATCH_THRESHOLD")
    EMBEDDING_QUALITY_MIN: float    = _require_float("EMBEDDING_QUALITY_MIN")

    # ── Multi-prototype identity model ────────────────────────────────────────
    MAX_PROTOTYPES: int          = _require_int("MAX_PROTOTYPES")
    PROTO_MERGE_THRESHOLD: float = _require_float("PROTO_MERGE_THRESHOLD")

    # ── Face tracker ─────────────────────────────────────────────────────────
    TRACK_MIN_FRAMES: int      = _require_int("TRACK_MIN_FRAMES")
    TRACK_MAX_LOST: int        = _require_int("TRACK_MAX_LOST")
    TRACK_IOU_THRESHOLD: float = _require_float("TRACK_IOU_THRESHOLD")
    TRACK_MIN_EMBEDDINGS: int  = _require_int("TRACK_MIN_EMBEDDINGS")

    # ── Merge suggestion engine ───────────────────────────────────────────────
    SUGGESTION_THRESHOLD: float      = _require_float("SUGGESTION_THRESHOLD")
    SIMILARITY_SOFT_THRESHOLD: float = _require_float("SIMILARITY_SOFT_THRESHOLD")
    SUGGESTION_MAX_RESULTS: int      = _require_int("SUGGESTION_MAX_RESULTS")
    OUTLIER_MIN_SIMILARITY: float    = _require_float("OUTLIER_MIN_SIMILARITY")
    FACE_DET_SCORE_MIN: float        = _require_float("FACE_DET_SCORE_MIN")

    # ── Clustering ────────────────────────────────────────────────────────────
    CLUSTER_EPS: float              = _require_float("CLUSTER_EPS")
    CLUSTER_MIN_SAMPLES: int        = _require_int("CLUSTER_MIN_SAMPLES")
    CLUSTER_EVERY_N_VIOLATIONS: int = _require_int("CLUSTER_EVERY_N_VIOLATIONS")

    # ── Stream / Violations ───────────────────────────────────────────────────
    VIOLATION_COOLDOWN_SECONDS: int  = _require_int("VIOLATION_COOLDOWN")
    IDENTITY_VIOLATION_COOLDOWN: int = _require_int("IDENTITY_VIOLATION_COOLDOWN")
    STREAM_JPEG_QUALITY: int         = _require_int("STREAM_JPEG_QUALITY")
    MAX_CONCURRENT_STREAMS: int      = _require_int("MAX_CONCURRENT_STREAMS")
    FRAME_RATE_LIMIT: int            = _require_int("FRAME_RATE_LIMIT")
    FACE_EMBED_EVERY_N_FRAMES: int   = _require_int("FACE_EMBED_EVERY_N_FRAMES")
    IDENTITY_RECHECK_SECONDS: float = _require_float("IDENTITY_RECHECK_SECONDS")
    MAX_PENDING_EMBEDDINGS: int     = _require_int("MAX_PENDING_EMBEDDINGS")
    PROCESS_WIDTH: int               = _require_int("PROCESS_WIDTH")
    STREAM_OUTPUT_FPS: int           = _require_int("STREAM_OUTPUT_FPS")

    # ── Task Queue ────────────────────────────────────────────────────────────
    TASK_WORKER_THREADS: int = _require_int("TASK_WORKER_THREADS")
    TASK_QUEUE_MAXSIZE: int  = _require_int("TASK_QUEUE_MAXSIZE")

    # ── Identity temporal bias ───────────────────────────────────────────────
    TEMPORAL_BOOST: float         = _require_float("TEMPORAL_BOOST")
    EMA_ALPHA: float              = _require_float("EMA_ALPHA")
    STRONG_MATCH_THRESHOLD: float = _require_float("STRONG_MATCH_THRESHOLD")
    RECENT_WINDOW: float          = _require_float("RECENT_WINDOW")

    # ── Pipeline queues ───────────────────────────────────────────────────────
    PROCESS_QUEUE_SIZE: int     = int(os.environ.get("PROCESS_QUEUE_SIZE", "2"))
    OUTPUT_QUEUE_SIZE: int      = int(os.environ.get("OUTPUT_QUEUE_SIZE", "2"))
    FACE_QUEUE_SIZE: int        = int(os.environ.get("FACE_QUEUE_SIZE", "4"))

    # ── Adaptive frame skip ───────────────────────────────────────────────────
    MAX_FRAME_SKIP: int          = int(os.environ.get("MAX_FRAME_SKIP", "4"))
    LOAD_HIGH_THRESHOLD: float   = float(os.environ.get("LOAD_HIGH_THRESHOLD", "0.8"))
    LOAD_LOW_THRESHOLD: float    = float(os.environ.get("LOAD_LOW_THRESHOLD", "0.3"))
    TRACK_STALE_TIMEOUT_S: float = float(os.environ.get("TRACK_STALE_TIMEOUT_S", "30.0"))
    EMBED_QUALITY_IMPROVE_MARGIN: float  = float(os.environ.get("EMBED_QUALITY_IMPROVE_MARGIN", "0.12"))
    IDENTITY_MIN_EMBEDDINGS_DELTA: int   = int(os.environ.get("IDENTITY_MIN_EMBEDDINGS_DELTA", "3"))

    # ── Violation write batching ───────────────────────────────────────────────
    VIOLATION_BATCH_SIZE: int        = int(os.environ.get("VIOLATION_BATCH_SIZE", "8"))
    VIOLATION_BATCH_TIMEOUT_MS: int  = int(os.environ.get("VIOLATION_BATCH_TIMEOUT_MS", "400"))
    VIOLATION_COALESCE_WINDOW_S: float = float(os.environ.get("VIOLATION_COALESCE_WINDOW_S", "2.0"))

    # ── Stream restart and stall detection ─────────────────────────────────────
    STREAM_STALL_TIMEOUT_S: float       = float(os.environ.get("STREAM_STALL_TIMEOUT_S", "10.0"))
    STREAM_RESTART_DELAY_S: float       = float(os.environ.get("STREAM_RESTART_DELAY_S", "3.0"))

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