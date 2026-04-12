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
        "connect_args": {"check_same_thread": False},
    }

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "jwt-secret-change-in-production")
    JWT_ACCESS_TOKEN_EXPIRES: timedelta = timedelta(hours=int(os.environ.get("JWT_ACCESS_HOURS", "8")))
    JWT_REFRESH_TOKEN_EXPIRES: timedelta = timedelta(days=int(os.environ.get("JWT_REFRESH_DAYS", "30")))
    JWT_ALGORITHM: str = "HS256"

    # ── File Storage ──────────────────────────────────────────────────────────
    VIOLATIONS_IMAGE_DIR: str = os.environ.get(
        "VIOLATIONS_IMAGE_DIR", os.path.join(BASE_DIR, "violations_images")
    )
    TEMP_DIR: str = os.environ.get(
        "TEMP_DIR", os.path.join(BASE_DIR, "temp")
    )

    # ── YOLO ──────────────────────────────────────────────────────────────────
    MODEL_PATH: str = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "..", "best.pt"))
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.4"))
    YOLO_DEVICE: str = os.environ.get("YOLO_DEVICE", "auto")

    # ── InsightFace ───────────────────────────────────────────────────────────
    # buffalo_l: RetinaFace detection + ArcFace 512-dim embeddings (L2-normalised)
    INSIGHTFACE_MODEL: str = os.environ.get("INSIGHTFACE_MODEL", "buffalo_l")
    # 0.55 is the reliable lower bound for ArcFace same-person on occluded surveillance footage
    # 0.45 (old default) was too low — caused identity fragmentation
    IDENTITY_MATCH_THRESHOLD: float = float(os.environ.get("IDENTITY_MATCH_THRESHOLD", "0.55"))
    # Minimum combined quality score to accept an embedding (det_score × area_ratio)
    EMBEDDING_QUALITY_MIN: float = float(os.environ.get("EMBEDDING_QUALITY_MIN", "0.45"))

    # ── Multi-prototype identity model ────────────────────────────────────────
    # Max prototypes kept per identity in the live cache (K in K-prototype model)
    MAX_PROTOTYPES: int = int(os.environ.get("MAX_PROTOTYPES", "5"))
    # Cosine similarity above which a new embedding merges into an existing prototype
    # rather than creating a new one (same pose / lighting condition)
    PROTO_MERGE_THRESHOLD: float = float(os.environ.get("PROTO_MERGE_THRESHOLD", "0.80"))

    # ── Merge suggestion engine ───────────────────────────────────────────────
    # Minimum cross-identity similarity to surface as a merge suggestion
    SUGGESTION_THRESHOLD: float = float(os.environ.get("SUGGESTION_THRESHOLD", "0.68"))
    # Maximum suggestions returned per call
    SUGGESTION_MAX_RESULTS: int = int(os.environ.get("SUGGESTION_MAX_RESULTS", "30"))
    # Minimum cosine similarity to update centroid (below this = outlier, stored but ignored)
    OUTLIER_MIN_SIMILARITY: float = float(os.environ.get("OUTLIER_MIN_SIMILARITY", "0.35"))
    FACE_DET_SCORE_MIN: float = float(os.environ.get("FACE_DET_SCORE_MIN", "0.6"))

    # ── Clustering ────────────────────────────────────────────────────────────
    CLUSTER_EPS: float = float(os.environ.get("CLUSTER_EPS", "0.40"))
    CLUSTER_MIN_SAMPLES: int = int(os.environ.get("CLUSTER_MIN_SAMPLES", "2"))
    # Auto-trigger clustering every N violations (0 = disabled)
    CLUSTER_EVERY_N_VIOLATIONS: int = int(os.environ.get("CLUSTER_EVERY_N_VIOLATIONS", "50"))

    # ── Stream / Violations ───────────────────────────────────────────────────
    VIOLATION_COOLDOWN_SECONDS: int = int(os.environ.get("VIOLATION_COOLDOWN", "10"))
    # Identity-level dedup window (seconds) — authoritative second layer
    IDENTITY_VIOLATION_COOLDOWN: int = int(os.environ.get("IDENTITY_VIOLATION_COOLDOWN", "30"))
    STREAM_JPEG_QUALITY: int = int(os.environ.get("STREAM_JPEG_QUALITY", "80"))
    MAX_CONCURRENT_STREAMS: int = int(os.environ.get("MAX_CONCURRENT_STREAMS", "8"))
    FRAME_RATE_LIMIT: int = int(os.environ.get("FRAME_RATE_LIMIT", "30"))

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