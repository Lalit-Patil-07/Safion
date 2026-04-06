"""
Safion — Flask application factory
===================================
Usage:
    from app import create_app
    app = create_app()
    app.run(...)

Or via Flask CLI:
    export FLASK_APP=app:create_app
    flask run
"""

import logging
import os

from flask import Flask, send_from_directory, jsonify

from config import Config
from extensions import db, bcrypt, jwt


def create_app(config_class=Config) -> Flask:
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "build"),
        static_url_path="/",
    )
    app.config.from_object(config_class)

    _configure_logging(app)
    _init_extensions(app)
    _create_directories(app)
    _register_blueprints(app)
    _register_frontend_catch_all(app)
    _register_error_handlers(app)
    _init_services(app)

    return app


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging(app: Flask) -> None:
    level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Flask extensions
# ---------------------------------------------------------------------------
def _init_extensions(app: Flask) -> None:
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)

    # JWT error handlers — return JSON instead of HTML
    @jwt.unauthorized_loader
    def missing_token_callback(reason):
        return jsonify({"error": "Authentication required.", "detail": reason}), 401

    @jwt.invalid_token_loader
    def invalid_token_callback(reason):
        return jsonify({"error": "Invalid token.", "detail": reason}), 401

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_data):
        return jsonify({"error": "Token has expired."}), 401

    with app.app_context():
        # Import all models so SQLAlchemy knows about them before create_all
        import auth.models       # noqa: F401
        import face.models       # noqa: F401
        db.create_all()


# ---------------------------------------------------------------------------
# Required directories
# ---------------------------------------------------------------------------
def _create_directories(app: Flask) -> None:
    for key in ("VIOLATIONS_IMAGE_DIR", "KNOWN_FACES_DIR", "TEMP_DIR"):
        path = app.config[key]
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Blueprints
# ---------------------------------------------------------------------------
def _register_blueprints(app: Flask) -> None:
    from auth.routes import auth_bp
    from streams.routes import stream_bp
    from violations.routes import violations_bp
    from face.routes import face_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(stream_bp)
    app.register_blueprint(violations_bp)
    app.register_blueprint(face_bp)

    # Health check — no auth required
    @app.get("/health")
    def health():
        yolo = app.extensions.get("yolo_service")
        return jsonify({
            "status": "healthy",
            "model_loaded": yolo.is_loaded if yolo else False,
            "device": yolo._device if yolo else "unknown",
        })


# ---------------------------------------------------------------------------
# Frontend catch-all (serve React build)
# ---------------------------------------------------------------------------
def _register_frontend_catch_all(app: Flask) -> None:
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path: str):
        static = app.static_folder
        if path and os.path.exists(os.path.join(static, path)):
            return send_from_directory(static, path)
        return send_from_directory(static, "index.html")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Service initialisation (YOLO, face pipeline, task queue, stream manager)
# ---------------------------------------------------------------------------
def _init_services(app: Flask) -> None:
    from detection.yolo_service import YOLOService
    from face.pipeline import FaceRecognitionPipeline
    from tasks.queue import TaskQueue
    from streams.manager import StreamManager

    # ── YOLO ─────────────────────────────────────────────────────────────────
    yolo = YOLOService()
    yolo.init_app(app)
    app.extensions["yolo_service"] = yolo

    # ── Face recognition pipeline ─────────────────────────────────────────────
    pipeline = FaceRecognitionPipeline(app.config)
    app.extensions["face_pipeline"] = pipeline

    with app.app_context():
        pipeline.reload_cache()

    # ── Task queue (async face matching + violation logging) ──────────────────
    task_queue = TaskQueue()
    task_queue.init_app(app)
    app.extensions["task_queue"] = task_queue

    # ── Stream manager ────────────────────────────────────────────────────────
    stream_manager = StreamManager()
    stream_manager.init_app(app)
    app.extensions["stream_manager"] = stream_manager

    # ── Shutdown hook ─────────────────────────────────────────────────────────
    import atexit

    def _shutdown():
        stream_manager.stop_all()
        task_queue.shutdown()

    atexit.register(_shutdown)
