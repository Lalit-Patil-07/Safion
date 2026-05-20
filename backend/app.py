"""
Safion — Flask application factory.

Creates and configures the Flask app, initialises extensions (SQLAlchemy,
bcrypt, JWT), registers blueprints, and starts background services
(YOLO, InsightFace, stream manager, task queue).

The factory follows the ``create_app()`` pattern so the app can be
imported by Gunicorn, tests, and CLI scripts without side effects.
"""
import logging
import os
import uuid

from flask import Flask, send_from_directory, jsonify, request
from sqlalchemy import text

from config import Config
from extensions import db, bcrypt, jwt, limiter
from version import __version__, VERSION_STRING


def create_app(config_class=Config, config_overrides=None) -> Flask:
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "build"),
        static_url_path="/",
    )
    app.config.from_object(config_class)
    if config_overrides:
        app.config.update(config_overrides)

    _configure_logging(app)
    _init_extensions(app)
    _create_directories(app)
    _register_blueprints(app)
    _register_extra_routes(app)
    _register_frontend_catch_all(app)
    _register_error_handlers(app)
    _register_security_headers(app)

    # Skip in test mode — TestConfig sets TESTING=True.  Tests create the schema
    # themselves in their fixtures and do not need GPU services running.
    # In all other contexts (Docker via entrypoint.sh, local dev via run.py)
    # TESTING is unset, so the full initialization runs here.
    if not app.config.get("TESTING", False):
        with app.app_context():
            db.create_all()
        _ensure_admin(app)
        _init_services(app)

    return app


def _configure_logging(app):
    level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _init_extensions(app):
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)

    from flask_cors import CORS
    CORS(
        app,
        origins=app.config.get("CORS_ORIGINS", ["*"]),
        supports_credentials=app.config.get("CORS_SUPPORTS_CREDENTIALS", True),
    )

    limiter.init_app(app)

    @jwt.unauthorized_loader
    def missing_token(_r): return jsonify({"error": "Authentication required."}), 401

    @jwt.invalid_token_loader
    def invalid_token(_r): return jsonify({"error": "Invalid token."}), 401

    @jwt.expired_token_loader
    def expired_token(_h, _d): return jsonify({"error": "Token expired."}), 401

    from flask_jwt_extended.exceptions import CSRFError

    @app.errorhandler(CSRFError)
    def handle_csrf_error(e): return jsonify({"error": "CSRF token missing or invalid."}), 401

    with app.app_context():
        import auth.models   # noqa: F401
        import face.models   # noqa: F401

        db.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        db.session.commit()


def _create_directories(app):
    for key in ("VIOLATIONS_IMAGE_DIR", "TEMP_DIR"):
        os.makedirs(app.config[key], exist_ok=True)


def _register_blueprints(app):
    from auth.routes import auth_bp
    from streams.routes import stream_bp
    from violations.routes import violations_bp
    from face.routes import face_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(stream_bp)
    app.register_blueprint(violations_bp)
    app.register_blueprint(face_bp)


def _register_extra_routes(app):
    @app.get("/health")
    def health():
        yolo        = app.extensions.get("yolo_service")
        face        = app.extensions.get("face_pipeline")
        task_queue  = app.extensions.get("task_queue")
        return jsonify({
            "status":        "healthy",
            "version":       __version__,
            "model_loaded":  yolo.is_loaded if yolo else False,
            "face_ready":    face.is_ready  if face else False,
            "device":        yolo._device   if yolo else "unknown",
            "queue_dropped": task_queue.dropped_count if task_queue else 0,
        })

    @app.get("/version")
    def version():
        return jsonify({
            "version":        __version__,
            "version_string": VERSION_STRING,
        })

    @app.post("/upload/video")
    def upload_video():
        if "video" not in request.files:
            return jsonify({"error": "No video file (field: 'video')."}), 400
        video_file = request.files["video"]
        ext = os.path.splitext(video_file.filename)[1].lower()
        allowed = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        if ext not in allowed:
            return jsonify({"error": f"Unsupported format. Allowed: {allowed}"}), 400
        temp_dir = app.config["TEMP_DIR"]
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(temp_dir, filename)
        video_file.save(save_path)
        return jsonify({"path": save_path, "filename": filename}), 201

    @app.get("/violators/unknown")
    def get_unknown_violators():
        from face.models import Violation, FaceIdentity
        unconfirmed_ids = [
            i.id for i in FaceIdentity.query.filter_by(is_confirmed=False).all()
        ]
        if not unconfirmed_ids:
            return jsonify([]), 200

        unknowns = (
            Violation.query
            .filter(Violation.identity_id.in_(unconfirmed_ids))
            .order_by(Violation.timestamp.desc())
            .limit(200)
            .all()
        )
        return jsonify([v.to_dict() for v in unknowns]), 200


def _register_frontend_catch_all(app):
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        static = app.static_folder
        if path and os.path.exists(os.path.join(static, path)):
            return send_from_directory(static, path)
        return send_from_directory(static, "index.html")


def _register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(_e):     return jsonify({"error": "Not found."}), 404

    @app.errorhandler(405)
    def not_allowed(_e):   return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def server_error(_e):  return jsonify({"error": "Internal server error."}), 500


def _register_security_headers(app):
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        if not app.config.get("DEBUG"):
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


def _ensure_admin(app):
    from auth.utils import ensure_admin_user
    ensure_admin_user(app)


def _init_services(app):
    from detection.yolo_service import YOLOService
    from detection.batcher import YOLOBatcher
    from face.pipeline import InsightFacePipeline
    from tasks.queue import TaskQueue
    from streams.manager import StreamManager

    yolo = YOLOService()
    yolo.init_app(app)
    app.extensions["yolo_service"] = yolo

    batcher = YOLOBatcher()
    batcher.init_app(app)
    app.extensions["yolo_batcher"] = batcher

    pipeline = InsightFacePipeline(app.config)
    pipeline.init_app(app)
    app.extensions["face_pipeline"] = pipeline
    with app.app_context():
        pipeline.reload_cache()

    task_queue = TaskQueue()
    task_queue.init_app(app)
    app.extensions["task_queue"] = task_queue

    stream_manager = StreamManager()
    stream_manager.init_app(app)
    app.extensions["stream_manager"] = stream_manager

    import atexit
    atexit.register(lambda: (stream_manager.stop_all(), task_queue.shutdown(), batcher.shutdown()))