"""
Safion — Flask application factory
"""
import logging
import os
import uuid

from flask import Flask, send_from_directory, jsonify, request
from sqlalchemy import text

from config import Config
from extensions import db, bcrypt, jwt
from version import __version__, VERSION_STRING


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
    _register_extra_routes(app)
    _register_frontend_catch_all(app)
    _register_error_handlers(app)

    if not app.config.get("_SKIP_SERVICES", False):
        _init_services(app)

    _ensure_admin(app)

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

    @jwt.unauthorized_loader
    def missing_token(_r): return jsonify({"error": "Authentication required."}), 401

    @jwt.invalid_token_loader
    def invalid_token(_r): return jsonify({"error": "Invalid token."}), 401

    @jwt.expired_token_loader
    def expired_token(_h, _d): return jsonify({"error": "Token expired."}), 401

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


def _ensure_admin(app):
    from auth.utils import ensure_admin_user
    ensure_admin_user(app)