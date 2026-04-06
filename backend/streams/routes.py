import os
import uuid

from flask import Blueprint, request, jsonify, Response, current_app, send_from_directory

from middleware.auth_required import jwt_required_route

stream_bp = Blueprint("streams", __name__, url_prefix="/api/v1/stream")


def _manager():
    return current_app.extensions["stream_manager"]


@stream_bp.post("/start")
@jwt_required_route()
def start_stream():
    data = request.get_json(silent=True) or {}
    source_type = data.get("source_type")
    source_path = data.get("source_path")
    name = (data.get("name") or "Unnamed Stream").strip()

    if not source_type or source_path is None:
        return jsonify({"error": "source_type and source_path are required."}), 400

    allowed_types = {"webcam", "rtsp", "file"}
    if source_type not in allowed_types:
        return jsonify({"error": f"source_type must be one of: {allowed_types}"}), 400

    # Validate file paths to prevent directory traversal
    if source_type == "file":
        temp_dir = os.path.realpath(current_app.config["TEMP_DIR"])
        candidate = os.path.realpath(source_path)
        if not candidate.startswith(temp_dir):
            return jsonify({"error": "File path must be within the temp directory."}), 400

    try:
        result = _manager().start(source_type, str(source_path), name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 429

    return jsonify(result), 201


@stream_bp.post("/stop")
@jwt_required_route()
def stop_stream():
    data = request.get_json(silent=True) or {}
    stream_id = data.get("stream_id")
    if not stream_id:
        return jsonify({"error": "stream_id is required."}), 400

    found = _manager().stop(stream_id)
    if not found:
        return jsonify({"error": "Stream not found."}), 404

    return jsonify({"status": "stopped", "stream_id": stream_id}), 200


@stream_bp.get("/feed/<stream_id>")
@jwt_required_route()
def video_feed(stream_id: str):
    manager = _manager()
    if not manager.get_stats(stream_id):
        return jsonify({"error": "Stream not found."}), 404

    return Response(
        manager.frame_generator(stream_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@stream_bp.get("/detections/<stream_id>")
@jwt_required_route()
def get_detections(stream_id: str):
    stats = _manager().get_stats(stream_id)
    if stats is None:
        return jsonify({"error": "Stream not found."}), 404
    return jsonify(stats), 200


@stream_bp.get("s")  # GET /api/v1/streams
@jwt_required_route()
def list_streams():
    return jsonify(_manager().list_streams()), 200


@stream_bp.post("/upload/video")
@jwt_required_route()
def upload_video():
    """Upload a video file for local-file streaming."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided (field: 'video')."}), 400

    video_file = request.files["video"]
    ext = os.path.splitext(video_file.filename)[1].lower()
    allowed_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    if ext not in allowed_exts:
        return jsonify({"error": f"Unsupported video format. Allowed: {allowed_exts}"}), 400

    temp_dir = current_app.config["TEMP_DIR"]
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(temp_dir, filename)
    video_file.save(save_path)

    return jsonify({"path": save_path, "filename": filename}), 201
