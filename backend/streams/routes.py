import os
import uuid

from flask import Blueprint, request, jsonify, Response, current_app

stream_bp = Blueprint("streams", __name__, url_prefix="/stream")


def _manager():
    return current_app.extensions["stream_manager"]


@stream_bp.post("/start")
def start_stream():
    data = request.get_json(silent=True) or {}
    source_type = data.get("source_type")
    source_path = data.get("source_path")
    name = (data.get("name") or "Unnamed Stream").strip()

    if not source_type or source_path is None:
        return jsonify({"error": "source_type and source_path are required."}), 400

    # Accept 'video' as alias for 'file' (frontend uses 'video' for uploaded files)
    if source_type == "video":
        source_type = "file"

    allowed_types = {"webcam", "rtsp", "file"}
    if source_type not in allowed_types:
        return jsonify({"error": f"source_type must be one of: {allowed_types}"}), 400

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
def stop_stream():
    data = request.get_json(silent=True) or {}
    stream_id = data.get("stream_id")
    if not stream_id:
        return jsonify({"error": "stream_id is required."}), 400

    found = _manager().stop(stream_id)
    if not found:
        return jsonify({"error": "Stream not found."}), 404

    return jsonify({"status": "stopped", "stream_id": stream_id}), 200


# Frontend calls /stream/video_feed/<id>
@stream_bp.get("/video_feed/<stream_id>")
def video_feed(stream_id: str):
    manager = _manager()
    if not manager.get_stats(stream_id):
        return jsonify({"error": "Stream not found."}), 404

    return Response(
        manager.frame_generator(stream_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@stream_bp.get("/detections/<stream_id>")
def get_detections(stream_id: str):
    stats = _manager().get_stats(stream_id)
    if stats is None:
        return jsonify({"error": "Stream not found."}), 404
    return jsonify(stats), 200


@stream_bp.get("s")  # resolves to GET /streams
def list_streams():
    return jsonify(_manager().list_streams()), 200
