import os

from flask import Blueprint, request, jsonify, send_from_directory, current_app

from extensions import db
from face.models import Violation

violations_bp = Blueprint("violations", __name__, url_prefix="/violations")


@violations_bp.get("")
def get_violations():
    """
    Return violations as a flat list (what the frontend expects).
    Supports optional query params: page, limit, name, stream.
    """
    page = max(1, request.args.get("page", 1, type=int))
    limit = min(200, max(1, request.args.get("limit", 50, type=int)))
    name_filter = request.args.get("name", "").strip()
    stream_filter = request.args.get("stream", "").strip()

    query = Violation.query.order_by(Violation.timestamp.desc())

    if name_filter:
        # Join to FaceIdentity and filter by label (raw_name no longer exists)
        from face.models import FaceIdentity
        query = query.join(FaceIdentity, Violation.identity_id == FaceIdentity.id, isouter=True)\
                     .filter(FaceIdentity.label.ilike(f"%{name_filter}%"))
    if stream_filter:
        query = query.filter(Violation.stream_id == stream_filter)

    paginated = query.paginate(page=page, per_page=limit, error_out=False)

    # Frontend does violations.map(v => ...) — expects a flat array, not a wrapped object
    return jsonify([v.to_dict() for v in paginated.items]), 200


@violations_bp.post("/clear")
def clear_violations():
    """Delete all violation records and their images."""
    violations_dir = current_app.config["VIOLATIONS_IMAGE_DIR"]

    violations = Violation.query.all()
    for v in violations:
        if v.image_filename:
            img_path = os.path.join(violations_dir, v.image_filename)
            if os.path.exists(img_path):
                os.remove(img_path)
        db.session.delete(v)

    db.session.commit()
    return jsonify({"message": f"Cleared {len(violations)} violations."}), 200


@violations_bp.get("/image/<filename>")
def get_violation_image(filename: str):
    """Serve a violation image by UUID filename."""
    if os.sep in filename or "/" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename."}), 400

    violations_dir = current_app.config["VIOLATIONS_IMAGE_DIR"]
    return send_from_directory(violations_dir, filename)