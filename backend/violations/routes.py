import os

from flask import Blueprint, request, jsonify, send_from_directory, current_app

from extensions import db
from face.models import Violation
from middleware.auth_required import jwt_required_route, admin_required

violations_bp = Blueprint("violations", __name__, url_prefix="/api/v1/violations")


@violations_bp.get("")
@jwt_required_route()
def get_violations():
    """
    Return paginated violations.
    Query params:
        page    (int, default 1)
        limit   (int, default 50, max 200)
        name    (str, optional filter — matches raw_name or identity.name)
        stream  (str, optional stream_id filter)
    """
    page = max(1, request.args.get("page", 1, type=int))
    limit = min(200, max(1, request.args.get("limit", 50, type=int)))
    name_filter = request.args.get("name", "").strip()
    stream_filter = request.args.get("stream", "").strip()

    query = Violation.query.order_by(Violation.timestamp.desc())

    if name_filter:
        query = query.filter(Violation.raw_name.ilike(f"%{name_filter}%"))
    if stream_filter:
        query = query.filter(Violation.stream_id == stream_filter)

    paginated = query.paginate(page=page, per_page=limit, error_out=False)

    return jsonify({
        "violations": [v.to_dict() for v in paginated.items],
        "total": paginated.total,
        "page": page,
        "pages": paginated.pages,
        "limit": limit,
    }), 200


@violations_bp.post("/clear")
@admin_required()
def clear_violations():
    """Delete all violation records and their images. Admin only."""
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
@jwt_required_route()
def get_violation_image(filename: str):
    """Serve a violation image by filename (UUID-based, no path traversal)."""
    # Reject any path separators — send_from_directory sanitises but belt + braces
    if os.sep in filename or "/" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename."}), 400

    violations_dir = current_app.config["VIOLATIONS_IMAGE_DIR"]
    return send_from_directory(violations_dir, filename)
