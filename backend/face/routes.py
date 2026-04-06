import os
import uuid

import cv2
import face_recognition
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import get_jwt_identity

from extensions import db
from face.models import FaceIdentity, FaceEmbedding
from face.models import Violation
from middleware.auth_required import jwt_required_route, admin_required

face_bp = Blueprint("face", __name__, url_prefix="/api/v1/face")


def _get_pipeline():
    return current_app.extensions["face_pipeline"]


# ---------------------------------------------------------------------------
# Enroll a new identity from a clean uploaded image
# ---------------------------------------------------------------------------
@face_bp.post("/enroll")
@jwt_required_route()
def enroll():
    """
    Enroll a new (or update an existing) identity from a clean face photograph.
    Accepts multipart/form-data with:
        - file  : image file (jpg/png)
        - name  : person's display name
    """
    pipeline = _get_pipeline()

    if "file" not in request.files:
        return jsonify({"error": "Image file is required (field: 'file')."}), 400

    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required (field: 'name')."}), 400

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        return jsonify({"error": "Only JPG and PNG images are accepted."}), 400

    # Decode the uploaded image in-memory (never trust the extension alone)
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image file."}), 400
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    encoding = pipeline.encode_image(img_rgb)
    if encoding is None:
        return jsonify({
            "error": "No face detected in the uploaded image. "
                     "Please use a clear, frontal photograph without heavy occlusion."
        }), 422

    # Upsert the identity
    user_id = get_jwt_identity()
    identity = FaceIdentity.query.filter_by(name=name).first()
    if not identity:
        identity = FaceIdentity(name=name, created_by=user_id)
        db.session.add(identity)
        db.session.flush()  # get identity.id before commit

    # Save the source image to known_faces for record-keeping
    known_faces_dir = current_app.config["KNOWN_FACES_DIR"]
    os.makedirs(known_faces_dir, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_")
    img_filename = f"{safe_name}_{uuid.uuid4().hex[:8]}.jpg"
    img_path = os.path.join(known_faces_dir, img_filename)
    cv2.imwrite(img_path, img_bgr)

    pipeline.add_embedding(
        identity_id=identity.id,
        encoding=encoding,
        source_image=img_filename,
    )

    return jsonify({
        "message": f"Identity '{name}' enrolled successfully.",
        "identity": identity.to_dict(),
    }), 201


# ---------------------------------------------------------------------------
# Merge unknown violations into a named identity
# ---------------------------------------------------------------------------
@face_bp.post("/merge")
@jwt_required_route()
def merge_faces():
    """
    Assign a name to one or more "Unknown Person" violations.
    Attempts to extract a face embedding from each violation's head crop;
    only stores embeddings that pass the quality threshold.

    Body: { "name": str, "violation_ids": [str, ...] }
    """
    pipeline = _get_pipeline()
    data = request.get_json(silent=True) or {}

    name = (data.get("name") or "").strip()
    violation_ids = data.get("violation_ids") or []

    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not violation_ids:
        return jsonify({"error": "violation_ids list is required."}), 400

    user_id = get_jwt_identity()

    # Upsert the identity
    identity = FaceIdentity.query.filter_by(name=name).first()
    if not identity:
        identity = FaceIdentity(name=name, created_by=user_id)
        db.session.add(identity)
        db.session.flush()

    violations = Violation.query.filter(Violation.id.in_(violation_ids)).all()
    if not violations:
        return jsonify({"error": "No matching violations found."}), 404

    violations_image_dir = current_app.config["VIOLATIONS_IMAGE_DIR"]
    embeddings_added = 0

    for violation in violations:
        violation.identity_id = identity.id
        violation.raw_name = name

        # Attempt to extract an embedding from the stored violation crop
        if violation.image_filename:
            img_path = os.path.join(violations_image_dir, violation.image_filename)
            if os.path.exists(img_path):
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    encoding = pipeline.encode_image(img_rgb)
                    if encoding is not None:
                        quality = pipeline.compute_quality_score(encoding, identity.id)
                        if quality >= current_app.config["FACE_MIN_QUALITY_SCORE"]:
                            pipeline.add_embedding(
                                identity_id=identity.id,
                                encoding=encoding,
                                source_image=violation.image_filename,
                                quality_score=quality,
                            )
                            embeddings_added += 1

    db.session.commit()
    # Final cache reload in case add_embedding was not called (no quality crops)
    pipeline.reload_cache()

    return jsonify({
        "message": f"Merged {len(violations)} violations into identity '{name}'.",
        "identity": identity.to_dict(),
        "embeddings_added": embeddings_added,
    }), 200


# ---------------------------------------------------------------------------
# List all known identities
# ---------------------------------------------------------------------------
@face_bp.get("/identities")
@jwt_required_route()
def list_identities():
    identities = FaceIdentity.query.order_by(FaceIdentity.name).all()
    return jsonify([i.to_dict() for i in identities]), 200


# ---------------------------------------------------------------------------
# Delete an identity and all its embeddings
# ---------------------------------------------------------------------------
@face_bp.delete("/identity/<identity_id>")
@admin_required()
def delete_identity(identity_id: str):
    identity = FaceIdentity.query.get(identity_id)
    if not identity:
        return jsonify({"error": "Identity not found."}), 404

    db.session.delete(identity)
    db.session.commit()
    _get_pipeline().reload_cache()

    return jsonify({"message": f"Identity '{identity.name}' deleted."}), 200
