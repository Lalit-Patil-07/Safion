"""
Face and identity API endpoints.

Prefix: /face

Provides CRUD for face identities, merge suggestions, clustering triggers,
enrollment from photos, and bulk import from ZIP archives.
"""
import os
from datetime import datetime, timezone, timedelta
import logging
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import func

from extensions import db
from face.models import FaceIdentity, FaceEmbedding, Violation
from middleware.auth_required import protect_blueprint

face_bp = Blueprint("face", __name__, url_prefix="/face")
protect_blueprint(face_bp)


def _pipeline():
    return current_app.extensions["face_pipeline"]


# ── Dashboard stats ───────────────────────────────────────────────────────────
@face_bp.get("/stats")
def get_stats():
    """
    Dashboard summary numbers.
    Called once on dashboard load — not polled.
    """
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    total_identities   = FaceIdentity.query.filter_by(is_archived=False).count()
    confirmed_count    = FaceIdentity.query.filter_by(is_confirmed=True,  is_archived=False).count()
    unconfirmed_count  = FaceIdentity.query.filter_by(is_confirmed=False, is_archived=False).count()
    total_violations   = Violation.query.count()
    violations_today   = Violation.query.filter(Violation.timestamp >= today_start).count()
    repeat_offenders   = (
        db.session.query(Violation.identity_id)
        .filter(Violation.identity_id.isnot(None))
        .group_by(Violation.identity_id)
        .having(func.count() >= 3)
        .count()
    )

    return jsonify({
        "total_identities":  total_identities,
        "confirmed_count":   confirmed_count,
        "unconfirmed_count": unconfirmed_count,
        "total_violations":  total_violations,
        "violations_today":  violations_today,
        "repeat_offenders":  repeat_offenders,
    }), 200


# ── Identity list (paginated, filtered) ───────────────────────────────────────
@face_bp.get("/identities")
def list_identities():
    """
    Paginated identity list with search and filter.

    Query params:
        page      (int, default 1)
        limit     (int, default 24, max 100)
        search    (str, label contains)
        confirmed (true | false | all, default all)
        sort      (last_seen | created_at | label, default last_seen)
    """
    page      = max(1, request.args.get("page",  1,    type=int))
    limit     = min(100, max(1, request.args.get("limit", 24, type=int)))
    search    = request.args.get("search",    "").strip()
    confirmed = request.args.get("confirmed", "all").strip()
    sort      = request.args.get("sort",      "last_seen").strip()

    query = FaceIdentity.query.filter_by(is_archived=False)

    if search:
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = query.filter(FaceIdentity.label.ilike(f"%{escaped}%", escape="\\"))
    if confirmed == "true":
        query = query.filter_by(is_confirmed=True)
    elif confirmed == "false":
        query = query.filter_by(is_confirmed=False)

    if sort == "label":
        query = query.order_by(FaceIdentity.label)
    elif sort == "created_at":
        query = query.order_by(FaceIdentity.created_at.desc())
    else:  # last_seen
        query = query.order_by(FaceIdentity.last_seen.desc().nullslast())

    paginated = query.paginate(page=page, per_page=limit, error_out=False)

    items = []
    for identity in paginated.items:
        d = identity.to_summary()
        d["violation_count"] = identity.violations.count()
        d["embedding_count"] = identity.embeddings.count()
        items.append(d)

    return jsonify({
        "identities": items,
        "total":      paginated.total,
        "page":       page,
        "pages":      paginated.pages,
        "limit":      limit,
    }), 200


# ── Single identity ───────────────────────────────────────────────────────────
@face_bp.get("/identity/<identity_id>")
def get_identity(identity_id: str):
    identity = FaceIdentity.query.get(identity_id)
    if not identity:
        return jsonify({"error": "Identity not found."}), 404
    return jsonify(identity.to_dict()), 200


# ── Violations for one identity (timeline) ────────────────────────────────────
@face_bp.get("/identity/<identity_id>/violations")
def get_identity_violations(identity_id: str):
    """
    Violation timeline for an identity.
    Includes aggregated counts by violation type for the summary bar.
    """
    identity = FaceIdentity.query.get(identity_id)
    if not identity:
        return jsonify({"error": "Identity not found."}), 404

    violations = (
        Violation.query
        .filter_by(identity_id=identity_id)
        .order_by(Violation.timestamp.desc())
        .all()
    )

    type_counts: dict = {}
    for v in violations:
        type_counts[v.violation_type] = type_counts.get(v.violation_type, 0) + 1

    return jsonify({
        "identity":    identity.to_dict(),
        "violations":  [v.to_dict() for v in violations],
        "type_counts": type_counts,
    }), 200


# ── Identity similarity ───────────────────────────────────────────────────────
@face_bp.get("/identity/<identity_id>/similarity")
def get_identity_similarity(identity_id: str):
    """
    Similarity report for a single identity.

    Returns:
      - intra_similarity  : mean pairwise cosine similarity among this
                            identity's own stored embeddings.  High (>0.6)
                            means the stored embeddings are consistent.
                            Low (<0.3) indicates the identity contains
                            mixed or poor-quality embeddings.
      - prototype_norms   : list of L2 norms for each cached prototype.
                            All should be ~1.0; any value far from 1.0
                            indicates a normalisation bug in the pipeline.
      - similar           : up to `limit` other identities ranked by
                            prototype similarity, with their scores and
                            confirmation state.  Used by the frontend
                            "Potentially same person" panel.

    Query params:
        limit (int, default 3, max 20)
    """
    identity = FaceIdentity.query.get(identity_id)
    if not identity:
        return jsonify({"error": "Identity not found."}), 404

    limit = min(20, max(1, request.args.get("limit", 3, type=int)))

    # ── Intra-identity similarity ─────────────────────────────────────────────
    # Measures how consistent the stored embeddings are with each other.
    # A confirmed identity with good data should score > 0.55 for ArcFace.
    emb_rows = identity.embeddings.order_by(FaceEmbedding.quality_score.desc()).all()

    intra_similarity: float = 0.0
    prototype_norms:  list  = []

    if len(emb_rows) >= 2:
        vecs = np.stack([
            np.asarray(r.embedding_vec, dtype=np.float32) for r in emb_rows
        ])
        # Recompute norms to surface any normalisation drift from pgvector
        norms = np.linalg.norm(vecs, axis=1)
        # Re-normalise before computing similarities so result is always in [-1,1]
        safe = norms > 0
        vecs[safe] = vecs[safe] / norms[safe, None]

        sim_matrix = vecs @ vecs.T                 # (N, N) cosine similarity matrix
        n = len(vecs)
        # Extract upper triangle (excluding diagonal) — N*(N-1)/2 unique pairs
        upper_idx = np.triu_indices(n, k=1)
        pair_sims = sim_matrix[upper_idx]
        intra_similarity = float(np.mean(pair_sims)) if len(pair_sims) else 1.0
        prototype_norms  = norms.tolist()
    elif len(emb_rows) == 1:
        intra_similarity = 1.0   # trivially self-similar
        vec  = np.asarray(emb_rows[0].embedding_vec, dtype=np.float32)
        prototype_norms = [float(np.linalg.norm(vec))]

    # ── Similar identities from live cache ────────────────────────────────────
    # Pulls from the in-memory prototype cache — no additional DB queries.
    # Uses the same similarity formula as the merge-suggestion engine so the
    # scores are consistent with what the operator sees in the Suggestions page.
    pipeline    = _pipeline()
    snapshot    = pipeline._snapshot()
    this_entry  = snapshot.get(identity_id)
    similar_out: list = []

    if this_entry:
        from face.similarity import identity_similarity

        candidates = []
        for other_id, other_data in snapshot.items():
            if other_id == identity_id:
                continue
            sim = identity_similarity(this_entry, other_data)
            if sim > 0.0:
                candidates.append((other_id, other_data, sim))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for other_id, other_data, sim in candidates[:limit]:
            other_identity = FaceIdentity.query.get(other_id)
            similar_out.append({
                "identity_id":  other_id,
                "label":        other_data["label"],
                "similarity":   round(sim, 4),
                "is_confirmed": other_data.get("is_confirmed", False),
                "is_archived":  other_data.get("is_archived",  False),
                "thumbnail":    (
                    f"/violations/image/{other_identity.thumbnail_filename}"
                    if other_identity and other_identity.thumbnail_filename else None
                ),
            })

    # ── Cache entry diagnostics ───────────────────────────────────────────────
    # Expose prototype count and per-prototype norm so operators can confirm
    # that the runtime cache is healthy for this identity.
    cached_prototypes = []
    if this_entry:
        for i, proto in enumerate(this_entry.get("prototypes", [])):
            vec  = proto["vec"]
            norm = float(np.linalg.norm(vec))
            cached_prototypes.append({
                "index":  i,
                "weight": round(proto["weight"], 4),
                "count":  proto["count"],
                "norm":   round(norm, 6),   # should always be ~1.0
            })

    return jsonify({
        "identity_id":       identity_id,
        "label":             identity.label,
        "embedding_count":   len(emb_rows),
        "intra_similarity":  round(intra_similarity, 4),
        "prototype_norms":   [round(n, 6) for n in prototype_norms],
        "cached_prototypes": cached_prototypes,
        "cache_hit":         this_entry is not None,
        "similar":           similar_out,
    }), 200


# ── Rename / confirm ──────────────────────────────────────────────────────────
@face_bp.patch("/identity/<identity_id>")
def update_identity(identity_id: str):
    """
    Rename and/or confirm an identity.
    Body: {"label": str}  — sets is_confirmed=True automatically.
    """
    identity = FaceIdentity.query.get(identity_id)
    if not identity:
        return jsonify({"error": "Identity not found."}), 404

    data  = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip()
    if not label:
        return jsonify({"error": "label is required."}), 400

    existing = FaceIdentity.query.filter(
        FaceIdentity.label == label,
        FaceIdentity.id    != identity_id,
    ).first()
    if existing:
        return jsonify({"error": f"Label '{label}' already in use."}), 409

    identity.label        = label
    identity.is_confirmed = True
    db.session.commit()
    _pipeline().reload_cache()

    return jsonify({"message": "Identity updated.", "identity": identity.to_dict()}), 200


# ── Merge ─────────────────────────────────────────────────────────────────────
@face_bp.post("/identity/merge")
def merge_identities():
    """
    Merge source into target.
    Source is archived (not deleted) — all data moves to target.
    Body: {"source_id": str, "target_id": str}
    """
    data      = request.get_json(silent=True) or {}
    source_id = data.get("source_id", "").strip()
    target_id = data.get("target_id", "").strip()

    if not source_id or not target_id or source_id == target_id:
        return jsonify({"error": "source_id and target_id must differ."}), 400

    source = FaceIdentity.query.get(source_id)
    target = FaceIdentity.query.get(target_id)
    if not source: return jsonify({"error": "source_id not found."}), 404
    if not target: return jsonify({"error": "target_id not found."}), 404

    FaceEmbedding.query.filter_by(identity_id=source_id).update({"identity_id": target_id})
    Violation.query.filter_by(identity_id=source_id).update({"identity_id": target_id})

    source.is_archived    = True
    source.merged_into_id = target_id

    if source.thumbnail_filename and not target.thumbnail_filename:
        target.thumbnail_filename = source.thumbnail_filename

    db.session.commit()
    _pipeline().reload_cache()

    return jsonify({
        "message":  f"Merged '{source.label}' into '{target.label}'.",
        "identity": target.to_dict(),
    }), 200


# ── Delete ────────────────────────────────────────────────────────────────────
@face_bp.delete("/identity/<identity_id>")
def delete_identity(identity_id: str):
    """Hard delete — archives instead of destroying."""
    identity = FaceIdentity.query.get(identity_id)
    if not identity:
        return jsonify({"error": "Identity not found."}), 404
    identity.is_archived = True
    db.session.commit()
    _pipeline().reload_cache()
    return jsonify({"message": f"Identity '{identity.label}' archived."}), 200


# ── Trigger clustering ────────────────────────────────────────────────────────
@face_bp.post("/cluster")
def trigger_clustering():
    from face.clustering import run_clustering
    eps         = request.args.get("eps",        current_app.config["CLUSTER_EPS"],         type=float)
    min_samples = request.args.get("min_samples", current_app.config["CLUSTER_MIN_SAMPLES"], type=int)
    result      = run_clustering(eps=eps, min_samples=min_samples)
    _pipeline().reload_cache()
    return jsonify(result), 200


# ── Enroll from clean photo ───────────────────────────────────────────────────
@face_bp.post("/enroll")
def enroll():
    pipeline = _pipeline()
    if "file" not in request.files:
        return jsonify({"error": "file required."}), 400
    label = (request.form.get("label") or "").strip()
    if not label:
        return jsonify({"error": "label required."}), 400

    file     = request.files["file"]
    ext      = os.path.splitext(file.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        return jsonify({"error": "JPG or PNG only."}), 400

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image."}), 400

    faces = pipeline.embed_crop(img_bgr)
    if not faces:
        return jsonify({"error": "No face detected."}), 422
    if len(faces) > 1:
        return jsonify({"error": f"{len(faces)} faces found. Use a single-face photo."}), 422

    identity = FaceIdentity.query.filter_by(label=label).first()
    if not identity:
        identity = FaceIdentity(label=label, is_confirmed=True)
        db.session.add(identity)
        db.session.flush()

    emb_row           = FaceEmbedding(identity_id=identity.id, det_score=faces[0]["det_score"])
    emb_row.embedding = faces[0]["embedding"]
    db.session.add(emb_row)
    db.session.commit()
    pipeline.reload_cache()

    return jsonify({"message": f"'{label}' enrolled.", "identity": identity.to_dict()}), 201


# ── Legacy: merge from violation IDs ─────────────────────────────────────────
@face_bp.post("/merge")
def merge_from_violations():
    """Kept for frontend compat. Links violations to a named identity."""
    data          = request.get_json(silent=True) or {}
    label         = (data.get("name") or "").strip()
    violation_ids = data.get("violation_ids") or []

    if not label:          return jsonify({"error": "name required."}), 400
    if not violation_ids:  return jsonify({"error": "violation_ids required."}), 400

    identity = FaceIdentity.query.filter_by(label=label).first()
    if not identity:
        identity = FaceIdentity(label=label, is_confirmed=True)
        db.session.add(identity)
        db.session.flush()
    else:
        identity.is_confirmed = True

    violations = Violation.query.filter(Violation.id.in_(violation_ids)).all()
    if not violations:
        return jsonify({"error": "No matching violations found."}), 404

    for v in violations:
        v.identity_id = identity.id

    db.session.commit()
    _pipeline().reload_cache()

    return jsonify({
        "message":  f"Linked {len(violations)} violations to '{label}'.",
        "identity": identity.to_dict(),
    }), 200


# ── Review queue ──────────────────────────────────────────────────────────────
@face_bp.get("/review-queue")
def review_queue():
    """
    Returns unconfirmed, non-archived identities sorted by:
      1. violation_count DESC  (high-risk first)
      2. last_seen DESC        (recent activity first)
    """
    from sqlalchemy import func

    vcount = (
        db.session.query(
            Violation.identity_id,
            func.count(Violation.id).label("vcount"),
        )
        .group_by(Violation.identity_id)
        .subquery()
    )

    identities = (
        FaceIdentity.query
        .filter_by(is_confirmed=False, is_archived=False)
        .outerjoin(vcount, FaceIdentity.id == vcount.c.identity_id)
        .order_by(
            vcount.c.vcount.desc().nullslast(),
            FaceIdentity.last_seen.desc().nullslast(),
        )
        .limit(100)
        .all()
    )

    results = []
    for identity in identities:
        d = identity.to_summary()
        d["violation_count"] = identity.violations.count()
        d["embedding_count"] = identity.embeddings.count()

        recent_violations = (
            identity.violations
            .filter(Violation.image_filename.isnot(None))
            .order_by(Violation.timestamp.desc())
            .limit(2)
            .all()
        )
        d["preview_images"] = [
            f"/violations/image/{v.image_filename}"
            for v in recent_violations
        ]
        results.append(d)

    return jsonify(results), 200


# ── Merge suggestions ─────────────────────────────────────────────────────────
@face_bp.get("/merge-suggestions")
def get_merge_suggestions():
    """
    Return ranked merge suggestions for unconfirmed identities.
    Generated on demand from the live cache — always fresh, no DB reads.
    """
    from face.similarity import generate_merge_suggestions

    pipeline  = _pipeline()
    threshold = request.args.get(
        "threshold", current_app.config.get("SUGGESTION_THRESHOLD", 0.68), type=float
    )
    limit = request.args.get(
        "limit", current_app.config.get("SUGGESTION_MAX_RESULTS", 30), type=int
    )

    snapshot    = pipeline._snapshot()
    suggestions = generate_merge_suggestions(snapshot, threshold=threshold, max_results=limit)

    for s in suggestions:
        for key in ("identity_a_id", "identity_b_id"):
            identity  = FaceIdentity.query.get(s[key])
            thumb_key = "thumbnail_a" if key == "identity_a_id" else "thumbnail_b"
            s[thumb_key] = (
                f"/violations/image/{identity.thumbnail_filename}"
                if identity and identity.thumbnail_filename else None
            )

    return jsonify({
        "suggestions": suggestions,
        "count":       len(suggestions),
        "threshold":   threshold,
    }), 200


# ── Face samples for an identity ──────────────────────────────────────────────
@face_bp.get("/identity/<identity_id>/samples")
def get_identity_samples(identity_id: str):
    """
    Return the top-N clearest violation images for this identity, ordered by
    match_score DESC.
    """
    identity = FaceIdentity.query.get(identity_id)
    if not identity:
        return jsonify({"error": "Identity not found."}), 404

    samples = (
        Violation.query
        .filter_by(identity_id=identity_id)
        .filter(Violation.image_filename.isnot(None))
        .filter(Violation.match_score.isnot(None))
        .order_by(Violation.match_score.desc())
        .limit(6)
        .all()
    )

    if not samples:
        samples = (
            Violation.query
            .filter_by(identity_id=identity_id)
            .filter(Violation.image_filename.isnot(None))
            .order_by(Violation.timestamp.desc())
            .limit(6)
            .all()
        )

    return jsonify([
        {
            "image_path":  f"/violations/image/{v.image_filename}",
            "match_score": round(v.match_score, 3) if v.match_score else None,
            "timestamp":   v.timestamp.isoformat(),
        }
        for v in samples
    ]), 200


@face_bp.post("/import")
def import_identities():
    from face.importer import run_import

    if "file" not in request.files:
        return jsonify({"error": "Multipart field 'file' is required."}), 400

    zip_file = request.files["file"]
    if not zip_file.filename:
        return jsonify({"error": "No filename provided."}), 400
    if not zip_file.filename.lower().endswith(".zip"):
        return jsonify({"error": "Uploaded file must be a .zip archive."}), 400

    try:
        zip_bytes = zip_file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded ZIP: %s", exc)
        return jsonify({"error": "Failed to read uploaded file."}), 400

    pipeline = _pipeline()
    if not pipeline.is_ready:
        return jsonify({"error": "Face pipeline is not ready."}), 503

    max_mb = current_app.config.get("IMPORT_MAX_ZIP_MB", 100)

    try:
        result = run_import(
            zip_bytes=zip_bytes,
            pipeline=pipeline,
            config=current_app.config,
            max_mb=max_mb,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Import failed: %s", exc, exc_info=True)
        return jsonify({"error": "Import failed."}), 500

    try:
        pipeline.reload_cache()
    except Exception:
        pass

    status_code = 207 if (result["failed"] > 0 or result["errors"]) else 200
    return jsonify(result), status_code
