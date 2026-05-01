# Safion - backend/face/importer.py

import os
import zipfile
import tempfile
import logging
import shutil
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import cv2

from face.models import FaceIdentity, FaceEmbedding
from extensions import db

logger = logging.getLogger(__name__)


def _load_bgr(path: str) -> Optional[np.ndarray]:
    """
    Read image file respecting EXIF orientation. Returns BGR uint8 array or None.
    cv2.imread() ignores EXIF rotation; mobile photos stored sideways cause
    RetinaFace misalignment and garbage ArcFace embeddings.
    """
    try:
        with Image.open(path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        logger.warning("PIL read failed for %s (%s) — falling back to cv2.imread", path, exc)
        return cv2.imread(path)


def _embed_face_crop(image_bgr: np.ndarray, pipeline) -> Optional[dict]:
    """
    Two-stage embedding that matches the runtime person-crop detection context.

    Root cause of the previous mismatch
    ------------------------------------
    The prior Stage-2 used a tight face crop (30 % padding), making the face
    fill ~60-70 % of the input to embed_crop().  With det_size=(640,640) that
    puts the face at ~490 px in detection space — above the effective anchor
    range of RetinaFace (buffalo_l anchors top out around 320 px for reliable
    landmark regression).  When RetinaFace fails to detect at that scale,
    embed_crop() returns an empty list and the code falls back to the Stage-1
    embedding: full photo → face is ~32 px in detection space → poor landmark
    precision → effectively random ArcFace output.

    At runtime, YOLO person crops have the face in the upper ~25-40 % of the
    crop height, placing the face at ~160-250 px in the 640-px detection window.
    That is the "good zone" for RetinaFace: det_score ≥ 0.9, landmark RMSE < 2 px.

    Fix
    ---
    Stage 2 now builds a person-crop-scale region:
      - small clearance above the face  (~15 % of face height)
      - body context below the face     (~150 % of face height)
      - symmetric horizontal padding    (~30 % of face width)

    This places the face at ~33 % of the crop height, matching runtime person
    crops, so RetinaFace sees the same effective scale in both paths.

    Face selection (deterministic, unchanged from prior version)
    ------------------------------------------------------------
    · Zero detections → return None
    · Single detection → use it
    · Multiple detected → select by largest bbox area (most prominent person).
    """

    # ── Stage 1: detect face(s) in full image ─────────────────────────────────
    detections = pipeline.embed_crop(image_bgr)

    if not detections:
        return None

    if len(detections) == 1:
        selected = detections[0]
    else:
        def _bbox_area(f: dict) -> float:
            x1, y1, x2, y2 = f["bbox"]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        selected = max(detections, key=_bbox_area)
        logger.warning(
            "Multiple faces detected (%d) — selected largest (area=%.0f px²). "
            "Use single-person photos for reliable import.",
            len(detections), _bbox_area(selected),
        )

    # ── Stage 2: re-embed using a person-crop-scale region ───────────────────
    # Build a crop where the face occupies ~33 % of the crop height so that
    # RetinaFace operates in the same anchor range as the runtime stream path.
    x1, y1, x2, y2 = [int(v) for v in selected["bbox"]]
    ih, iw = image_bgr.shape[:2]

    face_h = max(1, y2 - y1)
    face_w = max(1, x2 - x1)

    pad_x        = max(1, int(face_w * 0.30))   # 30 % horizontal (unchanged)
    pad_y_top    = max(1, int(face_h * 0.15))   # small head clearance above
    pad_y_bottom = max(1, int(face_h * 1.50))   # body context below face

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y_top)
    cx2 = min(iw, x2 + pad_x)
    cy2 = min(ih, y2 + pad_y_bottom)

    person_scale_crop = image_bgr[cy1:cy2, cx1:cx2]

    if person_scale_crop.size == 0:
        logger.debug("Person-scale crop is empty — using stage-1 embedding as fallback")
        return selected

    crop_detections = pipeline.embed_crop(person_scale_crop)

    if not crop_detections:
        logger.debug("Stage-2 re-embed returned no faces — using stage-1 embedding")
        return selected

    # Within this crop the face is the largest object; quality_score is a
    # meaningful signal — pick the highest.
    return max(crop_detections, key=lambda f: f["quality_score"])


def run_import(zip_bytes: bytes, pipeline, config, max_mb: int = 100) -> Dict[str, Any]:
    """Run the identity import process from a ZIP archive."""
    max_bytes = max_mb * 1024 * 1024
    if len(zip_bytes) > max_bytes:
        raise ValueError(f"ZIP file exceeds maximum size of {max_mb}MB")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "import.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        identities = _extract_identities_from_zip(zip_path, temp_dir)
        return _import_identities(identities, pipeline, config)


def _extract_identities_from_zip(zip_path: str, temp_dir: str) -> List[Dict[str, Any]]:
    """Extract identities from ZIP archive."""
    identities = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

        for root, dirs, files in os.walk(temp_dir):
            for dir_name in dirs:
                identity_folder = os.path.join(root, dir_name)
                identity_data = _process_identity_folder(identity_folder)
                if identity_data:
                    identities.append({
                        "name": dir_name,
                        "data": identity_data,
                    })
    return identities


def _process_identity_folder(identity_folder: str) -> List[Tuple[str, str]]:
    """
    Validate images in an identity folder.
    Returns list of (absolute_file_path, file_name) for each valid image.
    """
    faces = []
    for file in os.listdir(identity_folder):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        file_path = os.path.join(identity_folder, file)
        try:
            img = Image.open(file_path)
            img.verify()
            faces.append((file_path, file))
        except Exception as e:
            logger.warning("Invalid image %s: %s", file_path, e)
    return faces


def _import_identities(
    identities: List[Dict[str, Any]],
    pipeline,
    config,
) -> Dict[str, Any]:
    """
    Import validated identities into the database.

    Embedding pipeline per image:
      _load_bgr()          — EXIF-aware read → BGR uint8
      _embed_face_crop()   — Stage-1 detect in full image, select primary face
                           — Stage-2 build person-crop-scale region, re-embed
                           — Face now at ~33 % of crop height, matching runtime
      FaceEmbedding setter — L2 normalise + pgvector serialisation
    """
    total_identities  = 0
    total_faces       = 0
    failed_identities = 0
    errors: List[str] = []

    storage_path = config.get("IDENTITY_STORAGE_PATH", "identity_storage")
    os.makedirs(storage_path, exist_ok=True)

    for identity in identities:
        total_identities += 1
        identity_name: str = identity["name"]
        identity_data: List[Tuple[str, str]] = identity["data"]

        try:
            # ── 1. Copy source images to persistent storage ────────────────────
            identity_path = os.path.join(storage_path, identity_name)
            os.makedirs(identity_path, exist_ok=True)

            dest_paths: List[str] = []
            for src_path, file_name in identity_data:
                dest = os.path.join(identity_path, file_name)
                shutil.copy2(src_path, dest)
                dest_paths.append(dest)

            # ── 2. Find or create FaceIdentity keyed by `label` ───────────────
            identity_row = FaceIdentity.query.filter_by(label=identity_name).first()
            if not identity_row:
                identity_row = FaceIdentity(
                    label=identity_name,
                    is_confirmed=True,
                )
                db.session.add(identity_row)
                db.session.flush()

            # ── 3. Extract embeddings via two-stage _embed_face_crop() ─────────
            stored = 0
            for dest_path in dest_paths:
                image = _load_bgr(dest_path)
                if image is None:
                    logger.warning("Could not decode image: %s", dest_path)
                    continue

                try:
                    best = _embed_face_crop(image, pipeline)
                except Exception as exc:
                    fname = os.path.basename(dest_path)
                    logger.error("_embed_face_crop failed for %s: %s", fname, exc)
                    errors.append(
                        f"embed_face_crop failed for '{fname}' in '{identity_name}': {exc}"
                    )
                    continue

                if best is None:
                    logger.debug("No detectable face in %s", dest_path)
                    continue

                emb_row = FaceEmbedding(
                    identity_id=identity_row.id,
                    det_score=best["det_score"],
                    quality_score=best["quality_score"],
                )
                emb_row.embedding = best["embedding"]
                db.session.add(emb_row)
                stored += 1

            # ── 4. Commit or roll back ─────────────────────────────────────────
            if stored == 0:
                db.session.rollback()
                failed_identities += 1
                errors.append(
                    f"No valid face embeddings could be extracted for '{identity_name}'"
                )
                continue

            db.session.commit()
            total_faces += len(dest_paths)

        except Exception as exc:
            db.session.rollback()
            logger.error("Error importing identity '%s': %s", identity_name, exc)
            failed_identities += 1
            errors.append(f"Failed to import '{identity_name}': {exc}")

    return {
        "total":          total_identities,
        "successful":     total_identities - failed_identities,
        "failed":         failed_identities,
        "errors":         errors,
        "faces_processed": total_faces,
    }