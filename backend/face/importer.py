# Safion - backend/face/importer.py

import os
import zipfile
import tempfile
import logging
import shutil
from typing import Dict, List, Any, Tuple

import numpy as np
from PIL import Image
import cv2

from face.models import FaceIdentity, FaceEmbedding
from extensions import db

logger = logging.getLogger(__name__)


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

    For each identity folder:
      1. Copy images to persistent IDENTITY_STORAGE_PATH.
      2. Find or create a FaceIdentity row keyed by `label` (not `name`).
      3. Run embed_crop() on each image — identical to the live pipeline path.
      4. Store each successful embedding as an individual FaceEmbedding row via
         the model's `.embedding` setter, which handles L2 normalisation and
         pgvector serialisation.

    Design decisions that match the rest of the system:
      - Individual rows per source image (not an averaged blob) so that
        _build_prototypes() and DBSCAN clustering have realistic data density.
      - FaceEmbedding.embedding setter is the single normalisation point;
        raw numpy arrays are passed in, never .tobytes().
      - filter_by(label=…) is idempotent: re-running the import on an existing
        identity appends more embeddings rather than duplicating the identity row.
      - db.session.flush() after creating a new FaceIdentity gives us the PK
        before writing child FaceEmbedding rows, without committing early.
      - db.session.rollback() on zero-embedding identities prevents orphan
        FaceIdentity rows with no associated embeddings entering the gallery.
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
            # FIX: FaceIdentity uses `label`, NOT `name`. Querying by `name`
            # raises "namespace has no property 'name'" and crashes the import.
            identity_row = FaceIdentity.query.filter_by(label=identity_name).first()
            if not identity_row:
                identity_row = FaceIdentity(
                    label=identity_name,
                    is_confirmed=True,
                )
                db.session.add(identity_row)
                # flush() populates identity_row.id for the FK on FaceEmbedding
                # without committing — lets us roll back cleanly if no faces found.
                db.session.flush()

            # ── 3. Extract embeddings from each image ─────────────────────────
            stored = 0
            for dest_path in dest_paths:
                image = cv2.imread(dest_path)
                if image is None:
                    logger.warning("Could not decode image: %s", dest_path)
                    continue

                try:
                    # embed_crop() mirrors the live detection path:
                    #   RetinaFace detection → ArcFace embedding → L2 normalise
                    #   → quality gate (FACE_DET_SCORE_MIN / EMBEDDING_QUALITY_MIN)
                    # Returns [] when no face passes the quality threshold.
                    faces = pipeline.embed_crop(image)
                except Exception as exc:
                    fname = os.path.basename(dest_path)
                    logger.error("embed_crop failed for %s: %s", fname, exc)
                    errors.append(
                        f"embed_crop failed for '{fname}' in '{identity_name}': {exc}"
                    )
                    continue

                if not faces:
                    logger.debug("No detectable face in %s", dest_path)
                    continue

                # Use the highest-quality detection from this image.
                best = max(faces, key=lambda f: f["quality_score"])

                # ── 4. Store via FaceEmbedding — DO NOT assign to FaceIdentity ─
                # FIX: FaceIdentity has NO `embedding` column.  Embeddings live in
                # the FaceEmbedding table.  Assigning directly to FaceIdentity
                # (or storing raw .tobytes()) bypasses the normalisation setter and
                # produces vectors incompatible with pgvector cosine matching.
                #
                # FaceEmbedding.embedding (property setter):
                #   arr = np.asarray(value, float32).ravel()
                #   norm = np.linalg.norm(arr)
                #   self.embedding_vec = (arr / norm if norm > 0 else arr).tolist()
                #
                # Passing the already-L2-normalised ndarray from embed_crop()
                # through the setter is safe — normalising a unit vector is a no-op.
                emb_row = FaceEmbedding(
                    identity_id=identity_row.id,
                    det_score=best["det_score"],
                    quality_score=best["quality_score"],
                )
                emb_row.embedding = best["embedding"]   # setter: normalise + store
                db.session.add(emb_row)
                stored += 1

            # ── 5. Commit or roll back ─────────────────────────────────────────
            if stored == 0:
                # No usable faces — roll back any new FaceIdentity row so we
                # don't leave orphan identity entries with zero embeddings.
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
