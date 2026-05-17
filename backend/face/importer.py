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

_TARGET_FACE_PX: int = 250


def _load_bgr(path: str) -> Optional[np.ndarray]:
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
    Two-stage embedding.

    Stage 1: detect face bbox in full image.
    Stage 2: crop person-scale region, resize to _TARGET_FACE_PX, embed.

    Returns the best Stage-2 detection dict, or None if either stage fails.
    Never falls back to Stage-1 embedding — Stage-1 embeddings are generated
    from full photos where the face is small in detection space, producing
    imprecise ArcFace alignment incompatible with runtime embeddings.
    """

    # ── Stage 1: locate face bbox ──────────────────────────────────────────────
    stage1 = pipeline.embed_crop(image_bgr)
    if not stage1:
        logger.warning("Stage-1: no face detected in full image — skipping.")
        return None

    if len(stage1) == 1:
        selected = stage1[0]
    else:
        def _area(f: dict) -> float:
            x1, y1, x2, y2 = f["bbox"]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)
        selected = max(stage1, key=_area)
        logger.warning(
            "Stage-1: %d faces detected — selected largest (area=%.0f px²). "
            "Use single-person photos for reliable import.",
            len(stage1), _area(selected),
        )

    logger.info(
        "Stage-1: det_score=%.3f  quality=%.3f  bbox=%s",
        selected["det_score"], selected["quality_score"],
        [round(v, 1) for v in selected["bbox"]],
    )

    # ── Stage 2: build person-scale crop and resize ───────────────────────────
    x1, y1, x2, y2 = [int(v) for v in selected["bbox"]]
    ih, iw = image_bgr.shape[:2]

    face_h = max(1, y2 - y1)
    face_w = max(1, x2 - x1)

    pad_x        = max(1, int(face_w * 0.30))
    pad_y_top    = max(1, int(face_h * 0.15))
    pad_y_bottom = max(1, int(face_h * 1.50))

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y_top)
    cx2 = min(iw, x2 + pad_x)
    cy2 = min(ih, y2 + pad_y_bottom)

    crop = image_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        logger.warning("Stage-2: person-scale crop is empty — skipping.")
        return None

    scale  = _TARGET_FACE_PX / face_h
    new_h  = max(1, int(crop.shape[0] * scale))
    new_w  = max(1, int(crop.shape[1] * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    crop   = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    logger.info(
        "Stage-2 crop: face_h=%dpx  scale=%.3f  crop=%dx%d  "
        "face_in_det_space≈%dpx  InsightFace %s",
        face_h, scale, new_w, new_h,
        int(_TARGET_FACE_PX * (640.0 / new_h)),
        "downscales ✓" if new_h >= 640 else f"UPSCALES {640/new_h:.2f}× ← quality loss risk",
    )

    # ── Stage 2: embed ────────────────────────────────────────────────────────
    stage2 = pipeline.embed_crop(crop)
    if not stage2:
        logger.warning(
            "Stage-2: embed_crop returned no detections "
            "(crop=%dx%d, face_h=%dpx, scale=%.3f) — skipping image. "
            "Do NOT fall back to Stage-1 embedding.",
            new_w, new_h, face_h, scale,
        )
        return None

    best = max(stage2, key=lambda f: f["quality_score"])
    logger.info(
        "Stage-2: det_score=%.3f  quality=%.3f",
        best["det_score"], best["quality_score"],
    )
    return best


def run_import(zip_bytes: bytes, pipeline, config, max_mb: int = 100) -> Dict[str, Any]:
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
            identity_path = os.path.join(storage_path, identity_name)
            os.makedirs(identity_path, exist_ok=True)

            dest_paths: List[str] = []
            for src_path, file_name in identity_data:
                dest = os.path.join(identity_path, file_name)
                shutil.copy2(src_path, dest)
                dest_paths.append(dest)

            identity_row = FaceIdentity.query.filter_by(label=identity_name).first()
            if not identity_row:
                identity_row = FaceIdentity(label=identity_name, is_confirmed=True)
                db.session.add(identity_row)
                db.session.flush()

            stored = 0
            for dest_path in dest_paths:
                fname = os.path.basename(dest_path)
                image = _load_bgr(dest_path)
                if image is None:
                    logger.warning("Could not decode image: %s", dest_path)
                    continue

                try:
                    best = _embed_face_crop(image, pipeline)
                except Exception as exc:
                    logger.error("_embed_face_crop raised for %s: %s", fname, exc)
                    errors.append(f"embed_face_crop failed for '{fname}' in '{identity_name}': {exc}")
                    continue

                # None means Stage 2 failed — do not store anything for this image.
                if best is None:
                    logger.warning(
                        "No valid Stage-2 embedding for '%s' in '%s' — image skipped.",
                        fname, identity_name,
                    )
                    continue

                emb_row = FaceEmbedding(
                    identity_id=identity_row.id,
                    det_score=best["det_score"],
                    quality_score=best["quality_score"],
                )
                emb_row.embedding = best["embedding"]
                db.session.add(emb_row)
                stored += 1
                logger.info(
                    "Stored Stage-2 embedding for '%s' from '%s'  "
                    "det=%.3f  quality=%.3f  norm=%.6f",
                    identity_name, fname,
                    best["det_score"], best["quality_score"],
                    float(np.linalg.norm(best["embedding"])),
                )

            if stored == 0:
                db.session.rollback()
                failed_identities += 1
                errors.append(
                    f"No valid Stage-2 embeddings extracted for '{identity_name}'. "
                    f"Check that images are clear, well-lit, single-person photos."
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