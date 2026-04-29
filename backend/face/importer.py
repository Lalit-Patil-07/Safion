# Safion - backend/face/importer.py

import os
import zipfile
import tempfile
import logging
from io import BytesIO
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
    # Validate ZIP size
    max_bytes = max_mb * 1024 * 1024
    if len(zip_bytes) > max_bytes:
        raise ValueError(f"ZIP file exceeds maximum size of {max_mb}MB")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "import.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        # Process the ZIP file
        identities = _extract_identities_from_zip(zip_path, temp_dir)
        return _import_identities(identities, pipeline, config)

def _extract_identities_from_zip(zip_path: str, temp_dir: str) -> List[Dict[str, Any]]:
    """Extract identities from ZIP archive."""
    identities = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all files
        zip_ref.extractall(temp_dir)

        # Process each folder as an identity
        for root, dirs, files in os.walk(temp_dir):
            for dir_name in dirs:
                identity_folder = os.path.join(root, dir_name)
                identity_data = _process_identity_folder(identity_folder)
                if identity_data:
                    identities.append({
                        "name": dir_name,
                        "data": identity_data
                    })
    return identities

def _process_identity_folder(identity_folder: str) -> List[Tuple[np.ndarray, str]]:
    """Process an identity folder and extract face images."""
    faces = []
    for file in os.listdir(identity_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(identity_folder, file)
            # Load image to verify it's valid
            try:
                img = Image.open(file_path)
                img.verify()
                faces.append((file_path, file))
            except Exception as e:
                logger.warning(f"Invalid image {file_path}: {e}")
                continue
    return faces

def _import_identities(identities: List[Dict[str, Any]], pipeline, config) -> Dict[str, Any]:
    """Import processed identities into the system."""
    total_identities = 0
    total_faces = 0
    failed_identities = 0
    errors = []

    storage_path = config.get("IDENTITY_STORAGE_PATH", "identity_storage")
    os.makedirs(storage_path, exist_ok=True)

    for identity in identities:
        total_identities += 1
        identity_name = identity["name"]
        identity_data = identity["data"]

        try:
            # Create identity directory
            identity_path = os.path.join(storage_path, identity_name)
            os.makedirs(identity_path, exist_ok=True)

            # Copy face images
            face_files = []
            for face_path, face_name in identity_data:
                dest_path = os.path.join(identity_path, face_name)
                shutil.copy2(face_path, dest_path)
                face_files.append(face_name)

            # Process faces and generate embeddings
            embeddings = []
            for face_name in face_files:
                face_path = os.path.join(identity_path, face_name)
                try:
                    # Preprocess image
                    image = cv2.imread(face_path)
                    if image is None:
                        continue

                    # Generate embedding using correct pipeline method
                    faces = pipeline.embed_crop(image)

                    if not faces:
                        continue

                    embedding = faces[0].get('embedding')
                    if embedding is not None:
                        embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error processing face {face_name}: {e}")
                    errors.append(f"Failed processing face {face_name} for {identity_name}: {str(e)}")

            if not embeddings:
                failed_identities += 1
                errors.append(f"No valid embeddings for {identity_name}")
                continue

            # Calculate average embedding
            avg_embedding = np.mean(embeddings, axis=0)

            # Create or update identity
            identity_row = FaceIdentity.query.filter_by(label=identity_name).first()
            if not identity_row:
                identity_row = FaceIdentity(label=identity_name, is_confirmed=True)
                db.session.add(identity_row)
                db.session.flush()

            emb_row = FaceEmbedding(identity_id=identity_row.id)
            emb_row.embedding = avg_embedding
            db.session.add(emb_row)

            total_faces += len(face_files)

        except Exception as e:
            logger.error(f"Error importing identity {identity_name}: {e}")
            failed_identities += 1
            errors.append(f"Failed to import {identity_name}: {str(e)}")

    successful_identities = total_identities - failed_identities

    return {
        "total": total_identities,
        "successful": successful_identities,
        "failed": failed_identities,
        "errors": errors,
        "faces_processed": total_faces,
    }