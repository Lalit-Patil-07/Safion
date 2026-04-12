"""
TaskQueue — async violation pipeline
"""
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _now(): return datetime.now(timezone.utc)


@dataclass
class ViolationJob:
    stream_id:       str
    violation_type:  str
    confidence:      float
    person_crop_bgr: np.ndarray
    person_bbox:     list[float]


class TaskQueue:
    def __init__(self, num_workers: int = 2, maxsize: int = 200):
        self._q       = queue.Queue(maxsize=maxsize)
        self._workers = []
        self._stop    = threading.Event()
        self._app     = None
        self._n       = num_workers

        self._identity_cooldown: dict  = {}
        self._identity_cd_lock         = threading.Lock()
        self._identity_cooldown_s: int = 30

        self._violations_since_cluster = 0
        self._cluster_every_n          = 50
        self._cluster_lock             = threading.Lock()

        self._dropped      = 0
        self._dropped_lock = threading.Lock()

    def init_app(self, app) -> None:
        self._app                 = app
        n                         = app.config.get("TASK_WORKER_THREADS", self._n)
        maxsize                   = app.config.get("TASK_QUEUE_MAXSIZE",  200)
        self._identity_cooldown_s = app.config.get("IDENTITY_VIOLATION_COOLDOWN", 30)
        self._cluster_every_n     = app.config.get("CLUSTER_EVERY_N_VIOLATIONS",  50)
        self._q                   = queue.Queue(maxsize=maxsize)

        for i in range(n):
            t = threading.Thread(target=self._loop, name=f"task-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

        logger.info("TaskQueue: %d workers.", n)

    def put(self, job: Any) -> bool:
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1
            logger.warning("Queue full — dropped job #%d.", self._dropped)
            return False

    @property
    def dropped_count(self) -> int:
        with self._dropped_lock:
            return self._dropped

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                with self._app.app_context():
                    self._dispatch(job)
            except Exception as exc:
                logger.error("Worker error: %s", exc, exc_info=True)
            finally:
                self._q.task_done()

    def _dispatch(self, job: Any) -> None:
        if isinstance(job, ViolationJob):
            self._handle_violation(job)
        else:
            logger.warning("Unknown job: %s", type(job))

    def _handle_violation(self, job: ViolationJob) -> None:
        """
        Pipeline:
          1. embed_crop() — quality-gated InsightFace
          2. match_or_create() with quality score — outlier-safe
          3. Identity-level dedup (Layer 2)
          4. Save image + write Violation
          5. Update identity metadata (last_seen, thumbnail, confidence)
          6. Auto-cluster trigger
        """
        import cv2
        from flask import current_app
        from face.models import Violation, FaceIdentity
        from extensions import db

        pipeline      = current_app.extensions["face_pipeline"]
        violation_dir = current_app.config["VIOLATIONS_IMAGE_DIR"]
        os.makedirs(violation_dir, exist_ok=True)

        # ── 1. Embed ──────────────────────────────────────────────────────────
        faces       = pipeline.embed_crop(job.person_crop_bgr)
        identity_id : Optional[str]   = None
        label                         = "Unknown Person"
        score       : Optional[float] = None
        quality     : float           = 0.0

        if faces:
            best    = max(faces, key=lambda f: f["quality_score"])
            quality = best["quality_score"]
            # Pass quality into match_or_create so centroid weighting is correct
            identity_id, label, score = pipeline.match_or_create(
                best["embedding"], quality=quality, stream_id=job.stream_id
            )

        # ── 2. Identity-level dedup ───────────────────────────────────────────
        dedup_key = (identity_id or job.stream_id, job.violation_type)
        now       = time.monotonic()

        with self._identity_cd_lock:
            last = self._identity_cooldown.get(dedup_key, 0.0)
            if now - last < self._identity_cooldown_s:
                logger.debug("Deduped: %s / %s (%.1fs ago)", label, job.violation_type, now - last)
                return
            self._identity_cooldown[dedup_key] = now

        # ── 3. Save image ─────────────────────────────────────────────────────
        image_filename = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(os.path.join(violation_dir, image_filename), job.person_crop_bgr)

        # ── 4. Write violation ────────────────────────────────────────────────
        violation = Violation(
            stream_id=job.stream_id,
            violation_type=job.violation_type,
            confidence=job.confidence,
            identity_id=identity_id,
            match_score=score,
            image_filename=image_filename,
        )
        db.session.add(violation)

        # ── 5. Update identity metadata ───────────────────────────────────────
        if identity_id:
            identity_row = FaceIdentity.query.get(identity_id)
            if identity_row:
                identity_row.last_seen = _now()

                # Set thumbnail only once (first good violation image)
                if not identity_row.thumbnail_filename:
                    identity_row.thumbnail_filename = image_filename

                # Update stored identity_confidence from running cache value
                cache_entry = pipeline._cache.get(identity_id)
                if cache_entry:
                    identity_row.identity_confidence = cache_entry.get("confidence", 0.0)

        db.session.commit()

        logger.info(
            "Violation: %s → %s (score=%s quality=%.3f)",
            job.violation_type, label,
            f"{score:.4f}" if score else "none",
            quality,
        )

        # ── 6. Auto-cluster ───────────────────────────────────────────────────
        self._maybe_cluster(current_app._get_current_object())

    def _maybe_cluster(self, app) -> None:
        with self._cluster_lock:
            self._violations_since_cluster += 1
            if self._violations_since_cluster < self._cluster_every_n:
                return
            self._violations_since_cluster = 0

        logger.info("Auto-consolidation triggered.")
        try:
            from face.clustering import run_clustering
            result = run_clustering(
                eps=app.config["CLUSTER_EPS"],
                min_samples=app.config["CLUSTER_MIN_SAMPLES"],
            )
            if result["identities_merged"] > 0:
                with app.app_context():
                    app.extensions["face_pipeline"].reload_cache()
                logger.info("Post-cluster cache rebuilt: %s", result)
        except Exception as exc:
            logger.error("Auto-consolidation failed: %s", exc, exc_info=True)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._workers:
            t.join(timeout=timeout)
        logger.info("TaskQueue shut down.")