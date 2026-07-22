"""
TaskQueue — async violation pipeline
=====================================
ViolationJob now carries an optional pre-resolved identity_id.
When the worker stream has already run IoU tracking and identity assignment,
the task worker skips embed_crop() + match_or_create() entirely.
This eliminates the double-match race condition and is ~300ms faster per job.

The identity-level dedup (Layer 2) now uses identity_id when available,
falling back to the stream+type key for unknown faces.
"""
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
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
    # Pre-resolved by the stream tracker — when set, match_or_create() is skipped
    identity_id:     Optional[str] = None
    identity_label:  str           = "Unknown Person"


class TaskQueue:
    def __init__(self, num_workers: int = 2, maxsize: int = 200):
        self._q       = queue.Queue(maxsize=maxsize)
        self._workers = []
        self._stop    = threading.Event()
        self._app     = None
        self._n       = num_workers

        self._identity_cooldown:   dict  = {}
        self._identity_cd_lock           = threading.Lock()
        self._identity_cooldown_s: int   = 30

        self._violations_since_cluster   = 0
        self._cluster_every_n: int       = 50
        self._cluster_lock               = threading.Lock()

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
            # Non-daemon threads — the main process waits for them during shutdown.
            # This prevents silent data loss: pending ViolationJoys are drained
            # before the process exits (see shutdown()).
            t = threading.Thread(target=self._loop, name=f"task-worker-{i}", daemon=False)
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
        Fast path (identity pre-resolved by tracker):
          identity already set → skip embed_crop + match_or_create
          → dedup check → save image → write Violation

        Slow path (no tracker, e.g. future direct API calls):
          embed_crop → match_or_create → dedup → save image → write Violation
        """
        import cv2
        from flask import current_app
        from face.models import Violation, FaceIdentity
        from extensions import db

        pipeline      = current_app.extensions["face_pipeline"]
        violation_dir = current_app.config["VIOLATIONS_IMAGE_DIR"]
        os.makedirs(violation_dir, exist_ok=True)

        identity_id = job.identity_id
        label       = job.identity_label
        score: Optional[float] = None

        if identity_id is None:
            # Slow path — tracker not available or identity not yet confirmed
            faces = pipeline.embed_crop(job.person_crop_bgr)
            if faces:
                best    = max(faces, key=lambda f: f["quality_score"])
                identity_id, label, score = pipeline.match_or_create(
                    best["embedding"],
                    quality=best["quality_score"],
                    stream_id=job.stream_id,
                )
        # else: fast path — trust the tracker's resolution

        # ── Identity-level dedup ───────────────────────────────────────────────
        dedup_key = (identity_id or job.stream_id, job.violation_type)
        now       = time.monotonic()
        with self._identity_cd_lock:
            last = self._identity_cooldown.get(dedup_key, 0.0)
            if now - last < self._identity_cooldown_s:
                logger.debug("Deduped: %s / %s (%.1fs ago)", label, job.violation_type, now - last)
                return
            self._identity_cooldown[dedup_key] = now

        # ── Save image ────────────────────────────────────────────────────────
        image_filename = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(os.path.join(violation_dir, image_filename), job.person_crop_bgr)

        # ── Write violation ───────────────────────────────────────────────────
        violation = Violation(
            stream_id=job.stream_id,
            violation_type=job.violation_type,
            confidence=job.confidence,
            identity_id=identity_id,
            match_score=score,
            image_filename=image_filename,
        )
        db.session.add(violation)

        # ── Update identity metadata ──────────────────────────────────────────
        if identity_id:
            identity_row = FaceIdentity.query.get(identity_id)
            if identity_row:
                identity_row.last_seen = _now()
                if not identity_row.thumbnail_filename:
                    identity_row.thumbnail_filename = image_filename
                cache_entry = pipeline._cache.get(identity_id)
                if cache_entry:
                    identity_row.identity_confidence = cache_entry.get("confidence", 0.0)

        db.session.commit()
        logger.info("Violation: %s → %s", job.violation_type, label)

        # ── Auto-cluster ──────────────────────────────────────────────────────
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
        """Signal workers to stop, drain the queue, then join all threads."""
        self._stop.set()

        # Drain remaining jobs so workers unblock from queue.get()
        drained = 0
        while True:
            try:
                self._q.get_nowait()
                self._q.task_done()
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.warning(
                "TaskQueue shutdown: drained %d unprocessed job(s).", drained,
            )

        for t in self._workers:
            t.join(timeout=timeout)
        logger.info("TaskQueue shut down.")