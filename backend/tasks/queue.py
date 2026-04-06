"""
TaskQueue
=========
A thread-pool-backed async task queue.

The video loop puts lightweight job objects onto the queue with a
non-blocking put (drops jobs when full rather than stalling the loop).
Worker threads consume jobs independently.

Two job types are registered:
    FaceRecognitionJob  — runs face matching and updates the DB
    ViolationLogJob     — writes the violation row and saves the image
"""

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FaceRecognitionJob:
    violation_id: str
    head_crop_rgb: np.ndarray   # RGB, already extracted from frame
    stream_id: str


@dataclass
class ViolationLogJob:
    stream_id: str
    violation_type: str
    confidence: float
    head_crop_bgr: np.ndarray   # BGR for saving with cv2
    person_bbox: list[float]


# ---------------------------------------------------------------------------
# Queue + worker pool
# ---------------------------------------------------------------------------
class TaskQueue:
    def __init__(self, num_workers: int = 2, maxsize: int = 200):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._num_workers = num_workers
        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._app = None             # Flask app for context

    def init_app(self, app) -> None:
        """Register with the Flask app and start worker threads."""
        self._app = app
        num = app.config.get("TASK_WORKER_THREADS", self._num_workers)
        maxsize = app.config.get("TASK_QUEUE_MAXSIZE", 200)
        self._queue = queue.Queue(maxsize=maxsize)

        for i in range(num):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"task-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

        logger.info("TaskQueue started with %d workers.", num)

    def put(self, job: Any, block: bool = False) -> bool:
        """
        Enqueue a job.  Non-blocking by default — drops the job and returns
        False if the queue is full (never stalls the video loop).
        """
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            logger.warning(
                "TaskQueue full — dropping %s job.", type(job).__name__
            )
            return False

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                with self._app.app_context():
                    self._dispatch(job)
            except Exception as exc:
                logger.error(
                    "Unhandled exception in task worker processing %s: %s",
                    type(job).__name__, exc, exc_info=True,
                )
            finally:
                self._queue.task_done()

    def _dispatch(self, job: Any) -> None:
        if isinstance(job, FaceRecognitionJob):
            self._handle_face_job(job)
        elif isinstance(job, ViolationLogJob):
            self._handle_violation_log_job(job)
        else:
            logger.warning("Unknown job type: %s", type(job))

    # ------------------------------------------------------------------
    # Face recognition handler
    # ------------------------------------------------------------------
    def _handle_face_job(self, job: FaceRecognitionJob) -> None:
        from flask import current_app
        from face.models import Violation
        from face.pipeline import MatchStatus
        from extensions import db

        pipeline = current_app.extensions["face_pipeline"]
        violation = Violation.query.get(job.violation_id)
        if not violation:
            logger.warning("FaceJob: violation %s not found.", job.violation_id)
            return

        # head_crop_rgb is RGB — match_from_crop expects BGR, convert back
        import cv2
        head_bgr = cv2.cvtColor(job.head_crop_rgb, cv2.COLOR_RGB2BGR)
        result = pipeline.match_from_crop(head_bgr)

        # Only commit a match for MATCHED status — never commit AMBIGUOUS as a
        # confirmed identity, surface it for human review instead.
        if result.status == MatchStatus.MATCHED:
            violation.identity_id   = result.identity_id
            violation.raw_name      = result.name
            violation.match_distance = result.distance
        elif result.status == MatchStatus.AMBIGUOUS:
            violation.raw_name      = f"[Ambiguous] {result.name}?"
            violation.match_distance = result.distance
        else:
            violation.raw_name      = "Unknown Person"
            violation.match_distance = result.distance if result.distance < 1.0 else None

        db.session.commit()

        logger.debug(
            "Face job complete — violation=%s status=%s name=%s dist=%.4f margin=%.4f",
            job.violation_id, result.status, result.name,
            result.distance, result.margin,
        )

    # ------------------------------------------------------------------
    # Violation log handler
    # ------------------------------------------------------------------
    def _handle_violation_log_job(self, job: ViolationLogJob) -> None:
        import os
        import uuid
        import cv2
        from flask import current_app
        from face.models import Violation
        from extensions import db

        violation_dir = current_app.config["VIOLATIONS_IMAGE_DIR"]
        os.makedirs(violation_dir, exist_ok=True)

        image_filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(violation_dir, image_filename)
        cv2.imwrite(image_path, job.head_crop_bgr)

        violation = Violation(
            stream_id=job.stream_id,
            violation_type=job.violation_type,
            confidence=job.confidence,
            raw_name="Unknown Person",
            image_filename=image_filename,
        )
        db.session.add(violation)
        db.session.commit()

        # Immediately enqueue face recognition for this violation
        face_pipeline = current_app.extensions["face_pipeline"]
        head_crop_rgb = None
        try:
            import cv2 as _cv2
            head_crop_rgb = _cv2.cvtColor(job.head_crop_bgr, _cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        if head_crop_rgb is not None:
            face_job = FaceRecognitionJob(
                violation_id=violation.id,
                head_crop_rgb=head_crop_rgb,
                stream_id=job.stream_id,
            )
            self.put(face_job)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=timeout)
        logger.info("TaskQueue shut down.")
