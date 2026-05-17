"""
YOLOBatcher — cross-stream frame batching for GPU utilisation
=============================================================
StreamWorkers call submit(frame) and block on a threading.Event.
A single background thread drains the queue every YOLO_BATCH_TIMEOUT_MS ms
or when YOLO_BATCH_SIZE frames are waiting, whichever comes first, then
fires one batched YOLO inference and fans results back to each caller.

Thread model
------------
- N StreamWorker threads → each calls submit() → blocks on item.event
- 1 YOLOBatcher thread   → drains queue, calls inference_batch(), sets events
No locks beyond the Queue's own internal lock.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _BatchItem:
    frame:  np.ndarray
    event:  threading.Event = field(default_factory=threading.Event)
    result: list            = field(default_factory=list)


class YOLOBatcher:
    """
    Drop-in replacement call site for per-stream yolo.inference().

    Usage (StreamWorker):
        detections = self.batcher.submit(infer_frame)   # blocks; returns list[dict]
    """

    def __init__(self) -> None:
        self._queue:      queue.Queue             = queue.Queue()
        self._batch_size: int                     = 4
        self._timeout:    float                   = 0.02          # seconds
        self._yolo                                = None
        self._thread:     Optional[threading.Thread] = None
        self._stop:       threading.Event         = threading.Event()

    # ── Init ─────────────────────────────────────────────────────────────────

    def init_app(self, app) -> None:
        """Call from app factory after YOLOService is registered."""
        self._yolo       = app.extensions["yolo_service"]
        self._batch_size = app.config["YOLO_BATCH_SIZE"]
        self._timeout    = app.config["YOLO_BATCH_TIMEOUT_MS"] / 1000.0

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="yolo-batcher"
        )
        self._thread.start()
        logger.info(
            "YOLOBatcher started (batch_size=%d, timeout_ms=%d).",
            self._batch_size,
            app.config["YOLO_BATCH_TIMEOUT_MS"],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(self, frame: np.ndarray) -> list[dict]:
        """
        Enqueue frame for batched inference.  Blocks until result is ready.
        Returns list[dict] identical to YOLOService.inference().
        """
        item = _BatchItem(frame=frame)
        self._queue.put(item)
        item.event.wait()
        return item.result

    def shutdown(self) -> None:
        self._stop.set()

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            batch: list[_BatchItem] = []
            deadline = time.monotonic() + self._timeout

            # Collect up to batch_size frames within the timeout window
            while len(batch) < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            if not batch:
                continue

            try:
                per_frame = self._yolo.inference_batch([i.frame for i in batch])
            except Exception as exc:
                logger.error("YOLOBatcher inference error: %s", exc)
                per_frame = [[] for _ in batch]

            for item, detections in zip(batch, per_frame):
                item.result = detections
                item.event.set()