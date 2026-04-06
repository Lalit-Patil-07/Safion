"""
Stream worker
=============
Runs in a dedicated thread per active video stream.

Responsibilities
----------------
- Read frames from VideoStream (threaded capture)
- Run YOLO inference via YOLOService (shared, lock-protected)
- Identify person↔violation associations
- Extract head crops (NOT full-body crops)
- Enqueue ViolationLogJobs — never block on logging or face recognition
- Annotate the frame with bboxes/labels
- JPEG-encode and store the frame for MJPEG delivery

Explicitly NOT responsible for
-------------------------------
- Face recognition (async in TaskQueue)
- Database writes (async in TaskQueue)
- Any I/O that would stall the frame loop
"""

import logging
import time
from collections import deque
from threading import Event, Thread
from typing import Optional

import cv2
import numpy as np

from detection.association import split_detections, check_association

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threaded frame capture
# ---------------------------------------------------------------------------
class VideoStream:
    """
    Reads frames in a background thread so the main loop is never blocked
    waiting for a slow capture device.
    Frame rate is capped by `fps_limit` via a sleep in the update loop.
    """

    def __init__(self, src, fps_limit: int = 30):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.fps_limit = fps_limit
        self._frame_interval = 1.0 / max(1, fps_limit)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self) -> "VideoStream":
        Thread(target=self._update, daemon=True, name="vs-capture").start()
        return self

    def _update(self) -> None:
        while not self.stopped:
            t0 = time.monotonic()
            grabbed, frame = self.stream.read()
            if grabbed:
                self.grabbed = grabbed
                self.frame = frame
            elapsed = time.monotonic() - t0
            sleep_for = self._frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def read(self) -> Optional[np.ndarray]:
        return self.frame if self.grabbed else None

    def stop(self) -> None:
        self.stopped = True
        time.sleep(0.1)
        if self.stream.isOpened():
            self.stream.release()


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------
def _annotate_frame(frame: np.ndarray, detections: list[dict]) -> None:
    """Draw bounding boxes and labels onto `frame` in-place."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        hex_color = det["color"].lstrip("#")
        bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), bgr, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------
def stream_worker(
    app,
    stream_id: str,
    source_type: str,
    source_path: str,
    stop_event: Event,
    stream_store: dict,
    store_lock,
) -> None:
    """
    Main stream processing loop.  Designed to run as a daemon Thread.

    Parameters
    ----------
    app          : Flask application instance (for app context in sub-tasks)
    stream_id    : UUID string identifying this stream
    source_type  : "webcam" | "rtsp" | "file"
    source_path  : device index (int str) or URL/path
    stop_event   : set() to signal the worker to stop
    stream_store : shared dict holding per-stream state
    store_lock   : threading.Lock protecting stream_store reads/writes
    """
    from flask import current_app

    vs: Optional[VideoStream] = None

    with app.app_context():
        yolo = app.extensions["yolo_service"]
        task_queue = app.extensions["task_queue"]
        jpeg_quality = app.config["STREAM_JPEG_QUALITY"]
        cooldown_s = app.config["VIOLATION_COOLDOWN_SECONDS"]
        fps_limit = app.config["FRAME_RATE_LIMIT"]
        face_pipeline = app.extensions["face_pipeline"]

    try:
        src = int(source_path) if source_type == "webcam" else source_path
        vs = VideoStream(src=src, fps_limit=fps_limit).start()

        time.sleep(0.5)  # give the capture thread a moment to fill self.frame

        if not vs.stream.isOpened():
            logger.error("Could not open video source: %s", src)
            return

        fps_deque: deque = deque(maxlen=30)
        # cooldown dict: (person_bbox_key, violation_type) → last_logged_timestamp
        recent_violations: dict = {}
        frame_count = 0

        logger.info("Stream %s started (source=%s).", stream_id, src)

        while not stop_event.is_set():
            frame = vs.read()
            if frame is None:
                time.sleep(0.05)
                continue

            t_start = time.monotonic()

            # ── YOLO inference ────────────────────────────────────────────────
            with app.app_context():
                detections = yolo.inference(frame)

            persons, violations = split_detections(detections)

            # ── Violation association + job dispatch ──────────────────────────
            if persons and violations:
                for person in persons:
                    associated = [
                        v for v in violations
                        if check_association(person["bbox"], v["bbox"])
                    ]
                    if not associated:
                        continue

                    # Extract head crop (BGR — top 35% of person bbox)
                    head_crop_bgr = face_pipeline.get_head_crop(frame, person["bbox"])

                    if head_crop_bgr is None:
                        continue

                    # Convert to RGB for the face task worker
                    head_crop_rgb = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2RGB)

                    for viol in associated:
                        # Cooldown key: approximate position bucket + violation type
                        # Bucket the person's centre to avoid per-pixel uniqueness
                        cx = int((person["bbox"][0] + person["bbox"][2]) / 2 / 50)
                        cy = int((person["bbox"][1] + person["bbox"][3]) / 2 / 50)
                        cooldown_key = (cx, cy, viol["class_name"])
                        now = time.monotonic()

                        if now - recent_violations.get(cooldown_key, 0) < cooldown_s:
                            continue

                        recent_violations[cooldown_key] = now

                        # Enqueue the violation log job (non-blocking)
                        from tasks.queue import ViolationLogJob
                        task_queue.put(ViolationLogJob(
                            stream_id=stream_id,
                            violation_type=viol["class_name"],
                            confidence=viol["confidence"],
                            head_crop_bgr=head_crop_bgr.copy(),
                            person_bbox=person["bbox"],
                        ))

            # ── Frame annotation + encode ─────────────────────────────────────
            _annotate_frame(frame, detections)

            ret, buffer = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if not ret:
                continue

            frame_count += 1
            elapsed = time.monotonic() - t_start
            fps_deque.append(1.0 / elapsed if elapsed > 0 else 0.0)

            violation_count = len([d for d in detections if not d["safe"]])

            with store_lock:
                if stream_id in stream_store:
                    stream_store[stream_id]["frame"] = buffer.tobytes()
                    stream_store[stream_id]["stats"].update({
                        "fps": round(sum(fps_deque) / len(fps_deque), 1),
                        "frame_count": frame_count,
                        "violation_count": stream_store[stream_id]["stats"].get(
                            "violation_count", 0
                        ) + (1 if violation_count > 0 else 0),
                        "last_detections": detections,
                        "resolution": [frame.shape[1], frame.shape[0]],
                    })

    except Exception as exc:
        import traceback
        logger.error(
            "Stream worker %s crashed: %s\n%s",
            stream_id, exc, traceback.format_exc(),
        )
    finally:
        if vs:
            vs.stop()
        logger.info("Stream worker %s stopped.", stream_id)
