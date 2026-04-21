"""
Stream worker — YOLO + IoU tracking + violation dispatch
=========================================================
Architecture
------------
VideoStream (capture thread)
  └─ continuously reads frames; always stores only the latest (no queue)

StreamWorker (processing loop — this thread)
  └─ reads latest frame from VideoStream
  └─ YOLO inference
  └─ per-person IoU tracking
  └─ face embedding (SKIPPED on most frames — see FACE_EMBED_EVERY_N_FRAMES)
  └─ identity assignment
  └─ violation dispatch → TaskQueue (async, non-blocking)
  └─ JPEG encode → stream_store

Face embedding optimisation
---------------------------
embed_crop() is ~100ms per call on GPU, ~300ms on CPU.  Running it on every
frame serialises the entire pipeline and caps FPS at 3-10.

New behaviour:
  - Run embed_crop() when ANY violation is detected on a person crop, OR
  - Run embed_crop() every FACE_EMBED_EVERY_N_FRAMES frames (periodic refresh)
  - Skip it on all other frames

This decouples detection FPS (limited by YOLO, ~15-30 FPS) from face
embedding frequency (limited by InsightFace, ~3-10 FPS).

Multi-stream readiness
----------------------
All state is encapsulated in StreamWorker.  Multiple instances run in parallel
with no shared mutable state between streams.
"""

import logging
import time
from collections import deque
from threading import Event, Lock, Thread
from typing import Optional

import cv2
import numpy as np

from detection.association import split_detections, check_association
from streams.tracker import FaceTracker
from tasks.queue import ViolationJob

logger = logging.getLogger(__name__)


# ── Frame capture thread ──────────────────────────────────────────────────────

class VideoStream:
    """
    Dedicated capture thread.  Always keeps only the most recent frame.
    Processing never blocks capture and capture never blocks processing.
    A Lock protects the shared frame reference.
    """

    def __init__(self, src, fps_limit: int = 30):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._interval = 1.0 / max(1, fps_limit)
        self._lock  = Lock()
        self._frame: Optional[np.ndarray] = None
        self.stopped = False

        grabbed, frame = self.stream.read()
        if grabbed:
            self._frame = frame

    def start(self) -> "VideoStream":
        Thread(target=self._update, daemon=True, name="vs-capture").start()
        return self

    def _update(self) -> None:
        while not self.stopped:
            t0 = time.monotonic()
            grabbed, frame = self.stream.read()
            if grabbed:
                with self._lock:
                    self._frame = frame
            sleep = self._interval - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self.stopped = True
        time.sleep(0.1)
        if self.stream.isOpened():
            self.stream.release()


# ── Annotation helper ─────────────────────────────────────────────────────────

def _annotate(frame: np.ndarray, detections: list[dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        h = det["color"].lstrip("#")
        bgr = tuple(int(h[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), bgr, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ── Per-stream worker ─────────────────────────────────────────────────────────

class StreamWorker:
    """
    Encapsulates all mutable state for one stream.
    No global state — multiple instances run safely in parallel.

    Processing loop:
      1. Read latest frame (non-blocking; skip if no new frame)
      2. YOLO inference
      3. Per-person IoU tracking
      4. Face embedding — only when:
           a) a violation is associated with this person, OR
           b) every FACE_EMBED_EVERY_N_FRAMES frames (periodic identity refresh)
      5. Identity assignment (deferred until track is confirmed)
      6. Violation dispatch -> async TaskQueue (non-blocking)
      7. JPEG encode -> stream_store (lock-protected write)
    """

    def __init__(
        self,
        app,
        stream_id:   str,
        source_type: str,
        source_path: str,
        stop_event:  Event,
        stream_store: dict,
        store_lock:  Lock,
    ):
        self.app          = app
        self.stream_id    = stream_id
        self.source_type  = source_type
        self.source_path  = source_path
        self.stop_event   = stop_event
        self.stream_store = stream_store
        self.store_lock   = store_lock

        cfg = app.config
        self.yolo         = app.extensions["yolo_service"]
        self.task_queue   = app.extensions["task_queue"]
        self.pipeline     = app.extensions["face_pipeline"]
        self.jpeg_quality = cfg["STREAM_JPEG_QUALITY"]
        self.violation_cd = cfg["VIOLATION_COOLDOWN_SECONDS"]
        self.fps_limit    = cfg["FRAME_RATE_LIMIT"]
        self.face_every_n = cfg["FACE_EMBED_EVERY_N_FRAMES"]

        self.tracker = FaceTracker(
            iou_threshold  = cfg["TRACK_IOU_THRESHOLD"],
            max_lost       = cfg["TRACK_MAX_LOST"],
            min_frames     = cfg["TRACK_MIN_FRAMES"],
            min_embeddings = cfg["TRACK_MIN_EMBEDDINGS"],
        )

        # PROCESS_WIDTH: resize frame to this width before YOLO inference.
        # Smaller = faster GPU inference, same detection quality at 640.
        # 0 = disabled (use full resolution).
        self.process_width = cfg["PROCESS_WIDTH"]

        # STREAM_OUTPUT_FPS: how many annotated JPEG frames to publish per
        # second.  Lower values reduce CPU encode time across many streams.
        # Encode is skipped on intermediate frames; stats still update every frame.
        output_fps = cfg["STREAM_OUTPUT_FPS"]
        # FIX: guard against STREAM_OUTPUT_FPS=0 or values exceeding FRAME_RATE_LIMIT,
        #      both of which would cause ZeroDivisionError or encode on every frame.
        _fps_limit = cfg["FRAME_RATE_LIMIT"]
        if output_fps <= 0 or output_fps > _fps_limit:
            output_fps = _fps_limit   # encode every frame as safe fallback
        self.encode_every_n = max(1, round(_fps_limit / output_fps))

        self._fps_deque:  deque = deque(maxlen=30)
        self._frame_count: int  = 0

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        vs: Optional[VideoStream] = None
        try:
            src = int(self.source_path) if self.source_type == "webcam" else self.source_path
            vs  = VideoStream(src=src, fps_limit=self.fps_limit).start()
            time.sleep(0.5)

            if not vs.stream.isOpened():
                logger.error("Cannot open source: %s", src)
                return

            logger.info("Stream %s started (%s).", self.stream_id, src)

            while not self.stop_event.is_set():
                frame = vs.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                t0 = time.monotonic()
                self._process_frame(frame)
                elapsed = time.monotonic() - t0
                self._fps_deque.append(1.0 / elapsed if elapsed > 0 else 0.0)

        except Exception as exc:
            import traceback
            logger.error("Stream %s crashed: %s\n%s",
                         self.stream_id, exc, traceback.format_exc())
        finally:
            if vs:
                vs.stop()
            logger.info("Stream %s stopped.", self.stream_id)

    # ── Per-frame processing ──────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> None:
        fc   = self._frame_count
        sid8 = self.stream_id[:8]

        # ── YOLO (optionally on a downscaled frame) ───────────────────────────
        orig_h, orig_w = frame.shape[:2]
        if self.process_width > 0 and orig_w > self.process_width:
            scale = self.process_width / orig_w
            infer_frame = cv2.resize(
                frame, (self.process_width, int(orig_h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_x, scale_y = orig_w / infer_frame.shape[1], orig_h / infer_frame.shape[0]
        else:
            infer_frame = frame
            scale_x = scale_y = 1.0

        with self.app.app_context():
            detections = self.yolo.inference(infer_frame)

        # Scale bboxes from inference resolution back to original resolution.
        # Crops for InsightFace and annotation always use the original frame.
        # FIX: cast to int immediately so downstream code (map(int, bbox),
        #      check_association) never receives float coordinates.
        if scale_x != 1.0 or scale_y != 1.0:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [
                    int(x1 * scale_x), int(y1 * scale_y),
                    int(x2 * scale_x), int(y2 * scale_y),
                ]

        persons, violations = split_detections(detections)

        if fc % 30 == 0:
            logger.debug("[stream=%s frame=%d] YOLO: %d persons, %d violations",
                         sid8, fc, len(persons), len(violations))

        # ── Per-person tracking + face + violation dispatch ───────────────────
        active_bboxes: list[list[float]] = []

        for person in persons:
            bbox = person["bbox"]
            active_bboxes.append(bbox)
            track = self.tracker.update(bbox)

            px1, py1, px2, py2 = map(int, bbox)
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(frame.shape[1], px2), min(frame.shape[0], py2)
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            # Violation association — cheap bbox check, done before face
            associated = [v for v in violations if check_association(bbox, v["bbox"])]
            has_violation = bool(associated)

            # Face embedding decision:
            #   a) run immediately when violation present (need identity for logging), OR
            #   b) periodic refresh to accumulate embeddings for future matching
            #   c) skip all other frames — this is the key FPS improvement
            run_face = has_violation or (fc % self.face_every_n == 0)

            # FIX: initialise faces before the conditional so the debug log
            #      below is safe even when run_face is False.
            faces: list = []
            if run_face:
                with self.app.app_context():
                    faces = self.pipeline.embed_crop(crop)
                if faces:
                    best = max(faces, key=lambda f: f["quality_score"])
                    track.add_embedding(best["embedding"], best["quality_score"])
            if run_face and fc % 30 == 0:
                logger.debug("[stream=%s] track=%d frames=%d embs=%d faces=%d",
                             sid8, track.track_id,
                             track.frames_seen, track.n_embeddings, len(faces))

            # Identity assignment — deferred until track is confirmed
            if self.tracker.is_confirmed(track) and track.identity_id is None:
                logger.debug("[stream=%s] track=%d confirmed — matching identity",
                             sid8, track.track_id)
                if track.pending_embeddings:
                    with self.app.app_context():
                        identity_id, label, score = self.pipeline.match_and_store_track(
                            track.pending_embeddings,
                            track.pending_quality,
                            stream_id=self.stream_id,
                        )
                    track.identity_id    = identity_id
                    track.identity_label = label
                    logger.debug("[stream=%s] track=%d -> '%s' (score=%.3f)",
                                 sid8, track.track_id, label, score)

            # Violation dispatch
            if track.frames_seen < self.tracker.min_frames:
                continue
            if not associated:
                continue

            now = time.monotonic()
            if now - track.last_violation_time < self.violation_cd:
                continue
            track.last_violation_time = now

            for viol in associated:
                logger.debug("[stream=%s] track=%d -> violation '%s' identity=%s",
                             sid8, track.track_id, viol["class_name"], track.identity_id)
                self.task_queue.put(ViolationJob(
                    stream_id       = self.stream_id,
                    violation_type  = viol["class_name"],
                    confidence      = viol["confidence"],
                    person_crop_bgr = crop.copy(),
                    person_bbox     = bbox,
                    identity_id     = track.identity_id,
                    identity_label  = track.identity_label or "Unknown Person",
                ))

        self.tracker.mark_missing(active_bboxes)

        self._frame_count += 1

        # ── Encode + publish (throttled to STREAM_OUTPUT_FPS) ─────────────────
        # Annotation and JPEG encode are CPU-heavy. Skipping them on intermediate
        # frames keeps detection running at full rate while reducing CPU load.
        # Stats (fps, violation_count) update every frame regardless.
        encode_this_frame = (self._frame_count % self.encode_every_n == 0)

        with self.store_lock:
            if self.stream_id in self.stream_store:
                stats = self.stream_store[self.stream_id]["stats"]
                stats["fps"]         = round(
                    sum(self._fps_deque) / max(len(self._fps_deque), 1), 1
                )
                stats["frame_count"] = self._frame_count
                stats["violation_count"] = stats.get("violation_count", 0) + (
                    1 if any(not d["safe"] for d in detections) else 0
                )
                stats["last_detections"] = detections
                stats["resolution"]      = [orig_w, orig_h]

                if encode_this_frame:
                    _annotate(frame, detections)
                    ret, buf = cv2.imencode(
                        ".jpg", frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if ret:
                        self.stream_store[self.stream_id]["frame"] = buf.tobytes()


# ── Entry point (StreamManager interface — unchanged) ─────────────────────────

def stream_worker(app, stream_id, source_type, source_path, stop_event, stream_store, store_lock):
    """
    Called by StreamManager.  Signature is identical to the previous version.
    Constructs a StreamWorker inside the app context and runs it.
    """
    with app.app_context():
        worker = StreamWorker(
            app, stream_id, source_type, source_path,
            stop_event, stream_store, store_lock,
        )
    worker.run()