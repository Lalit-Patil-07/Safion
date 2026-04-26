"""
Stream worker — YOLO + IoU tracking + violation dispatch
=========================================================
Architecture
------------
VideoStream (capture thread)
  └─ pushes frames into process_queue (bounded, drop on full)

StreamWorker._processing_loop (main worker thread)
  └─ adaptive frame skip gate (queue-pressure-based)
  └─ YOLO batched inference (via YOLOBatcher — non-blocking submit)
  └─ per-person IoU tracking
  └─ face job enqueue → face_queue (non-blocking put_nowait)
  └─ violation dispatch → TaskQueue
  └─ pushes _ProcessedFrame → output_queue (non-blocking put_nowait)

StreamWorker._face_worker_loop (daemon thread per stream)
  └─ reads _FaceJob from face_queue
  └─ runs embed_crop() (blocking InsightFace — isolated here)
  └─ quality gate + embedding update (under track.lock)
  └─ identity assignment when track confirmed (DB I/O outside lock)

StreamWorker._encoding_loop (daemon thread per stream)
  └─ reads _ProcessedFrame from output_queue
  └─ annotation + JPEG encode (throttled to STREAM_OUTPUT_FPS)
  └─ stats update under store_lock

Adaptive frame skipping
-----------------------
_processing_loop evaluates queue fill ratios before each frame.
  pressure ≥ LOAD_HIGH_THRESHOLD → increase skip rate (up to MAX_FRAME_SKIP)
  pressure ≤ LOAD_LOW_THRESHOLD  → decrease skip rate (down to 0)
  in between                     → hold current rate
Skipped frames are dropped before YOLO — no tracking, face, or encode cost.

Thread safety
-------------
Fields written by face worker and read by processing loop are protected by
track.lock (added to Track in tracker.py).  Fields written only by the
processing loop (frames_seen, frames_lost, bbox, last_violation_time) need
no lock — single writer.
"""

import logging
import queue
import time
from collections import deque
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Optional

import cv2
import numpy as np

from detection.association import split_detections, check_association
from streams.tracker import FaceTracker
from tasks.queue import ViolationJob

import weakref

logger = logging.getLogger(__name__)

_registry_lock: Lock = Lock()
_worker_registry = weakref.WeakSet()


# ── Frame capture thread ──────────────────────────────────────────────────────

def _gstreamer_pipeline(src, fps_limit: int) -> str:
    """
    Build a GStreamer pipeline string for cv2.VideoCapture.

    RTSP sources use rtspsrc → decodebin → videoconvert → appsink.
    File / device sources fall back to uridecodebin / v4l2src paths.
    appsink drop=true keeps only the latest frame and never blocks the
    capture thread — equivalent to the latest-frame behaviour without a ring buffer.
    """
    caps = f"video/x-raw,framerate={fps_limit}/1"
    sink = f"videoconvert ! appsink max-buffers=2 drop=true sync=false caps={caps}"

    if isinstance(src, str) and src.startswith("rtsp://"):
        return (
            f"rtspsrc location={src} latency=100 ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! {sink}"
        )
    if isinstance(src, int):
        return f"v4l2src device=/dev/video{src} ! {sink}"
    # File or other URI — let GStreamer resolve it
    return f"uridecodebin uri={src} ! {sink}"


class VideoStream:
    """
    Dedicated capture thread.  Pushes frames into process_queue.
    Frames are dropped (put_nowait) when the queue is full so that the
    capture thread never blocks the processing thread, and the processing
    thread never blocks the capture thread.

    If backend='GSTREAMER' a GStreamer appsink pipeline is used; appsink
    drop=true replicates the latest-frame behaviour without a ring buffer.
    On any GStreamer open failure the code retries with the default OpenCV
    backend so a misconfigured GStreamer install never takes down a stream.
    """

    def __init__(self, src, fps_limit: int = 30, process_queue: queue.Queue = None,
                 backend: str = "OPENCV"):
        self._interval = 1.0 / max(1, fps_limit)
        self._queue    = process_queue
        self.stopped   = False
        self.last_frame_time:      float = time.monotonic()
        self.consecutive_failures: int   = 0

        self.stream = self._open(src, fps_limit, backend)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        grabbed, frame = self.stream.read()
        if grabbed and self._queue is not None:
            self.last_frame_time      = time.monotonic()
            self.consecutive_failures = 0
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def start(self) -> "VideoStream":
        Thread(target=self._update, daemon=True, name="vs-capture").start()
        return self

    def _update(self) -> None:
        while not self.stopped:
            t0 = time.monotonic()
            try:
                grabbed, frame = self.stream.read()
            except Exception as exc:
                logger.warning("VideoStream read() exception: %s", exc)
                grabbed = False
                frame   = None

            if grabbed and self._queue is not None:
                self.last_frame_time      = time.monotonic()
                self.consecutive_failures = 0
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass   # processing is behind — drop frame, never block
            else:
                self.consecutive_failures += 1
            sleep = self._interval - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def stop(self) -> None:
        self.stopped = True
        time.sleep(0.1)
        if self.stream.isOpened():
            self.stream.release()

    @staticmethod
    def _open(src, fps_limit: int, backend: str) -> cv2.VideoCapture:
        """
        Open VideoCapture with the requested backend.
        Falls back to the default OpenCV backend on any GStreamer failure so
        a misconfigured GStreamer install never takes down a stream.
        """
        if backend == "GSTREAMER":
            pipeline = _gstreamer_pipeline(src, fps_limit)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                logger.info("VideoStream: GStreamer backend opened (%s…)", pipeline[:60])
                return cap
            logger.warning(
                "VideoStream: GStreamer failed to open '%s' — falling back to OpenCV backend.",
                src,
            )
            cap.release()
        cap = cv2.VideoCapture(src)
        logger.info("VideoStream: OpenCV backend opened for '%s'.", src)
        return cap


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


# ── Pipeline data classes ─────────────────────────────────────────────────────

@dataclass
class _ProcessedFrame:
    """Payload passed from the processing loop to the encoding loop."""
    frame:           np.ndarray
    detections:      list
    orig_w:          int
    orig_h:          int
    frame_count:     int
    fps_sample:      float   # filled by _processing_loop after timing
    violation_delta: int     # 1 if any unsafe detection this frame, else 0


@dataclass
class _FaceJob:
    """
    Crop + context enqueued by the processing loop for the face worker.
    Holds a live reference to the Track so the worker can write results back
    directly under track.lock — no shared data structure needed.
    """
    track:         object      # Track; typed as object to avoid circular import
    crop:          np.ndarray  # copy — processing loop must not reuse this buffer
    has_violation: bool
    frame_count:   int


# ── Per-stream worker ─────────────────────────────────────────────────────────

class StreamWorker:
    """
    Encapsulates all mutable state for one stream.
    No global state — multiple instances run safely in parallel.

    Three concurrent threads per stream:
      _processing_loop  — YOLO + tracking + face job dispatch  (this thread)
      _face_worker_loop — InsightFace embed_crop + identity    (daemon)
      _encoding_loop    — annotation + JPEG encode + stats     (daemon)
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
        self.batcher      = app.extensions["yolo_batcher"]
        self.task_queue   = app.extensions["task_queue"]
        self.pipeline     = app.extensions["face_pipeline"]
        self.jpeg_quality = cfg["STREAM_JPEG_QUALITY"]
        self.violation_cd = cfg["VIOLATION_COOLDOWN_SECONDS"]
        self.fps_limit    = cfg["FRAME_RATE_LIMIT"]
        self.face_every_n           = cfg["FACE_EMBED_EVERY_N_FRAMES"]
        self.identity_recheck_secs  = cfg["IDENTITY_RECHECK_SECONDS"]
        self.max_pending_embeddings = cfg["MAX_PENDING_EMBEDDINGS"]

        # ── Pipeline queues ───────────────────────────────────────────────────
        # process_queue : VideoStream      → _processing_loop   (raw frames)
        # output_queue  : _processing_loop → _encoding_loop     (annotated results)
        # face_queue    : _processing_loop → _face_worker_loop  (crops for embedding)
        # All bounded; put_nowait/drop-oldest keeps each stage fully non-blocking.
        self.process_queue = queue.Queue(maxsize=cfg["PROCESS_QUEUE_SIZE"])
        self.output_queue  = queue.Queue(maxsize=cfg["OUTPUT_QUEUE_SIZE"])
        self.face_queue    = queue.Queue(maxsize=cfg["FACE_QUEUE_SIZE"])

        self.tracker = FaceTracker(
            iou_threshold  = cfg["TRACK_IOU_THRESHOLD"],
            max_lost       = cfg["TRACK_MAX_LOST"],
            min_frames     = cfg["TRACK_MIN_FRAMES"],
            min_embeddings = cfg["TRACK_MIN_EMBEDDINGS"],
            max_embeddings = cfg["MAX_PENDING_EMBEDDINGS"],
            stale_timeout  = cfg["TRACK_STALE_TIMEOUT_S"],
        )

        # PROCESS_WIDTH: resize frame to this width before YOLO inference.
        self.process_width = cfg["PROCESS_WIDTH"]

        # STREAM_OUTPUT_FPS: how many annotated JPEG frames to publish per second.
        output_fps = cfg["STREAM_OUTPUT_FPS"]
        _fps_limit = cfg["FRAME_RATE_LIMIT"]
        if output_fps <= 0 or output_fps > _fps_limit:
            output_fps = _fps_limit
        self.encode_every_n = max(1, round(_fps_limit / output_fps))

        self._fps_deque:   deque = deque(maxlen=30)
        self._frame_count: int   = 0

        # ── Adaptive frame skip ───────────────────────────────────────────────
        # _current_skip: frames to drop between each processed frame (0 = off).
        # _skip_counter: consecutive frames skipped in the current run.
        # Both adapted each frame by _adapt_skip() via queue fill ratios.
        self._max_skip:     int   = cfg["MAX_FRAME_SKIP"]
        self._load_high:    float = cfg["LOAD_HIGH_THRESHOLD"]
        self._load_low:     float = cfg["LOAD_LOW_THRESHOLD"]
        self._current_skip: int   = 0
        self._skip_counter: int   = 0
        self._last_had_violation: bool = False  # proxy for HIGH priority detection

        self._quality_improve_margin:  float = cfg["EMBED_QUALITY_IMPROVE_MARGIN"]
        self._identity_min_embs_delta: int   = cfg["IDENTITY_MIN_EMBEDDINGS_DELTA"]
        self._strong_match_threshold:  float = cfg["STRONG_MATCH_THRESHOLD"]

        # ── Violation write buffer ─────────────────────────────────────────────
        # Jobs accumulate here and are flushed to TaskQueue in bulk when either
        # _violation_batch_size is reached OR _violation_batch_timeout elapses.
        # Coalescing drops duplicate (identity_id, violation_type) pairs that
        # arrive within _violation_coalesce_window — complementing the existing
        # per-track violation cooldown which guards the detection side.
        self._violation_buffer:          list  = []
        self._violation_last_flush:      float = time.monotonic()
        self._violation_batch_size:      int   = cfg["VIOLATION_BATCH_SIZE"]
        self._violation_batch_timeout:   float = cfg["VIOLATION_BATCH_TIMEOUT_MS"] / 1000.0
        self._violation_coalesce_window: float = cfg["VIOLATION_COALESCE_WINDOW_S"]
        # (identity_id, violation_type) → last buffered monotonic timestamp
        self._violation_coalesce_seen:   dict  = {}

        # ── Stream restart and stall detection ───────────────────────────────────
        self._stall_timeout:   float = cfg["STREAM_STALL_TIMEOUT_S"]
        self._restart_delay:   float = cfg["STREAM_RESTART_DELAY_S"]

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        import traceback
        src = int(self.source_path) if self.source_type == "webcam" else self.source_path
        vs: Optional[VideoStream] = None

        try:
            Thread(target=self._encoding_loop, daemon=True,
                   name=f"enc-{self.stream_id[:8]}").start()
            Thread(target=self._face_worker_loop, daemon=True,
                   name=f"face-{self.stream_id[:8]}").start()

            # ✅ REGISTER ONCE (CORRECT PLACEMENT)
            with _registry_lock:
                _worker_registry.add(self)

            while not self.stop_event.is_set():

                if vs is not None:
                    vs.stop()

                try:
                    vs = VideoStream(
                        src=src,
                        fps_limit=self.fps_limit,
                        process_queue=self.process_queue,
                        backend=self.app.config["VIDEO_CAPTURE_BACKEND"],
                    ).start()
                    time.sleep(0.5)
                except Exception as exc:
                    logger.error("Stream %s open failed: %s — retry in %.1fs",
                                 self.stream_id, exc, self._restart_delay)
                    time.sleep(self._restart_delay)
                    continue

                if not vs.stream.isOpened():
                    logger.error("Stream %s cannot open source — retry in %.1fs",
                                 self.stream_id, self._restart_delay)
                    time.sleep(self._restart_delay)
                    continue

                logger.info("Stream %s started.", self.stream_id)

                self._run_with_stall_watch(vs)

                if not self.stop_event.is_set():
                    logger.warning("Stream %s restarting — delay %.1fs",
                                   self.stream_id, self._restart_delay)
                    time.sleep(self._restart_delay)

        except Exception as exc:
            logger.error("Stream %s fatal error: %s\n%s",
                         self.stream_id, exc, traceback.format_exc())
        finally:
            self._flush_violations()
            with _registry_lock:
                _worker_registry.discard(self)
            if vs:
                vs.stop()
            logger.info("Stream %s stopped.", self.stream_id)

    # ── Per-frame processing ──────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> _ProcessedFrame:
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

        detections = self.batcher.submit(infer_frame)

        # Scale bboxes from inference resolution back to original resolution.
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

        # ── Per-person tracking + face job enqueue + violation dispatch ────────
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
            associated    = [v for v in violations if check_association(bbox, v["bbox"])]
            has_violation = bool(associated)

            # ── Face job enqueue — non-blocking ───────────────────────────────
            # Processing loop never calls embed_crop() directly.
            # Cooldown / periodic checks preserve Stage 3 semantics; the only
            # change is work is handed off and we continue immediately rather
            # than blocking on InsightFace.
            now_mono = time.monotonic()
            with track.lock:
                identity_assigned = track.identity_id is not None
                cooldown_expired  = (
                    (now_mono - track.last_identity_check) > self.identity_recheck_secs
                )
            periodic_frame = (fc % self.face_every_n == 0)

            if identity_assigned and not cooldown_expired and not has_violation:
                logger.debug(
                    "[stream=%s] track=%d identity cooldown active (%.1fs remaining)",
                    sid8, track.track_id,
                    self.identity_recheck_secs - (now_mono - track.last_identity_check),
                )
                run_face = False
            else:
                run_face = (
                    has_violation
                    or (not identity_assigned)
                    or (cooldown_expired and periodic_frame)
                )

            # ── Significance guard ────────────────────────────────────────────
            # Skip enqueue when the track already has a high-confidence identity
            # and there is no violation — embed_crop would produce an embedding
            # that is unlikely to improve on what we already have.
            # Violations always bypass this guard (run_face=True + no suppression).
            if run_face and not has_violation:
                with track.lock:
                    _conf      = track.last_match_confidence
                    _n_at_last = track.embeddings_at_last_match
                    _n_now     = track.n_embeddings
                    _id        = track.identity_id

                if (
                    _id is not None
                    and _conf >= self._strong_match_threshold
                    and (_n_now - _n_at_last) < self._identity_min_embs_delta
                ):
                    run_face = False
                    logger.debug(
                        "[stream=%s] track=%d face enqueue suppressed — "
                        "stable identity conf=%.3f emb_delta=%d",
                        sid8, track.track_id, _conf, _n_now - _n_at_last,
                    )

            if run_face and crop.size > 0:
                face_job = _FaceJob(
                    track         = track,
                    crop          = crop.copy(),   # isolated copy — worker owns this buffer
                    has_violation = has_violation,
                    frame_count   = fc,
                )
                try:
                    self.face_queue.put_nowait(face_job)
                except queue.Full:
                    # Drop oldest job; violation frames are more urgent than stale crops.
                    try:
                        self.face_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.face_queue.put_nowait(face_job)
                    except queue.Full:
                        pass   # concurrent fill between the two put_nowait — skip

            if fc % 30 == 0:
                logger.debug("[stream=%s] track=%d frames=%d",
                             sid8, track.track_id, track.frames_seen)

            # ── Violation dispatch ─────────────────────────────────────────────
            if track.frames_seen < self.tracker.min_frames:
                continue
            if not associated:
                continue

            now = time.monotonic()
            if now - track.last_violation_time < self.violation_cd:
                continue
            track.last_violation_time = now

            # Read identity fields under lock — face worker may be writing them
            # concurrently on a different thread.
            with track.lock:
                identity_id    = track.identity_id
                identity_label = track.identity_label or "Unknown Person"

            for viol in associated:
                logger.debug("[stream=%s] track=%d -> violation '%s' identity=%s",
                             sid8, track.track_id, viol["class_name"], identity_id)
                self._buffer_violation(
                    ViolationJob(
                        stream_id       = self.stream_id,
                        violation_type  = viol["class_name"],
                        confidence      = viol["confidence"],
                        person_crop_bgr = crop.copy(),
                        person_bbox     = bbox,
                        identity_id     = identity_id,
                        identity_label  = identity_label,
                    )
                )

        self.tracker.mark_missing(active_bboxes)

        self._frame_count += 1

        return _ProcessedFrame(
            frame           = frame,
            detections      = detections,
            orig_w          = orig_w,
            orig_h          = orig_h,
            frame_count     = self._frame_count,
            fps_sample      = 0.0,   # filled by _processing_loop after timing
            violation_delta = 1 if any(not d["safe"] for d in detections) else 0,
        )

    # ── Violation write buffer ────────────────────────────────────────────────

    def _buffer_violation(self, job: ViolationJob) -> None:
        """
        Coalesce and buffer a ViolationJob.

        Coalescing key: (identity_id, violation_type).
        Jobs whose key was buffered within _violation_coalesce_window are
        dropped — the first occurrence in the window is sufficient for the DB.
        Unknown identity (None) uses track bbox as a tiebreaker key so that
        distinct unidentified persons are not collapsed together.

        After buffering, flushes immediately if batch size is reached.
        """
        now = time.monotonic()
        coalesce_key = (
            job.identity_id or str(job.person_bbox),
            job.violation_type,
        )
        last_seen = self._violation_coalesce_seen.get(coalesce_key, 0.0)
        if now - last_seen < self._violation_coalesce_window:
            logger.debug(
                "[stream=%s] violation coalesced — key=%s delta=%.2fs",
                self.stream_id[:8], coalesce_key, now - last_seen,
            )
            return

        self._violation_coalesce_seen[coalesce_key] = now
        self._violation_buffer.append(job)

        if len(self._violation_buffer) >= self._violation_batch_size:
            self._flush_violations()

    def _flush_violations(self) -> None:
        """
        Push all buffered ViolationJobs into TaskQueue and reset the buffer.
        Called when batch size is reached OR timeout elapses in the processing
        loop.  TaskQueue.put() is non-blocking from the caller's perspective
        (the queue worker handles DB writes asynchronously).
        Stale coalesce keys older than 2× the coalesce window are evicted here
        to prevent unbounded dict growth across long-running streams.
        """
        if not self._violation_buffer:
            self._violation_last_flush = time.monotonic()
            return

        for job in self._violation_buffer:
            self.task_queue.put(job)

        logger.debug(
            "[stream=%s] violation batch flushed — %d jobs",
            self.stream_id[:8], len(self._violation_buffer),
        )
        self._violation_buffer.clear()
        now = self._violation_last_flush = time.monotonic()

        # Evict stale coalesce keys — O(n keys), runs at most once per flush
        cutoff = now - self._violation_coalesce_window * 2
        stale  = [k for k, t in self._violation_coalesce_seen.items() if t < cutoff]
        for k in stale:
            del self._violation_coalesce_seen[k]

    # ── Adaptive skip helpers ─────────────────────────────────────────────────

    def _compute_pressure(self) -> float:
        """
        Returns the maximum fill ratio (0.0–1.0) across the three bounded
        pipeline queues.  Using MAX rather than average means a single
        saturated stage triggers back-pressure immediately — the slowest
        stage sets the pace.
        """
        queues = [self.process_queue, self.output_queue, self.face_queue]
        ratios = [q.qsize() / q.maxsize for q in queues if q.maxsize > 0]
        return max(ratios) if ratios else 0.0

    def _compute_global_pressure(self) -> float:
        with _registry_lock:
            workers = list(_worker_registry)
        workers = [w for w in workers if w is not self]
        if not workers:
            return 0.0
        return sum(w._compute_pressure() for w in workers) / len(workers)

    def _adapt_skip(self) -> None:
        """
        Raise or lower _current_skip by one step per frame based on pressure.

        Hysteresis via separate HIGH/LOW thresholds prevents rapid oscillation:
          pressure ≥ LOAD_HIGH → increase skip (up to MAX_FRAME_SKIP)
          pressure ≤ LOAD_LOW  → decrease skip (down to 0)
          in between           → hold current level
        """
        pressure = max(self._compute_pressure(), self._compute_global_pressure())
        if pressure >= self._load_high:
            self._current_skip = min(self._current_skip + 1, self._max_skip)
        elif pressure <= self._load_low:
            self._current_skip = max(self._current_skip - 1, 0)

    # ── Priority detection ────────────────────────────────────────────────────

    def _frame_priority(self) -> int:
        if self._last_had_violation:
            return 2
        if any(t.identity_id is None for t in self.tracker.active_tracks):
            return 1
        return 0

    def _run_with_stall_watch(self, vs: "VideoStream") -> None:
        import traceback
        sid8 = self.stream_id[:8]

        while not self.stop_event.is_set():

            stall_age = time.monotonic() - vs.last_frame_time
            if stall_age > self._stall_timeout:
                logger.error("[stream=%s] stall detected (%.1fs)",
                             sid8, stall_age)
                return

            try:
                frame = self.process_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._adapt_skip()
            priority = self._frame_priority()

            if priority == 2:
                effective_skip = 0
            elif priority == 1:
                effective_skip = self._current_skip // 2
            else:
                effective_skip = self._current_skip

            if self._skip_counter < effective_skip:
                self._skip_counter += 1
                continue
            self._skip_counter = 0

            try:
                pf = self._process_frame(frame)
            except Exception as exc:
                logger.error("[stream=%s] process error: %s\n%s",
                             sid8, exc, traceback.format_exc())
                continue

            try:
                self.output_queue.put_nowait(pf)
            except queue.Full:
                pass

    # ── Processing loop ───────────────────────────────────────────────────────

    def _processing_loop(self) -> None:
        """
        Reads raw frames from process_queue, runs YOLO + tracking + face job
        dispatch via _process_frame(), then pushes _ProcessedFrame into
        output_queue.

        Frame skipping is applied BEFORE any heavy work.  The skip counter is
        evaluated after each queue-pressure check so skipping adapts every
        frame.  Safety invariant: _current_skip is capped at MAX_FRAME_SKIP,
        so at worst 1 in (MAX_FRAME_SKIP + 1) frames is processed — the
        pipeline can never be completely starved.

        Dropping policy: if output_queue is full (encoding is behind) the
        processed frame is discarded with put_nowait so this loop never blocks
        on the encoding stage.
        """
        while not self.stop_event.is_set():
            try:
                frame = self.process_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # ── Adaptive skip gate (before ANY heavy work) ────────────────────
            # Re-evaluate pressure every frame; one step up/down per evaluation.
            # Skipped frames are discarded here — tracking, YOLO, face, encoding
            # are all bypassed.  The most recently captured frame is always the
            # one that gets through (VideoStream already drops stale frames).
            self._adapt_skip()

            priority = self._frame_priority()
            if priority == 2:
                effective_skip = 0
            elif priority == 1:
                effective_skip = self._current_skip // 2
            else:
                effective_skip = self._current_skip

            if self._skip_counter < effective_skip:
                self._skip_counter += 1
                logger.debug(
                    "[stream=%s] frame skipped (%d/%d) priority=%d pressure=%.2f",
                    self.stream_id[:8], self._skip_counter,
                    effective_skip, priority, self._compute_pressure(),
                )
                continue
            self._skip_counter = 0

            t0 = time.monotonic()
            pf = self._process_frame(frame)
            elapsed = time.monotonic() - t0

            self._fps_deque.append(1.0 / elapsed if elapsed > 0 else 0.0)
            pf.fps_sample = round(
                sum(self._fps_deque) / max(len(self._fps_deque), 1), 1
            )
            self._last_had_violation = bool(pf.violation_delta)

            try:
                self.output_queue.put_nowait(pf)
            except queue.Full:
                pass   # encoding loop is behind — drop, never block

            # Timeout flush — ensures buffered violations are not held
            # indefinitely when the stream is quiet (below batch size).
            if (time.monotonic() - self._violation_last_flush) >= self._violation_batch_timeout:
                self._flush_violations()

    # ── Face worker loop ──────────────────────────────────────────────────────

    def _face_worker_loop(self) -> None:
        """
        Dedicated daemon thread per stream.

        Reads _FaceJob items from face_queue, runs embed_crop(), applies the
        quality gate, updates track embeddings, and triggers identity assignment
        when the track is confirmed.  All mutations to track fields shared with
        the processing loop are performed under track.lock.

        The DB call (match_and_store_track) is made OUTSIDE track.lock to avoid
        holding the lock during I/O.  A double-check guard after re-acquiring
        the lock prevents two workers from assigning identity to the same track
        if jobs were queued rapidly (e.g. violation burst).
        """
        sid8 = self.stream_id[:8]

        while not self.stop_event.is_set():
            try:
                job = self.face_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                # ── Embedding (blocking InsightFace call — isolated here) ──────────
                with self.app.app_context():
                    faces = self.pipeline.embed_crop(job.crop)

                now_mono = time.monotonic()

                if not faces:
                    # Record the attempt so the cooldown timer advances correctly.
                    with job.track.lock:
                        job.track.last_identity_check = now_mono
                    continue

                best = max(faces, key=lambda f: f["quality_score"])

                # ── Quality gate + embedding update (under lock) ──────────────────
                should_assign  = False
                embs_copy:  list = []
                quals_copy: list = []

                with job.track.lock:
                    avg_quality = (
                        sum(job.track.pending_quality) / len(job.track.pending_quality)
                        if job.track.pending_quality else 0.0
                    )
                    few_embeddings = job.track.n_embeddings < 3
                    # Stable identified tracks require a meaningful quality improvement
                    # (EMBED_QUALITY_IMPROVE_MARGIN above average) to accept a new
                    # embedding — marginal frames don't move the prototype much and
                    # trigger unnecessary DB writes downstream.
                    # Unidentified tracks and low-count tracks use the relaxed gate.
                    stable = (
                        job.track.identity_id is not None
                        and job.track.last_match_confidence >= self._strong_match_threshold
                    )
                    if stable and not few_embeddings:
                        quality_ok = best["quality_score"] > avg_quality * (1.0 + self._quality_improve_margin)
                    else:
                        quality_ok = best["quality_score"] > avg_quality or few_embeddings

                    if quality_ok:
                        # deque(maxlen=MAX_PENDING_EMBEDDINGS) evicts oldest automatically —
                        # no manual cap or O(n) pop(0) needed.
                        job.track.add_embedding(best["embedding"], best["quality_score"])
                        logger.debug(
                            "[stream=%s] track=%d embedding added quality=%.3f n_embs=%d",
                            sid8, job.track.track_id, best["quality_score"], job.track.n_embeddings,
                        )
                    else:
                        logger.debug(
                            "[stream=%s] track=%d embedding skipped — quality %.3f <= avg %.3f",
                            sid8, job.track.track_id, best["quality_score"], avg_quality,
                        )

                    job.track.last_identity_check = now_mono

                    if job.has_violation and job.track.identity_id is not None:
                        logger.debug("[stream=%s] track=%d identity recheck triggered",
                                     sid8, job.track.track_id)

                    # Check whether this track is ready for identity assignment.
                    # frames_seen is written only by the processing loop (safe to
                    # read without lock), but n_embeddings reads pending_embeddings
                    # which we own under lock — call is_confirmed() here.
                    if (
                        self.tracker.is_confirmed(job.track)
                        and job.track.identity_id is None
                        and job.track.pending_embeddings
                    ):
                        should_assign = True
                        embs_copy  = list(job.track.pending_embeddings)
                        quals_copy = list(job.track.pending_quality)

                # ── Identity assignment — DB I/O outside lock ─────────────────────
                if should_assign:
                    logger.debug("[stream=%s] track=%d confirmed — matching identity",
                                 sid8, job.track.track_id)
                    with self.app.app_context():
                        identity_id, label, score = self.pipeline.match_and_store_track(
                            embs_copy, quals_copy,
                            stream_id=self.stream_id,
                        )
                    with job.track.lock:
                        # Guard: another job may have raced and already assigned an
                        # identity (e.g. two violation frames queued back-to-back).
                        # First writer wins.
                        if job.track.identity_id is None:
                            job.track.identity_id             = identity_id
                            job.track.identity_label          = label
                            job.track.last_match_confidence   = score
                            job.track.embeddings_at_last_match = job.track.n_embeddings
                            logger.debug(
                                "[stream=%s] track=%d -> '%s' (score=%.3f)",
                                sid8, job.track.track_id, label, score,
                            )
                        else:
                            logger.debug(
                                "[stream=%s] track=%d identity already assigned — skipping race result",
                                sid8, job.track.track_id,
                            )
            except Exception as exc:
                import traceback
                logger.error("[stream=%s] face_worker_loop error: %s\n%s",
                             sid8, exc, traceback.format_exc())
                continue

    # ── Encoding loop ─────────────────────────────────────────────────────────

    def _encoding_loop(self) -> None:
        """
        Reads _ProcessedFrame items from output_queue.
        Handles annotation, JPEG encoding (throttled by encode_every_n), and
        stats update under store_lock.  Runs in its own daemon thread so it
        never stalls the processing loop.
        """
        enc_count = 0
        while not self.stop_event.is_set():
            try:
                pf = self.output_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                enc_count += 1
                encode_this_frame = (enc_count % self.encode_every_n == 0)

                with self.store_lock:
                    if self.stream_id in self.stream_store:
                        stats = self.stream_store[self.stream_id]["stats"]
                        stats["fps"]             = pf.fps_sample
                        stats["frame_count"]     = pf.frame_count
                        stats["violation_count"] = (
                            stats.get("violation_count", 0) + pf.violation_delta
                        )
                        stats["last_detections"] = pf.detections
                        stats["resolution"]      = [pf.orig_w, pf.orig_h]

                        if encode_this_frame:
                            _annotate(pf.frame, pf.detections)
                            ret, buf = cv2.imencode(
                                ".jpg", pf.frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                            )
                            if ret:
                                self.stream_store[self.stream_id]["frame"] = buf.tobytes()
            except Exception as exc:
                import traceback
                logger.error("Encoding loop error: %s\n%s", exc, traceback.format_exc())
                continue


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