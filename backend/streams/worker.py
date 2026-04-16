"""
Stream worker — YOLO + IoU tracking + violation dispatch
=========================================================
New in this version: FaceTracker sits between YOLO detection and the
face pipeline.  Embeddings are accumulated per track (not per frame),
and identity assignment is delayed until TRACK_MIN_FRAMES frames and
TRACK_MIN_EMBEDDINGS embeddings have been collected.

Flow
----
Frame → YOLO → persons[] → FaceTracker.update(bbox) → Track
                                                         │
                                ┌────────────────────────┘
                                │
                track.frames_lost == 0 (currently visible)?
                                │
                         embed_crop() on person crop
                                │
                         track.add_embedding(emb, quality)
                                │
              tracker.is_confirmed(track)?   [≥ min_frames AND ≥ min_embeddings]
                     │                │
                    No               Yes
                     │                └── track.identity_id already set?
                     │                          │              │
                     │                         Yes            No
                     │                          │              └── match_and_store_track()
                     │                          │                  set track.identity_id
                     │
               (skip — accumulating)
                                │
                    ViolationJob for associated violations
                    using track.identity_id + track-level cooldown

BUG FIX (this version)
-----------------------
Previously, violation dispatch was gated entirely on track.identity_id being set.
In PPE footage, helmets/masks prevent InsightFace from detecting faces, so
n_embeddings never reaches TRACK_MIN_EMBEDDINGS, is_confirmed() never fires,
track.identity_id stays None, and no violations are ever stored.

Fix: gate violation dispatch on frame confirmation (frames_seen >= min_frames),
not on identity assignment.  Tracks with no face embeddings still dispatch
violations with identity_id=None; the task queue slow-path handles those.
"""

import logging
import time
from collections import deque
from threading import Event, Thread
from typing import Optional

import cv2
import numpy as np

from detection.association import split_detections, check_association
from streams.tracker import FaceTracker

logger = logging.getLogger(__name__)


class VideoStream:
    def __init__(self, src, fps_limit: int = 30):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._interval = 1.0 / max(1, fps_limit)
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
                self.grabbed, self.frame = grabbed, frame
            sleep = self._interval - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def read(self) -> Optional[np.ndarray]:
        return self.frame if self.grabbed else None

    def stop(self) -> None:
        self.stopped = True
        time.sleep(0.1)
        if self.stream.isOpened():
            self.stream.release()


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


def stream_worker(app, stream_id, source_type, source_path, stop_event, stream_store, store_lock):
    with app.app_context():
        yolo          = app.extensions["yolo_service"]
        task_queue    = app.extensions["task_queue"]
        pipeline      = app.extensions["face_pipeline"]
        jpeg_quality  = app.config["STREAM_JPEG_QUALITY"]
        violation_cd  = app.config["VIOLATION_COOLDOWN_SECONDS"]
        fps_limit     = app.config["FRAME_RATE_LIMIT"]

        tracker = FaceTracker(
            iou_threshold  = app.config.get("TRACK_IOU_THRESHOLD",  0.30),
            max_lost       = app.config.get("TRACK_MAX_LOST",       10),
            min_frames     = app.config.get("TRACK_MIN_FRAMES",     3),
            min_embeddings = app.config.get("TRACK_MIN_EMBEDDINGS", 3),
        )

    vs: Optional[VideoStream] = None
    try:
        src = int(source_path) if source_type == "webcam" else source_path
        vs  = VideoStream(src=src, fps_limit=fps_limit).start()
        time.sleep(0.5)

        if not vs.stream.isOpened():
            logger.error("Cannot open source: %s", src)
            return

        fps_deque   : deque = deque(maxlen=30)
        frame_count : int   = 0

        logger.info("Stream %s started (%s).", stream_id, src)

        while not stop_event.is_set():
            frame = vs.read()
            if frame is None:
                time.sleep(0.05)
                continue

            t0 = time.monotonic()

            # ── YOLO ──────────────────────────────────────────────────────────
            with app.app_context():
                detections = yolo.inference(frame)

            persons, violations = split_detections(detections)

            # DEBUG: log detection counts every 30 frames
            if frame_count % 30 == 0:
                logger.debug(
                    "[stream=%s frame=%d] YOLO: %d persons, %d violations detected",
                    stream_id[:8], frame_count, len(persons), len(violations),
                )

            # ── Per-person tracking + embedding ───────────────────────────────
            active_bboxes: list[list[float]] = []

            for person in persons:
                bbox = person["bbox"]
                active_bboxes.append(bbox)

                # Update (or create) a track for this detection
                track = tracker.update(bbox)

                # Extract face embedding from person crop every frame
                px1, py1, px2, py2 = map(int, bbox)
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(frame.shape[1], px2), min(frame.shape[0], py2)
                crop = frame[py1:py2, px1:px2]

                if crop.size == 0:
                    continue

                with app.app_context():
                    faces = pipeline.embed_crop(crop)

                # DEBUG: log embedding result periodically per track
                if frame_count % 30 == 0:
                    logger.debug(
                        "[stream=%s] track=%d frames=%d embs=%d faces_detected=%d",
                        stream_id[:8], track.track_id,
                        track.frames_seen, track.n_embeddings, len(faces),
                    )

                if faces:
                    best = max(faces, key=lambda f: f["quality_score"])
                    track.add_embedding(best["embedding"], best["quality_score"])

                # ── Identity assignment (delayed until confirmed by both gates) ──
                if tracker.is_confirmed(track) and track.identity_id is None:
                    # DEBUG: log before identity match
                    logger.debug(
                        "[stream=%s] track=%d confirmed (frames=%d, embs=%d) — matching identity",
                        stream_id[:8], track.track_id,
                        track.frames_seen, track.n_embeddings,
                    )
                    if track.pending_embeddings:
                        with app.app_context():
                            identity_id, label, score = pipeline.match_and_store_track(
                                track.pending_embeddings,
                                track.pending_quality,
                                stream_id=stream_id,
                            )
                        track.identity_id    = identity_id
                        track.identity_label = label
                        logger.debug(
                            "[stream=%s] track=%d → identity '%s' (score=%.3f)",
                            stream_id[:8], track.track_id, label, score,
                        )

                # ── Violation dispatch ─────────────────────────────────────────
                # Gate: track must be frame-confirmed (frames_seen >= min_frames).
                #
                # FIX: Previously gated on track.identity_id being set, which
                # requires face embeddings.  In PPE footage, helmets/masks block
                # InsightFace → no embeddings → identity never created → violations
                # silently dropped forever.
                #
                # Now: dispatch as soon as the person has been consistently visible
                # for min_frames frames.  Tracks without a face identity dispatch
                # with identity_id=None; the task queue slow-path handles those.
                if track.frames_seen < tracker.min_frames:
                    continue  # still too new — avoid one-frame ghost violations

                associated = [
                    v for v in violations
                    if check_association(bbox, v["bbox"])
                ]
                if not associated:
                    continue

                now = time.monotonic()
                # Track-level cooldown — one violation log per track per window
                if now - track.last_violation_time < violation_cd:
                    continue

                track.last_violation_time = now

                for viol in associated:
                    # DEBUG: log before DB insert
                    logger.debug(
                        "[stream=%s] track=%d dispatching violation '%s' "
                        "(identity=%s, frames=%d, embs=%d)",
                        stream_id[:8], track.track_id, viol["class_name"],
                        track.identity_id, track.frames_seen, track.n_embeddings,
                    )
                    from tasks.queue import ViolationJob
                    task_queue.put(ViolationJob(
                        stream_id=stream_id,
                        violation_type=viol["class_name"],
                        confidence=viol["confidence"],
                        person_crop_bgr=crop.copy(),
                        person_bbox=bbox,
                        # identity_id may be None if face not detectable (PPE occlusion).
                        # task queue slow-path will attempt embed_crop + match_or_create.
                        identity_id=track.identity_id,
                        identity_label=track.identity_label or "Unknown Person",
                    ))

            # Mark tracks not seen this frame
            tracker.mark_missing(active_bboxes)

            # ── Encode ────────────────────────────────────────────────────────
            _annotate(frame, detections)
            ret, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            if not ret:
                continue

            frame_count += 1
            elapsed = time.monotonic() - t0
            fps_deque.append(1.0 / elapsed if elapsed > 0 else 0.0)

            with store_lock:
                if stream_id in stream_store:
                    stream_store[stream_id]["frame"] = buf.tobytes()
                    stream_store[stream_id]["stats"].update({
                        "fps":             round(sum(fps_deque) / len(fps_deque), 1),
                        "frame_count":     frame_count,
                        "violation_count": stream_store[stream_id]["stats"].get("violation_count", 0)
                                           + (1 if any(not d["safe"] for d in detections) else 0),
                        "last_detections": detections,
                        "resolution":      [frame.shape[1], frame.shape[0]],
                    })

    except Exception as exc:
        import traceback
        logger.error("Stream %s crashed: %s\n%s", stream_id, exc, traceback.format_exc())
    finally:
        if vs:
            vs.stop()
        logger.info("Stream %s stopped.", stream_id)