"""
StreamManager
=============
Owns all active stream state.  The Flask routes delegate all stream
lifecycle operations here — routes never touch the stream dict directly.

Thread safety
-------------
`_streams` is protected by `_lock` (threading.Lock).
Worker threads write to their stream's sub-dict via the same lock
(passed in as `store_lock`).
"""

import logging
import time
import uuid
from threading import Event, Lock, Thread
from typing import Optional

logger = logging.getLogger(__name__)


class StreamManager:
    def __init__(self):
        self._streams: dict = {}
        self._lock = Lock()
        self._app = None

    def init_app(self, app) -> None:
        self._app = app

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------
    def start(self, source_type: str, source_path: str, name: str) -> dict:
        """
        Start a new stream.  Returns stream metadata dict.
        Raises ValueError if MAX_CONCURRENT_STREAMS is reached.
        """
        max_streams = self._app.config["MAX_CONCURRENT_STREAMS"]

        with self._lock:
            if len(self._streams) >= max_streams:
                raise ValueError(
                    f"Maximum concurrent streams ({max_streams}) reached."
                )

        stream_id = str(uuid.uuid4())
        stop_event = Event()

        stream_entry = {
            "id": stream_id,
            "name": name,
            "source_type": source_type,
            "source_path": source_path,
            "started_at": time.time(),
            "stop_event": stop_event,
            "frame": None,
            "stats": {
                "fps": 0,
                "frame_count": 0,
                "violation_count": 0,
                "last_detections": [],
                "resolution": [0, 0],
            },
        }

        with self._lock:
            self._streams[stream_id] = stream_entry

        from streams.worker import stream_worker

        thread = Thread(
            target=stream_worker,
            args=(
                self._app,
                stream_id,
                source_type,
                source_path,
                stop_event,
                self._streams,
                self._lock,
            ),
            name=f"stream-{stream_id[:8]}",
            daemon=True,
        )
        thread.start()

        with self._lock:
            self._streams[stream_id]["thread"] = thread

        logger.info("Stream %s started: %s (%s)", stream_id, name, source_path)
        return {"stream_id": stream_id, "name": name}

    def stop(self, stream_id: str) -> bool:
        """
        Signal the worker thread to stop and remove the stream from state.
        Returns True if found and stopped, False if not found.
        """
        with self._lock:
            entry = self._streams.pop(stream_id, None)

        if not entry:
            return False

        entry["stop_event"].set()
        thread: Optional[Thread] = entry.get("thread")
        if thread and thread.is_alive():
            thread.join(timeout=5.0)  # wait for the worker to exit cleanly

        logger.info("Stream %s stopped.", stream_id)
        return True

    def stop_all(self) -> None:
        with self._lock:
            stream_ids = list(self._streams.keys())
        for sid in stream_ids:
            self.stop(sid)

    # ------------------------------------------------------------------
    # Frame delivery
    # ------------------------------------------------------------------
    def get_frame(self, stream_id: str) -> Optional[bytes]:
        with self._lock:
            entry = self._streams.get(stream_id)
        return entry["frame"] if entry else None

    def frame_generator(self, stream_id: str):
        """
        MJPEG frame generator for Flask Response(mimetype=multipart/...).
        Yields boundary-delimited JPEG frames.
        """
        while True:
            with self._lock:
                entry = self._streams.get(stream_id)
                if not entry or entry["stop_event"].is_set():
                    break
                frame = entry.get("frame")

            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            time.sleep(1.0 / 60)  # max 60fps delivery

    # ------------------------------------------------------------------
    # Stats / listing
    # ------------------------------------------------------------------
    def list_streams(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "stream_id": sid,
                    "name": s["name"],
                    "source_type": s["source_type"],
                    "started_at": s["started_at"],
                    "stats": s["stats"],
                }
                for sid, s in self._streams.items()
            ]

    def get_stats(self, stream_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._streams.get(stream_id)
        if not entry:
            return None
        return {**entry["stats"], "name": entry["name"], "stream_id": stream_id}
