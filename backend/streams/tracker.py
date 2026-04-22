"""
Face tracker — temporal consistency across frames
==================================================
Sits between YOLO detection and the face pipeline.

Problem solved
--------------
Without tracking, every ViolationJob is treated as an independent stranger.
The same person in frame N and frame N+3 produces two separate embed_crop()
calls and two separate match_or_create() calls.  If the scores differ by even
0.05 across frames, one falls below the match threshold and creates a new identity.

Solution: Simple IoU-based tracker
-----------------------------------
Each track represents a person bounding box being followed across frames.
Tracks accumulate embeddings over time.  Identity assignment happens at the
track level, not the frame level.

Key properties
--------------
1. Same physical person → same track_id for consecutive frames.
2. Embeddings are averaged within a track before matching → more stable vector.
3. Identity is only created after TRACK_MIN_FRAMES frames AND TRACK_MIN_EMBEDDINGS
   successful embeddings.  One-frame ghosts never become identities.
4. Active tracks carry their assigned identity_id, so the cooldown in the
   task queue can use identity_id (not spatial grid) to prevent duplicate violations.

IoU tracking chosen over SORT/Deep SORT because:
- No additional dependencies
- Sufficient for a single-person-at-a-time surveillance scenario
- Deterministic, predictable behavior
- Zero GPU cost

Thread safety: FaceTracker instances are per-stream (created inside
stream_worker), so no locking is needed.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


def _iou(a: list[float], b: list[float]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    """One tracked person across frames."""
    _next_id = 1

    def __init__(self, bbox: list[float], iou_threshold: float):
        self.track_id      = Track._next_id
        Track._next_id    += 1
        self.bbox          = bbox
        self.iou_threshold = iou_threshold

        self.frames_seen   = 1          # total frames this track appeared in
        self.frames_lost   = 0          # consecutive frames without a match
        self.identity_id   : Optional[str] = None    # assigned after confirmation
        self.identity_label: str = ""

        # Accumulated embeddings (before identity assignment)
        self.pending_embeddings: list[np.ndarray] = []
        self.pending_quality:    list[float]       = []

        # After identity is assigned, keep the mean embedding for matching
        self.mean_embedding: Optional[np.ndarray] = None

        self.last_violation_time: float = 0.0   # monotonic, for violation cooldown

        # Stage 3: tracks when the last face embedding attempt was made for this
        # track.  Used by StreamWorker to enforce IDENTITY_RECHECK_SECONDS cooldown
        # — once an identity is assigned, face embedding is skipped until this
        # timestamp is older than the cooldown, reducing InsightFace load.
        self.last_identity_check: float = 0.0   # monotonic

    def update(self, bbox: list[float]) -> None:
        self.bbox         = bbox
        self.frames_seen += 1
        self.frames_lost  = 0

    def mark_lost(self) -> None:
        self.frames_lost += 1

    def matches(self, bbox: list[float]) -> bool:
        return _iou(self.bbox, bbox) >= self.iou_threshold

    def add_embedding(self, emb: np.ndarray, quality: float) -> None:
        """Accumulate an embedding.  Also updates mean_embedding for matching."""
        self.pending_embeddings.append(emb)
        self.pending_quality.append(quality)
        # Update running mean for quick lookup
        embs    = np.stack(self.pending_embeddings)
        weights = np.array(self.pending_quality, dtype=np.float32)
        wsum    = weights.sum()
        mean    = (embs * (weights / wsum)[:, None]).sum(axis=0)
        norm    = np.linalg.norm(mean)
        self.mean_embedding = mean / norm if norm > 0 else mean

    @property
    def best_embedding(self) -> Optional[np.ndarray]:
        """Highest-quality single embedding accumulated so far."""
        if not self.pending_embeddings:
            return None
        best_idx = int(np.argmax(self.pending_quality))
        return self.pending_embeddings[best_idx]

    @property
    def best_quality(self) -> float:
        return max(self.pending_quality) if self.pending_quality else 0.0

    @property
    def n_embeddings(self) -> int:
        return len(self.pending_embeddings)


class FaceTracker:
    """
    Per-stream IoU tracker.  Not thread-safe — call from one thread only.

    Usage (inside stream_worker):
        tracker = FaceTracker(config)
        ...
        for person_det in persons:
            track = tracker.update(person_det["bbox"])
            if track.frames_seen >= MIN_FRAMES and track.n_embeddings >= MIN_EMBS:
                # ready for identity assignment
    """

    def __init__(
        self,
        iou_threshold:  float = 0.30,
        max_lost:       int   = 10,
        min_frames:     int   = 3,
        min_embeddings: int   = 3,
    ):
        self.iou_threshold  = iou_threshold
        self.max_lost       = max_lost
        self.min_frames     = min_frames
        self.min_embeddings = min_embeddings
        self._tracks: list[Track] = []

    def update(self, bbox: list[float]) -> Track:
        """
        Match bbox to an existing track (IoU) or create a new one.
        Returns the matching Track.
        """
        # Find best-matching track
        best_track: Optional[Track] = None
        best_iou = self.iou_threshold - 1e-9   # at least iou_threshold to match

        for t in self._tracks:
            iou = _iou(t.bbox, bbox)
            if iou > best_iou:
                best_iou  = iou
                best_track = t

        if best_track is not None:
            best_track.update(bbox)
            return best_track

        # No match — create new track
        new_track = Track(bbox, self.iou_threshold)
        self._tracks.append(new_track)
        return new_track

    def mark_missing(self, active_bboxes: list[list[float]]) -> None:
        """
        Called after all current-frame detections are processed.
        Tracks not updated this frame are marked lost and evicted if past max_lost.
        """
        for t in self._tracks:
            if not any(_iou(t.bbox, b) >= self.iou_threshold for b in active_bboxes):
                t.mark_lost()

        self._tracks = [t for t in self._tracks if t.frames_lost <= self.max_lost]

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.frames_lost == 0]

    def is_confirmed(self, track: Track) -> bool:
        """True when the track has seen enough frames and embeddings for identity assignment."""
        return (
            track.frames_seen   >= self.min_frames
            and track.n_embeddings >= self.min_embeddings
        )