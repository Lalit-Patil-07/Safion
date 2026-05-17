"""
Tests for the IoU face tracker.
Run with: pytest tests/test_tracker.py -v

No camera, no models — pure geometry and state machine logic.
"""
import time
import pytest
import numpy as np
from streams.tracker import FaceTracker, Track, _iou


# ── IoU helper ────────────────────────────────────────────────────────────────
class TestIou:
    def test_full_overlap_is_one(self):
        box = [0.0, 0.0, 100.0, 100.0]
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self):
        assert _iou([0.0, 0.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]) == 0.0

    def test_half_overlap(self):
        a = [0.0, 0.0, 100.0, 100.0]
        b = [50.0, 0.0, 150.0, 100.0]
        # intersection = 50×100=5000, union = 100×100 + 100×100 - 5000 = 15000
        assert _iou(a, b) == pytest.approx(5000 / 15000, rel=1e-3)

    def test_touching_edges_is_zero(self):
        assert _iou([0.0, 0.0, 50.0, 50.0], [50.0, 0.0, 100.0, 50.0]) == 0.0


# ── Track ─────────────────────────────────────────────────────────────────────
class TestTrack:
    def _track(self):
        Track._next_id = 1
        return Track([10.0, 10.0, 60.0, 60.0], iou_threshold=0.30)

    def test_initial_state(self):
        t = self._track()
        assert t.frames_seen == 1
        assert t.frames_lost == 0
        assert t.identity_id is None
        assert t.n_embeddings == 0

    def test_update_resets_lost(self):
        t = self._track()
        t.mark_lost()
        t.mark_lost()
        assert t.frames_lost == 2
        t.update([11.0, 11.0, 61.0, 61.0])
        assert t.frames_lost == 0
        assert t.frames_seen == 2

    def test_add_embedding_updates_mean(self):
        t = self._track()
        rng = np.random.default_rng(0)
        e1 = rng.random(512).astype(np.float32)
        e1 /= np.linalg.norm(e1)
        t.add_embedding(e1, 0.80)
        assert t.mean_embedding is not None
        assert t.n_embeddings == 1
        # Mean of one embedding equals itself
        np.testing.assert_allclose(t.mean_embedding, e1, atol=1e-5)

    def test_mean_embedding_is_quality_weighted(self):
        t = self._track()
        rng = np.random.default_rng(1)
        e1 = np.array([1.0] + [0.0] * 511, dtype=np.float32)
        e2 = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)
        t.add_embedding(e1, 0.9)   # higher quality
        t.add_embedding(e2, 0.1)   # lower quality
        mean = t.mean_embedding
        # mean should be closer to e1 (higher weight)
        assert float(np.dot(mean, e1)) > float(np.dot(mean, e2))

    def test_matches_uses_iou_threshold(self):
        t = self._track()   # bbox [10,10,60,60]
        # Overlapping box
        assert t.matches([12.0, 12.0, 62.0, 62.0]) is True
        # Non-overlapping box far away
        assert t.matches([200.0, 200.0, 300.0, 300.0]) is False

    def test_best_embedding_returns_highest_quality(self):
        t = self._track()
        rng = np.random.default_rng(2)
        embs = [rng.random(512).astype(np.float32) for _ in range(3)]
        qualities = [0.5, 0.9, 0.3]
        for e, q in zip(embs, qualities):
            e /= np.linalg.norm(e)
            t.add_embedding(e, q)
        best = t.best_embedding
        np.testing.assert_allclose(best, embs[1] / np.linalg.norm(embs[1]), atol=1e-5)


# ── FaceTracker ───────────────────────────────────────────────────────────────
class TestFaceTracker:
    def _tracker(self, min_frames=3, min_embeddings=3):
        return FaceTracker(
            iou_threshold=0.30,
            max_lost=5,
            min_frames=min_frames,
            min_embeddings=min_embeddings,
        )

    def test_new_detection_creates_track(self):
        tr = self._tracker()
        track = tr.update([10.0, 10.0, 60.0, 60.0])
        assert len(tr._tracks) == 1
        assert track.frames_seen == 1

    def test_same_bbox_updates_existing_track(self):
        tr = self._tracker()
        tr.update([10.0, 10.0, 60.0, 60.0])
        tr.update([12.0, 12.0, 62.0, 62.0])   # overlapping — same track
        assert len(tr._tracks) == 1
        assert tr._tracks[0].frames_seen == 2

    def test_non_overlapping_creates_second_track(self):
        tr = self._tracker()
        tr.update([10.0, 10.0, 60.0, 60.0])
        tr.update([200.0, 200.0, 260.0, 260.0])
        assert len(tr._tracks) == 2

    def test_mark_missing_increments_lost(self):
        tr = self._tracker()
        tr.update([10.0, 10.0, 60.0, 60.0])
        tr.mark_missing([])   # no active bboxes
        assert tr._tracks[0].frames_lost == 1

    def test_evicts_stale_track(self):
        tr = self._tracker()
        tr.update([10.0, 10.0, 60.0, 60.0])
        for _ in range(6):   # max_lost=5
            tr.mark_missing([])
        assert len(tr._tracks) == 0

    def test_is_confirmed_requires_min_frames_and_embeddings(self):
        tr = self._tracker(min_frames=3, min_embeddings=3)
        track = tr.update([10.0, 10.0, 60.0, 60.0])

        rng = np.random.default_rng(5)
        emb = rng.random(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Only 1 frame, 1 embedding
        track.add_embedding(emb, 0.7)
        assert tr.is_confirmed(track) is False

        # Simulate 2 more frames
        for i in range(2):
            track.update([11.0, 11.0, 61.0, 61.0])
            e = rng.random(512).astype(np.float32)
            e /= np.linalg.norm(e)
            track.add_embedding(e, 0.7)

        # Now 3 frames, 3 embeddings
        assert tr.is_confirmed(track) is True

    def test_identity_assignment_persists(self):
        tr = self._tracker()
        track = tr.update([10.0, 10.0, 60.0, 60.0])
        track.identity_id    = "test-uuid"
        track.identity_label = "Person_001"

        # Update track — identity should persist
        track.update([12.0, 12.0, 62.0, 62.0])
        assert track.identity_id == "test-uuid"

    def test_track_level_cooldown(self):
        tr = self._tracker()
        track = tr.update([10.0, 10.0, 60.0, 60.0])
        track.identity_id       = "uuid-1"
        track.last_violation_time = time.monotonic()

        # Should be in cooldown immediately after setting
        elapsed = time.monotonic() - track.last_violation_time
        assert elapsed < 1.0   # well within cooldown
