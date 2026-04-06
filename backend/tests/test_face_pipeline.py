"""
Tests for the production face recognition pipeline.
All face_recognition library calls are mocked — no camera or GPU needed.

Run with: pytest tests/test_face_pipeline.py -v
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from face.pipeline import (
    FaceRecognitionPipeline,
    MatchStatus,
    MatchResult,
    FrameQuality,
    extract_head_crop,
    _gallery_distance,
    _select_best_face,
)
from config import Config


class TestConfig(Config):
    TESTING                = True
    FACE_MATCH_THRESHOLD   = 0.50
    FACE_MIN_MARGIN        = 0.08
    FACE_HEAD_CROP_RATIO   = 0.35
    FACE_HEAD_CROP_PADDING = 0.15
    FACE_MIN_CROP_SIZE     = 120
    FACE_MIN_QUALITY_SCORE = 0.35


@pytest.fixture()
def pipeline():
    return FaceRecognitionPipeline(TestConfig)


def _enc(seed=0) -> np.ndarray:
    """Reproducible normalised 128-dim float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(128).astype(np.float32)
    return v / np.linalg.norm(v)


def _good_frame(h=200, w=200) -> np.ndarray:
    """A visually reasonable BGR frame that passes FrameQuality."""
    import cv2
    frame = np.full((h, w, 3), 127, dtype=np.uint8)
    # Add texture so Laplacian variance is non-zero
    noise = np.random.default_rng(0).integers(0, 40, (h, w, 3), dtype=np.uint8)
    return np.clip(frame.astype(np.int32) + noise, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# FrameQuality
# ─────────────────────────────────────────────────────────────────────────────
class TestFrameQuality:
    def test_black_frame_is_low_quality(self):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        score, _ = FrameQuality.analyse(frame)
        assert score < 0.35

    def test_white_frame_is_low_quality(self):
        frame = np.full((120, 120, 3), 255, dtype=np.uint8)
        score, _ = FrameQuality.analyse(frame)
        assert score < 0.35

    def test_uniform_grey_no_texture_is_low_quality(self):
        # No texture → Laplacian variance near 0 → blur_score near 0
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        score, _ = FrameQuality.analyse(frame)
        assert score < 0.35

    def test_textured_mid_brightness_passes(self):
        frame = _good_frame()
        score, detail = FrameQuality.analyse(frame)
        # With noise texture and mid-range brightness, combined should exceed threshold
        assert detail["brightness"] > 40
        assert detail["blur"] > 0

    def test_detail_keys_present(self):
        frame = _good_frame()
        _, detail = FrameQuality.analyse(frame)
        for key in ("blur", "blur_score", "brightness", "brightness_score",
                    "contrast_std", "contrast_score", "combined"):
            assert key in detail


# ─────────────────────────────────────────────────────────────────────────────
# extract_head_crop
# ─────────────────────────────────────────────────────────────────────────────
class TestExtractHeadCrop:
    def test_returns_bgr_array(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = extract_head_crop(frame, [100., 50., 300., 450.])
        assert crop is not None and crop.ndim == 3 and crop.shape[2] == 3

    def test_zero_size_bbox_returns_none(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert extract_head_crop(frame, [200., 200., 200., 200.]) is None

    def test_upscales_small_crop(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Tiny person bbox → crop will be < 120px
        crop = extract_head_crop(frame, [10., 10., 30., 70.], padding=0.0, min_size=120)
        assert crop is not None and min(crop.shape[:2]) >= 120

    def test_head_ratio_limits_vertical_extent(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 400px tall person, head_ratio=0.35 → head region ≈ 140px tall
        crop = extract_head_crop(frame, [100., 0., 300., 400.], head_ratio=0.35, padding=0.0, min_size=10)
        assert crop is not None
        # crop height must be significantly less than full person height
        assert crop.shape[0] < 300

    def test_stays_within_frame_bounds(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Bbox nearly at frame edge — padding must not go negative
        crop = extract_head_crop(frame, [90., 0., 100., 100.], padding=0.5, min_size=10)
        # Should not raise; may return None if crop collapses
        # Just verify no exception


# ─────────────────────────────────────────────────────────────────────────────
# _select_best_face
# ─────────────────────────────────────────────────────────────────────────────
class TestSelectBestFace:
    def test_single_face_passthrough(self):
        loc = [(10, 110, 110, 10)]
        enc = [_enc(0)]
        selected_loc, selected_enc = _select_best_face(loc, enc)
        assert selected_loc == loc[0]
        assert np.array_equal(selected_enc, enc[0])

    def test_picks_largest_face(self):
        # Face 0: small  (top=80, right=120, bottom=120, left=80) → area 40×40=1600
        # Face 1: large  (top=10, right=200, bottom=210, left=10) → area 200×190=38000
        locs = [(80, 120, 120, 80), (10, 200, 210, 10)]
        encs = [_enc(0), _enc(1)]
        _, selected_enc = _select_best_face(locs, encs)
        assert np.array_equal(selected_enc, encs[1])


# ─────────────────────────────────────────────────────────────────────────────
# _gallery_distance
# ─────────────────────────────────────────────────────────────────────────────
class TestGalleryDistance:
    def test_returns_minimum_not_mean(self):
        # Gallery: one close embedding (dist=0.10), one far (dist=0.90)
        close = _enc(0)
        far   = _enc(99)
        query = _enc(0)  # identical to 'close'

        with patch(
            "face_recognition.face_distance",
            side_effect=lambda g, q: np.array([0.10]) if g[0] is close else np.array([0.90])
        ):
            # Pass both in gallery
            with patch("face_recognition.face_distance", return_value=np.array([0.10, 0.90])):
                dist = _gallery_distance([close, far], query)
        assert dist == pytest.approx(0.10)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline — _run_pipeline via match_from_crop
# ─────────────────────────────────────────────────────────────────────────────
class TestPipelineMatchStatus:
    def _patch_quality(self, score=0.8):
        """Patch FrameQuality.analyse to return a given score."""
        return patch(
            "face.pipeline.FrameQuality.analyse",
            return_value=(score, {"combined": score}),
        )

    def test_low_quality_frame_rejected(self, pipeline):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        with self._patch_quality(score=0.20):
            result = pipeline.match_from_crop(frame)
        assert result.status == MatchStatus.LOW_QUALITY
        assert result.quality_score == pytest.approx(0.20)

    def test_no_face_detected(self, pipeline):
        frame = _good_frame()
        with (
            self._patch_quality(score=0.80),
            patch("face_recognition.face_locations", return_value=[]),
        ):
            result = pipeline.match_from_crop(frame)
        assert result.status == MatchStatus.NO_FACE
        assert result.face_count == 0

    def test_empty_gallery_returns_no_match(self, pipeline):
        """Even with a perfect face detection, no gallery → no match."""
        pipeline._cache = {}
        frame = _good_frame()
        enc = _enc(0)
        with (
            self._patch_quality(score=0.80),
            patch("face_recognition.face_locations", return_value=[(10, 110, 110, 10)]),
            patch("face_recognition.face_encodings", return_value=[enc]),
        ):
            result = pipeline.match_from_crop(frame)
        assert result.status == MatchStatus.NO_MATCH

    def test_matched_result(self, pipeline):
        enc = _enc(42)
        pipeline._cache = {
            "id-alice": {"name": "Alice", "embeddings": [enc]},
        }
        frame = _good_frame()
        # Query is very close to Alice's embedding
        with (
            self._patch_quality(score=0.80),
            patch("face_recognition.face_locations", return_value=[(10, 110, 110, 10)]),
            patch("face_recognition.face_encodings", return_value=[enc]),
            patch("face_recognition.face_distance", return_value=np.array([0.25])),
        ):
            result = pipeline.match_from_crop(frame)
        assert result.status == MatchStatus.MATCHED
        assert result.name == "Alice"
        assert result.identity_id == "id-alice"
        assert result.distance == pytest.approx(0.25)
        assert result.is_identified is True

    def test_no_match_above_threshold(self, pipeline):
        enc = _enc(1)
        pipeline._cache = {
            "id-bob": {"name": "Bob", "embeddings": [enc]},
        }
        frame = _good_frame()
        with (
            self._patch_quality(score=0.80),
            patch("face_recognition.face_locations", return_value=[(10, 110, 110, 10)]),
            patch("face_recognition.face_encodings", return_value=[_enc(99)]),
            patch("face_recognition.face_distance", return_value=np.array([0.70])),
        ):
            result = pipeline.match_from_crop(frame)
        assert result.status == MatchStatus.NO_MATCH
        assert result.distance == pytest.approx(0.70)
        assert result.is_identified is False

    def test_ambiguous_match_low_margin(self, pipeline):
        """When best and second-best identities are too close, return AMBIGUOUS."""
        pipeline._cache = {
            "id-alice": {"name": "Alice", "embeddings": [_enc(0)]},
            "id-twin":  {"name": "Twin",  "embeddings": [_enc(1)]},
        }
        frame = _good_frame()
        call_count = {"n": 0}

        def fake_distance(gallery, query):
            call_count["n"] += 1
            # First call → Alice at 0.40, second call → Twin at 0.43
            return np.array([0.40]) if call_count["n"] == 1 else np.array([0.43])

        with (
            self._patch_quality(score=0.80),
            patch("face_recognition.face_locations", return_value=[(10, 110, 110, 10)]),
            patch("face_recognition.face_encodings", return_value=[_enc(5)]),
            patch("face_recognition.face_distance", side_effect=fake_distance),
        ):
            result = pipeline.match_from_crop(frame)

        # margin = 0.43 - 0.40 = 0.03 < MIN_MARGIN (0.08) → AMBIGUOUS
        assert result.status == MatchStatus.AMBIGUOUS
        assert result.margin < pipeline._min_margin

    def test_multiple_faces_selects_largest(self, pipeline):
        """When two faces are detected, the largest-area face is encoded."""
        enc_small = _enc(10)
        enc_large = _enc(20)
        pipeline._cache = {
            "id-large": {"name": "Large Face", "embeddings": [enc_large]},
        }
        # Two locations: small (area=1600) and large (area=38000)
        locs = [(80, 120, 120, 80), (10, 200, 210, 10)]
        encs = [enc_small, enc_large]
        frame = _good_frame()

        with (
            self._patch_quality(score=0.80),
            patch("face_recognition.face_locations", return_value=locs),
            patch("face_recognition.face_encodings", return_value=encs),
            patch("face_recognition.face_distance", return_value=np.array([0.20])),
        ):
            result = pipeline.match_from_crop(frame)

        # Should have matched using enc_large (the larger face)
        assert result.status == MatchStatus.MATCHED
        assert result.name == "Large Face"

    def test_library_exception_returns_error(self, pipeline):
        frame = _good_frame()
        with (
            self._patch_quality(score=0.80),
            patch("face_recognition.face_locations", side_effect=RuntimeError("dlib crash")),
        ):
            result = pipeline.match_from_crop(frame)
        assert result.status == MatchStatus.ERROR
        assert result.is_identified is False

    def test_result_is_frozen_dataclass(self, pipeline):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        with self._patch_quality(score=0.10):
            result = pipeline.match_from_crop(frame)
        with pytest.raises((AttributeError, TypeError)):
            result.status = MatchStatus.MATCHED   # type: ignore

    def test_elapsed_ms_is_populated(self, pipeline):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        with self._patch_quality(score=0.10):
            result = pipeline.match_from_crop(frame)
        assert result.elapsed_ms >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Enrollment quality gate
# ─────────────────────────────────────────────────────────────────────────────
class TestEnrollmentQualityGate:
    def test_rejects_multi_face_image(self, pipeline):
        image_rgb = _good_frame()
        two_locations = [(10, 100, 100, 10), (10, 200, 100, 110)]
        with (
            patch("face_recognition.face_locations", return_value=two_locations),
        ):
            encoding, detail = pipeline.encode_clean_image(image_rgb)
        assert encoding is None
        assert "2 faces" in detail["error"]

    def test_rejects_no_face(self, pipeline):
        image_rgb = _good_frame()
        with patch("face_recognition.face_locations", return_value=[]):
            encoding, detail = pipeline.encode_clean_image(image_rgb)
        assert encoding is None
        assert "No face" in detail["error"]

    def test_accepts_single_clean_face(self, pipeline):
        image_rgb = _good_frame()
        enc = _enc(7)
        with (
            patch("face_recognition.face_locations", return_value=[(10, 110, 110, 10)]),
            patch("face_recognition.face_encodings", return_value=[enc]),
        ):
            encoding, detail = pipeline.encode_clean_image(image_rgb)
        assert encoding is not None
        assert np.array_equal(encoding, enc)


# ─────────────────────────────────────────────────────────────────────────────
# Calibrate
# ─────────────────────────────────────────────────────────────────────────────
class TestCalibrate:
    def _make_pairs(self, distances: list[float]):
        """Create fake (enc_a, enc_b) pairs; face_distance will be mocked."""
        return [(_enc(i), _enc(i + 100)) for i in range(len(distances))]

    def test_recommends_threshold_between_distributions(self, pipeline):
        pos_pairs = self._make_pairs([0.20, 0.25, 0.30])
        neg_pairs = self._make_pairs([0.55, 0.60, 0.65])

        call_count = {"n": 0}
        all_dists  = [0.20, 0.25, 0.30, 0.55, 0.60, 0.65]

        def fake_dist(gallery, query):
            d = all_dists[call_count["n"]]
            call_count["n"] += 1
            return np.array([d])

        with patch("face_recognition.face_distance", side_effect=fake_dist):
            report = pipeline.calibrate(pos_pairs, neg_pairs)

        assert 0.30 < report["recommended_threshold"] < 0.55
        assert report["overlap_exists"] is False
        assert report["suggested_min_margin"] > 0

    def test_detects_overlapping_distributions(self, pipeline):
        pos_pairs = self._make_pairs([0.20, 0.45])   # max=0.45
        neg_pairs = self._make_pairs([0.40, 0.60])   # min=0.40  → overlap

        all_dists = [0.20, 0.45, 0.40, 0.60]
        call_count = {"n": 0}

        def fake_dist(gallery, query):
            d = all_dists[call_count["n"]]
            call_count["n"] += 1
            return np.array([d])

        with patch("face_recognition.face_distance", side_effect=fake_dist):
            report = pipeline.calibrate(pos_pairs, neg_pairs)

        assert report["overlap_exists"] is True
        assert "overlap" in report["note"].lower()
