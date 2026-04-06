"""
FaceRecognitionPipeline  —  production rewrite
===============================================

Encoding pipeline
-----------------
Raw YOLO person bbox  →  head_crop()  →  quality_check()  →  face_locations()
→  _select_best_face()  →  face_encodings()  →  128-dim float32 vector

Matching strategy
-----------------
For each candidate identity:
    distances = L2(query_encoding, all_gallery_embeddings_for_identity)
    score     = min(distances)          # closest single embedding wins
    margin    = second_best - best      # confidence gap between top-2 identities

Accept only when:
    best_score  <=  HARD_THRESHOLD      (absolute gate)
    margin      >=  MIN_MARGIN          (reject when two identities are too close)

Why min() instead of mean()
----------------------------
mean() rewards identities with many diverse embeddings regardless of whether
any single one is actually close.  min() asks the correct question:
"does the gallery contain at least one embedding that strongly resembles this query?"

Edge cases handled
------------------
  1. No face detected         → MatchResult(status=NO_FACE)
  2. Low-quality frame        → MatchResult(status=LOW_QUALITY)
  3. Multiple faces in crop   → largest face by bounding-box area wins
  4. Unknown face             → MatchResult(status=NO_MATCH)
  5. Ambiguous match          → MatchResult(status=AMBIGUOUS)
  6. Empty gallery            → MatchResult(status=NO_MATCH)
  7. Exception in lib call    → MatchResult(status=ERROR)
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import threading
import time
from typing import Optional

import cv2
import face_recognition
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

class MatchStatus(str, enum.Enum):
    MATCHED     = "matched"
    NO_MATCH    = "no_match"
    AMBIGUOUS   = "ambiguous"
    NO_FACE     = "no_face"
    LOW_QUALITY = "low_quality"
    ERROR       = "error"


@dataclasses.dataclass(frozen=True)
class MatchResult:
    status:        MatchStatus
    identity_id:   Optional[str] = None
    name:          str   = "Unknown Person"
    distance:      float = 1.0
    margin:        float = 0.0
    quality_score: float = 0.0
    face_count:    int   = 0
    elapsed_ms:    float = 0.0

    @property
    def is_identified(self) -> bool:
        return self.status == MatchStatus.MATCHED

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Readers-writer lock
# ─────────────────────────────────────────────────────────────────────────────

class _RWLock:
    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0

    class _R:
        __slots__ = ("_l",)
        def __init__(self, l): self._l = l
        def __enter__(self):
            with self._l._cond: self._l._readers += 1
        def __exit__(self, *_):
            with self._l._cond:
                self._l._readers -= 1
                if self._l._readers == 0:
                    self._l._cond.notify_all()

    class _W:
        __slots__ = ("_l",)
        def __init__(self, l): self._l = l
        def __enter__(self):
            self._l._cond.acquire()
            while self._l._readers > 0:
                self._l._cond.wait()
        def __exit__(self, *_):
            self._l._cond.release()

    def read(self):  return self._R(self)
    def write(self): return self._W(self)


# ─────────────────────────────────────────────────────────────────────────────
# Image quality analysis
# ─────────────────────────────────────────────────────────────────────────────

class FrameQuality:
    """
    Three-axis quality score returning a combined [0.0 … 1.0] value.

    blur_score       — Laplacian variance
    brightness_score — how close mean pixel is to 127
    contrast_score   — pixel std deviation

    Combined via geometric mean so a single bad axis tanks the overall score.
    Rejection threshold: combined < 0.35 (configurable via FACE_MIN_QUALITY_SCORE).
    """

    BLUR_THRESHOLD   = 80.0
    BRIGHTNESS_MIN   = 40
    BRIGHTNESS_MAX   = 215
    CONTRAST_MIN     = 20.0

    @classmethod
    def analyse(cls, bgr_crop: np.ndarray) -> tuple[float, dict]:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

        laplacian_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score     = float(np.tanh(laplacian_var / cls.BLUR_THRESHOLD))

        mean_brightness = float(gray.mean())
        if mean_brightness < cls.BRIGHTNESS_MIN or mean_brightness > cls.BRIGHTNESS_MAX:
            brightness_score = 0.0
        else:
            brightness_score = 1.0 - abs(mean_brightness - 127) / 88.0

        std_dev        = float(gray.std())
        contrast_score = float(np.tanh(std_dev / cls.CONTRAST_MIN))

        combined = float(np.cbrt(blur_score * brightness_score * contrast_score))

        detail = {
            "blur":             round(laplacian_var,     2),
            "blur_score":       round(blur_score,        4),
            "brightness":       round(mean_brightness,   2),
            "brightness_score": round(brightness_score,  4),
            "contrast_std":     round(std_dev,           2),
            "contrast_score":   round(contrast_score,    4),
            "combined":         round(combined,           4),
        }
        return combined, detail


# ─────────────────────────────────────────────────────────────────────────────
# Head crop extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_head_crop(
    frame: np.ndarray,
    person_bbox: list[float],
    head_ratio: float = 0.35,
    padding:    float = 0.15,
    min_size:   int   = 120,
) -> Optional[np.ndarray]:
    """
    Extract the head region (BGR) from a full frame given a YOLO person bbox.
    Returns None when the crop would be empty or zero-area.
    """
    px1, py1, px2, py2 = map(int, person_bbox)
    pw = px2 - px1
    ph = py2 - py1

    if pw <= 0 or ph <= 0:
        return None

    head_bottom = py1 + int(ph * head_ratio)
    pad_x = int(pw * padding)
    pad_y = int(ph * padding)

    x1 = max(0, px1 - pad_x)
    y1 = max(0, py1 - pad_y)
    x2 = min(frame.shape[1], px2 + pad_x)
    y2 = min(frame.shape[0], head_bottom + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if min(h, w) < min_size:
        scale  = min_size / min(h, w)
        crop   = cv2.resize(
            crop, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )

    return crop   # BGR — caller converts to RGB when needed


def _select_best_face(
    locations: list[tuple],
    encodings: list[np.ndarray],
) -> tuple[tuple, np.ndarray]:
    """
    When multiple faces appear in one crop, select the largest by bbox area.
    face_recognition locations are (top, right, bottom, left).
    """
    if len(locations) == 1:
        return locations[0], encodings[0]

    areas = [(r - l) * (b - t) for t, r, b, l in locations]
    best  = int(np.argmax(areas))
    logger.debug(
        "Multiple faces (%d) — selected index %d (area=%d px²).",
        len(locations), best, areas[best],
    )
    return locations[best], encodings[best]


# ─────────────────────────────────────────────────────────────────────────────
# Gallery distance
# ─────────────────────────────────────────────────────────────────────────────

def _gallery_distance(gallery: list[np.ndarray], query: np.ndarray) -> float:
    """
    Return the MINIMUM L2 distance from query to any embedding in the gallery.

    Why min() not mean()
    --------------------
    mean() penalises large diverse galleries: a person with 30 varied-angle
    embeddings will score a higher mean distance than a person with 3 near-
    identical embeddings, even when the 30-embedding gallery is the correct
    identity.  min() asks "is there at least one strong match?" — which is
    the right question for a multi-angle enrollment gallery.
    The margin check guards against lucky near-hits in large galleries.
    """
    return float(np.min(face_recognition.face_distance(gallery, query)))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FaceRecognitionPipeline:
    """
    Thread-safe face recognition service.  Instantiate once, share everywhere.

    Public API
    ----------
    match_from_frame(frame_bgr, person_bbox) → MatchResult
    match_from_crop(head_bgr)                → MatchResult
    encode_clean_image(image_rgb)            → (encoding | None, detail_dict)
    add_embedding(identity_id, encoding)
    reload_cache()
    calibrate(positive_pairs, negative_pairs) → threshold recommendations
    """

    def __init__(self, config) -> None:
        self._threshold   = getattr(config, "FACE_MATCH_THRESHOLD",   0.50)
        self._min_margin  = getattr(config, "FACE_MIN_MARGIN",         0.08)
        self._min_quality = getattr(config, "FACE_MIN_QUALITY_SCORE",  0.35)
        self._head_ratio  = getattr(config, "FACE_HEAD_CROP_RATIO",    0.35)
        self._padding     = getattr(config, "FACE_HEAD_CROP_PADDING",  0.15)
        self._min_size    = getattr(config, "FACE_MIN_CROP_SIZE",      120)

        self._lock  = _RWLock()
        self._cache: dict = {}   # {identity_id: {"name": str, "embeddings": [ndarray]}}

    # ── Cache ──────────────────────────────────────────────────────────────────

    def reload_cache(self) -> int:
        from face.models import FaceIdentity
        identities = FaceIdentity.query.all()
        new: dict = {}
        total = 0
        for identity in identities:
            embs = [e.embedding for e in identity.embeddings.all()]
            if embs:
                new[identity.id] = {"name": identity.name, "embeddings": embs}
                total += len(embs)
        with self._lock.write():
            self._cache = new
        logger.info("Cache reloaded: %d identities, %d embeddings.", len(new), total)
        return total

    def _snapshot(self) -> dict:
        with self._lock.read():
            return dict(self._cache)

    # ── Entry points ───────────────────────────────────────────────────────────

    def match_from_frame(
        self, frame_bgr: np.ndarray, person_bbox: list[float]
    ) -> MatchResult:
        t0 = time.perf_counter()
        head_bgr = extract_head_crop(
            frame_bgr, person_bbox,
            head_ratio=self._head_ratio,
            padding=self._padding,
            min_size=self._min_size,
        )
        if head_bgr is None:
            return MatchResult(status=MatchStatus.NO_FACE, elapsed_ms=_ms(t0))
        return self._run_pipeline(head_bgr, t0)

    def match_from_crop(self, head_bgr: np.ndarray) -> MatchResult:
        return self._run_pipeline(head_bgr, time.perf_counter())

    # ── Core pipeline ──────────────────────────────────────────────────────────

    def _run_pipeline(self, head_bgr: np.ndarray, t0: float) -> MatchResult:
        # ── 1. Quality gate ────────────────────────────────────────────────────
        quality_score, quality_detail = FrameQuality.analyse(head_bgr)
        logger.debug("Quality: %s", quality_detail)

        if quality_score < self._min_quality:
            logger.debug("Rejected — quality=%.4f < %.4f", quality_score, self._min_quality)
            return MatchResult(
                status=MatchStatus.LOW_QUALITY,
                quality_score=quality_score,
                elapsed_ms=_ms(t0),
            )

        head_rgb = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)

        # ── 2. Face detection ──────────────────────────────────────────────────
        try:
            locations = face_recognition.face_locations(head_rgb, model="hog")
        except Exception as exc:
            logger.exception("face_locations: %s", exc)
            return MatchResult(
                status=MatchStatus.ERROR, quality_score=quality_score, elapsed_ms=_ms(t0)
            )

        face_count = len(locations)
        if face_count == 0:
            return MatchResult(
                status=MatchStatus.NO_FACE,
                quality_score=quality_score,
                face_count=0,
                elapsed_ms=_ms(t0),
            )

        # ── 3. Encoding ────────────────────────────────────────────────────────
        try:
            # num_jitters=1 at runtime for speed; use 2+ for enrollment
            encodings = face_recognition.face_encodings(head_rgb, locations, num_jitters=1)
        except Exception as exc:
            logger.exception("face_encodings: %s", exc)
            return MatchResult(
                status=MatchStatus.ERROR,
                quality_score=quality_score,
                face_count=face_count,
                elapsed_ms=_ms(t0),
            )

        if not encodings:
            return MatchResult(
                status=MatchStatus.NO_FACE,
                quality_score=quality_score,
                face_count=face_count,
                elapsed_ms=_ms(t0),
            )

        # ── 4. Select best face (largest when multiple found) ──────────────────
        _, query_enc = _select_best_face(locations, encodings)

        # ── 5. Gallery ranking ─────────────────────────────────────────────────
        cache = self._snapshot()
        if not cache:
            return MatchResult(
                status=MatchStatus.NO_MATCH,
                quality_score=quality_score,
                face_count=face_count,
                elapsed_ms=_ms(t0),
            )

        ranked = self._rank_identities(cache, query_enc)
        best_id, best_name, best_dist = ranked[0]
        second_dist = ranked[1][2] if len(ranked) >= 2 else 1.0
        margin = second_dist - best_dist

        # ── 6. Threshold + margin gate ─────────────────────────────────────────
        if best_dist > self._threshold:
            return MatchResult(
                status=MatchStatus.NO_MATCH,
                distance=best_dist,
                margin=margin,
                quality_score=quality_score,
                face_count=face_count,
                elapsed_ms=_ms(t0),
            )

        if margin < self._min_margin:
            logger.info(
                "Ambiguous — best=%s(%.4f) margin=%.4f < %.4f",
                best_name, best_dist, margin, self._min_margin,
            )
            return MatchResult(
                status=MatchStatus.AMBIGUOUS,
                name=best_name,
                identity_id=best_id,
                distance=best_dist,
                margin=margin,
                quality_score=quality_score,
                face_count=face_count,
                elapsed_ms=_ms(t0),
            )

        logger.info(
            "MATCHED %s  dist=%.4f  margin=%.4f  quality=%.4f  %.1fms",
            best_name, best_dist, margin, quality_score, _ms(t0),
        )
        return MatchResult(
            status=MatchStatus.MATCHED,
            identity_id=best_id,
            name=best_name,
            distance=best_dist,
            margin=margin,
            quality_score=quality_score,
            face_count=face_count,
            elapsed_ms=_ms(t0),
        )

    def _rank_identities(
        self, cache: dict, query_enc: np.ndarray
    ) -> list[tuple[str, str, float]]:
        scores = []
        for identity_id, data in cache.items():
            gallery = data["embeddings"]
            if gallery:
                scores.append((identity_id, data["name"], _gallery_distance(gallery, query_enc)))
        scores.sort(key=lambda x: x[2])
        return scores

    # ── Enrollment ─────────────────────────────────────────────────────────────

    def encode_clean_image(
        self, image_rgb: np.ndarray, num_jitters: int = 2
    ) -> tuple[Optional[np.ndarray], dict]:
        """
        Encode a clean enrollment photo.  Rejects multi-face images and low quality.
        Returns (encoding | None, detail_dict).
        """
        detail: dict = {}
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        quality, quality_detail = FrameQuality.analyse(bgr)
        detail["quality"] = quality_detail

        if quality < 0.25:
            detail["error"] = "Enrollment image quality too low."
            return None, detail

        try:
            locations = face_recognition.face_locations(image_rgb, model="hog")
        except Exception as exc:
            detail["error"] = f"face_locations: {exc}"
            return None, detail

        detail["face_count"] = len(locations)

        if not locations:
            detail["error"] = "No face detected."
            return None, detail

        if len(locations) > 1:
            detail["error"] = (
                f"{len(locations)} faces detected. "
                "Enrollment image must contain exactly one face."
            )
            return None, detail

        try:
            encodings = face_recognition.face_encodings(
                image_rgb, locations, num_jitters=num_jitters
            )
        except Exception as exc:
            detail["error"] = f"face_encodings: {exc}"
            return None, detail

        if not encodings:
            detail["error"] = "Encoding failed."
            return None, detail

        return encodings[0], detail

    def compute_embedding_quality(
        self, new_encoding: np.ndarray, identity_id: str
    ) -> float:
        """
        min-distance from new_encoding to existing gallery.
        Low  → near-duplicate (rejected below FACE_MIN_QUALITY_SCORE).
        High → genuinely new angle (accepted, enriches the gallery).
        """
        cache = self._snapshot()
        existing = cache.get(identity_id, {}).get("embeddings", [])
        if not existing:
            return 1.0
        return float(np.min(face_recognition.face_distance(existing, new_encoding)))

    def add_embedding(
        self,
        identity_id: str,
        encoding: np.ndarray,
        source_image: Optional[str] = None,
        quality_score: Optional[float] = None,
    ):
        from face.models import FaceEmbedding
        from extensions import db

        if quality_score is None:
            quality_score = self.compute_embedding_quality(encoding, identity_id)

        emb = FaceEmbedding(
            identity_id=identity_id,
            source_image=source_image,
            quality_score=quality_score,
        )
        emb.embedding = encoding
        db.session.add(emb)
        db.session.commit()
        self.reload_cache()
        return emb

    # ── Threshold calibration ──────────────────────────────────────────────────

    def calibrate(
        self,
        positive_pairs: list[tuple[np.ndarray, np.ndarray]],
        negative_pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> dict:
        """
        Compute recommended threshold and margin from labelled embedding pairs.

        positive_pairs : (enc_a, enc_b) where a and b ARE the same person
        negative_pairs : (enc_a, enc_b) where a and b are DIFFERENT people

        Returns a dict with eer_threshold, conservative_threshold, distributions,
        and suggested_min_margin for use in .env configuration.
        """
        pos = [float(face_recognition.face_distance([a], b)[0]) for a, b in positive_pairs]
        neg = [float(face_recognition.face_distance([a], b)[0]) for a, b in negative_pairs]

        pa = np.array(pos) if pos else np.array([0.6])
        na = np.array(neg) if neg else np.array([0.6])

        # EER threshold: midpoint between worst positive and best negative
        eer = float((pa.max() + na.min()) / 2.0)
        # Conservative: accepts everyone who should be accepted, more false positives
        conservative = float(pa.max() + 0.02)

        overlap = pa.max() > na.min()
        margin_suggestion = float((na.min() - pa.max()) * 0.30) if not overlap else 0.05

        return {
            "recommended_threshold":   round(eer,          4),
            "conservative_threshold":  round(conservative, 4),
            "positive_distances": {
                "mean":   round(float(pa.mean()), 4),
                "std":    round(float(pa.std()),  4),
                "max":    round(float(pa.max()),  4),
                "values": [round(d, 4) for d in pos],
            },
            "negative_distances": {
                "mean":   round(float(na.mean()), 4),
                "std":    round(float(na.std()),  4),
                "min":    round(float(na.min()),  4),
                "values": [round(d, 4) for d in neg],
            },
            "overlap_exists":        overlap,
            "suggested_min_margin":  round(margin_suggestion, 4),
            "note": (
                "Distributions overlap — impossible to achieve zero errors "
                "at this image quality.  Improve enrollment photos or lower "
                "FACE_MIN_QUALITY_SCORE to reject more runtime frames."
                if overlap else
                "Clean separation — use recommended_threshold and suggested_min_margin."
            ),
        }

    # ── Convenience ────────────────────────────────────────────────────────────

    def get_head_crop(
        self, frame: np.ndarray, person_bbox: list[float]
    ) -> Optional[np.ndarray]:
        return extract_head_crop(
            frame, person_bbox,
            head_ratio=self._head_ratio,
            padding=self._padding,
            min_size=self._min_size,
        )


# ─────────────────────────────────────────────────────────────────────────────
def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
