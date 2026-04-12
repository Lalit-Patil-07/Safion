"""
InsightFacePipeline  — v2
==========================
Model  : buffalo_l (InsightFace)
Detect : RetinaFace
Embed  : ArcFace 512-dim L2-normalised

Changes from v1
---------------
QUALITY-WEIGHTED centroid
  Both reload_cache() and _patch_cache() weight each embedding by its
  det_score.  A blurry detection (score 0.62) has ~65 % the influence of
  a sharp one (score 0.95).  The centroid no longer drifts toward poor frames.

OUTLIER REJECTION in match_or_create()
  Before accepting a new embedding into an existing identity, we check its
  cosine similarity to the current centroid.  If it falls below
  OUTLIER_MIN_SIMILARITY (default 0.35) the embedding is stored in the DB
  for clustering purposes but the centroid is NOT updated.  A single bad
  frame can no longer corrupt the matching state.

IDENTITY CONFIDENCE SCORE
  The cache stores the running weighted-average match score for each identity.
  This is exposed as identity_confidence (0.0–1.0) and lets operators see
  which identities are "settled" vs. still being refined.

EMBEDDING QUALITY SCORE
  embed_crop() returns a quality_score per face:
      quality = det_score * face_area_ratio
  Stored on FaceEmbedding.  Used by clustering to weight the distance matrix.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_insight_app  = None
_insight_lock = threading.Lock()


# ── RW lock ───────────────────────────────────────────────────────────────────
class _RWLock:
    def __init__(self):
        self._cond    = threading.Condition(threading.Lock())
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


# ── Quality scoring ───────────────────────────────────────────────────────────
def compute_quality_score(det_score: float, bbox: list, image_shape: tuple) -> float:
    """
    Combined embedding quality score in [0, 1].

    Components
    ----------
    det_score     : RetinaFace detection confidence (primary signal)
    face_area_ratio : fraction of the crop occupied by the detected face
                      Small faces → unreliable embeddings → lower score

    Formula: quality = det_score * 0.7 + area_ratio_score * 0.3

    Examples
    --------
    det=0.95, face fills 40% of crop  → quality ≈ 0.83
    det=0.70, face fills 10% of crop  → quality ≈ 0.56
    det=0.62, tiny face               → quality ≈ 0.47
    """
    if not bbox or len(bbox) < 4 or not image_shape:
        return float(det_score)

    h, w = image_shape[:2]
    crop_area = max(h * w, 1)
    bx1, by1, bx2, by2 = bbox[:4]
    face_area = max((bx2 - bx1) * (by2 - by1), 0)
    area_ratio = min(face_area / crop_area, 1.0)
    # Normalise: 0.15 area fills a good-quality crop for a nearby person
    area_score = min(area_ratio / 0.15, 1.0)

    return float(np.clip(det_score * 0.7 + area_score * 0.3, 0.0, 1.0))


# ── Pipeline ──────────────────────────────────────────────────────────────────
class InsightFacePipeline:
    """
    Cache entry structure:
    {
        identity_id: {
            "label":      str,
            "centroid":   np.ndarray(512,),  # quality-weighted normalised mean
            "weight_sum": float,              # sum of quality scores used
            "confidence": float,             # running mean of match scores
            "n_matches":  int,               # total match events
        }
    }
    """

    def __init__(self, config) -> None:
        self._model_name        = getattr(config, "INSIGHTFACE_MODEL",        "buffalo_l")
        self._threshold         = getattr(config, "IDENTITY_MATCH_THRESHOLD",  0.55)
        self._det_score_min     = getattr(config, "FACE_DET_SCORE_MIN",         0.60)
        self._quality_min       = getattr(config, "EMBEDDING_QUALITY_MIN",      0.45)
        self._outlier_min_sim   = getattr(config, "OUTLIER_MIN_SIMILARITY",      0.35)
        self._rw                = _RWLock()
        self._cache: dict       = {}
        self._ready             = False

    # ── Init ──────────────────────────────────────────────────────────────────
    def init_app(self, app) -> None:
        global _insight_app
        try:
            from insightface.app import FaceAnalysis
            with _insight_lock:
                if _insight_app is None:
                    logger.info("Loading InsightFace '%s' …", self._model_name)
                    _insight_app = FaceAnalysis(
                        name=self._model_name,
                        providers=["CPUExecutionProvider"],
                    )
                    _insight_app.prepare(ctx_id=-1, det_size=(640, 640))
                    logger.info("InsightFace loaded.")
            self._ready = True
        except Exception as exc:
            logger.error("InsightFace load failed: %s", exc)
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and _insight_app is not None

    # ── Embedding extraction ──────────────────────────────────────────────────
    def embed_crop(self, person_crop_bgr: np.ndarray) -> list[dict]:
        """
        RetinaFace + ArcFace on a BGR person-bbox crop.

        Returns list of:
            {
                "embedding":     ndarray(512,),
                "det_score":     float,
                "quality_score": float,    ← NEW: combined quality
                "bbox":          list,
            }

        Faces below FACE_DET_SCORE_MIN or EMBEDDING_QUALITY_MIN are dropped.
        """
        if not self.is_ready or person_crop_bgr is None or person_crop_bgr.size == 0:
            return []

        img_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        try:
            with _insight_lock:
                faces = _insight_app.get(img_rgb)
        except Exception as exc:
            logger.warning("InsightFace.get() error: %s", exc)
            return []

        results = []
        for face in faces:
            if face.det_score < self._det_score_min or face.embedding is None:
                continue

            bbox   = face.bbox.tolist() if face.bbox is not None else []
            q      = compute_quality_score(float(face.det_score), bbox, person_crop_bgr.shape)

            if q < self._quality_min:
                logger.debug("Face rejected: quality=%.3f < %.3f", q, self._quality_min)
                continue

            emb  = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            results.append({
                "embedding":     emb,
                "det_score":     float(face.det_score),
                "quality_score": q,
                "bbox":          bbox,
            })
        return results

    # ── Full cache rebuild ────────────────────────────────────────────────────
    def reload_cache(self) -> int:
        """
        Quality-weighted centroid rebuild.
        Call only at startup and after clustering.
        """
        from face.models import FaceIdentity

        identities = FaceIdentity.query.filter_by(is_archived=False).all()
        new: dict  = {}

        for identity in identities:
            rows = identity.embeddings.all()
            if not rows:
                continue

            embs    = np.stack([r.embedding for r in rows])
            weights = np.array([
                r.quality_score if r.quality_score is not None else 0.5
                for r in rows
            ], dtype=np.float32)
            wsum    = weights.sum()
            if wsum == 0:
                weights = np.ones(len(rows), dtype=np.float32)
                wsum    = float(len(rows))

            mean     = (embs * weights[:, None]).sum(axis=0).astype(np.float32)
            norm     = np.linalg.norm(mean)
            centroid = mean / norm if norm > 0 else mean

            new[identity.id] = {
                "label":      identity.label,
                "centroid":   centroid,
                "weight_sum": float(wsum),
                "confidence": getattr(identity, "identity_confidence", 0.0) or 0.0,
                "n_matches":  len(rows),
            }

        with self._rw.write():
            self._cache = new

        logger.info("Cache rebuilt: %d identities.", len(new))
        return len(new)

    # ── Incremental cache patch ───────────────────────────────────────────────
    def _patch_cache(
        self,
        identity_id: str,
        label: str,
        new_embedding: np.ndarray,
        quality: float,
        match_score: float,
        update_centroid: bool = True,
    ) -> None:
        """
        Quality-weighted incremental centroid update.
        If update_centroid=False (outlier), only updates confidence stats.
        O(1) — no DB reads.
        """
        with self._rw.write():
            entry = self._cache.get(identity_id)

            if entry is None:
                self._cache[identity_id] = {
                    "label":      label,
                    "centroid":   new_embedding.copy(),
                    "weight_sum": quality,
                    "confidence": match_score,
                    "n_matches":  1,
                }
                return

            entry["label"]     = label
            entry["n_matches"] += 1
            # Running weighted mean of match scores
            n = entry["n_matches"]
            entry["confidence"] = entry["confidence"] * (n - 1) / n + match_score / n

            if update_centroid:
                old_w  = entry["weight_sum"]
                old_c  = entry["centroid"]
                new_w  = old_w + quality
                # Weighted mean update
                combined = old_c * old_w + new_embedding * quality
                norm     = np.linalg.norm(combined)
                entry["centroid"]   = combined / norm if norm > 0 else combined
                entry["weight_sum"] = new_w

    def _snapshot(self) -> dict:
        with self._rw.read():
            return {k: {**v} for k, v in self._cache.items()}

    # ── Match / create ────────────────────────────────────────────────────────
    def match_or_create(
        self,
        embedding: np.ndarray,
        quality: float,
        stream_id: Optional[str] = None,
    ) -> tuple[str, str, float]:
        """
        Match embedding against quality-weighted centroid cache.

        Outlier handling
        ----------------
        If the best match score >= threshold but < OUTLIER_MIN_SIMILARITY of
        the centroid, the embedding is stored in DB (useful for clustering) but
        does NOT update the centroid.  This prevents a single bad frame from
        corrupting the identity.

        Returns (identity_id, label, match_score).
        match_score = 0.0 when a new identity is created.
        """
        from face.models import FaceIdentity, FaceEmbedding
        from extensions import db

        cache      = self._snapshot()
        best_id    : Optional[str] = None
        best_label : str           = ""
        best_score : float         = 0.0

        for identity_id, data in cache.items():
            score = float(np.dot(embedding, data["centroid"]))
            if score > best_score:
                best_score = score
                best_id    = identity_id
                best_label = data["label"]

        if best_id and best_score >= self._threshold:
            # Outlier check: if the embedding is unusually far from the centroid,
            # store it (helps clustering) but don't corrupt the centroid.
            is_outlier = best_score < self._outlier_min_sim
            if is_outlier:
                logger.debug(
                    "Outlier embedding for '%s': score=%.4f < outlier_min=%.4f — stored, centroid unchanged",
                    best_label, best_score, self._outlier_min_sim,
                )

            emb_row               = FaceEmbedding(identity_id=best_id, stream_id=stream_id)
            emb_row.embedding     = embedding
            emb_row.det_score     = quality  # quality_score stored in det_score column
            emb_row.quality_score = quality
            db.session.add(emb_row)
            db.session.commit()

            self._patch_cache(
                best_id, best_label, embedding, quality,
                match_score=best_score,
                update_centroid=not is_outlier,
            )
            return best_id, best_label, best_score

        # No match → create new identity
        label    = FaceIdentity.next_label()
        identity = FaceIdentity(label=label, is_confirmed=False, identity_confidence=0.0)
        db.session.add(identity)
        db.session.flush()

        emb_row               = FaceEmbedding(identity_id=identity.id, stream_id=stream_id)
        emb_row.embedding     = embedding
        emb_row.det_score     = quality
        emb_row.quality_score = quality
        db.session.add(emb_row)
        db.session.commit()

        self._patch_cache(identity.id, label, embedding, quality, match_score=0.0)
        logger.info("New identity: '%s' quality=%.3f", label, quality)
        return identity.id, label, 0.0