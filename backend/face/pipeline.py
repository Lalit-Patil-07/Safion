"""
InsightFacePipeline
===================
Model: buffalo_l (InsightFace)
  Detection : RetinaFace
  Embedding : ArcFace — 512-dim, L2-normalised

Key design choices in this version
------------------------------------
INCREMENTAL cache updates (no full DB reload per violation)
  match_or_create() patches only the affected identity's centroid in memory.
  Full reload only happens at startup and after clustering.
  This reduces per-violation DB reads from O(N_identities × M_embeddings) → O(1).

Stable centroid via running mean
  Cache stores (centroid, count). When a new embedding is added:
      new_c = normalise(old_c * n + new_emb)
      new_n = n + 1
  No DB read required for the update.

Threshold at 0.55 (was 0.45)
  ArcFace on surveillance PPE footage:
      clean frontal:    0.75 – 0.92
      occluded/angled:  0.45 – 0.65
      different people: 0.05 – 0.30
  0.45 sits in the middle of the occluded-same-person band → too many misses.
  0.55 is the lower edge of the "reliably same person" zone.
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


# ── Readers-writer lock ───────────────────────────────────────────────────────
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


# ── Pipeline ──────────────────────────────────────────────────────────────────
class InsightFacePipeline:
    """
    Cache structure:
        {
          identity_id: {
            "label":    str,
            "centroid": np.ndarray (512,),  # running normalised mean
            "count":    int,                 # number of embeddings averaged in
          }
        }
    """

    def __init__(self, config) -> None:
        self._model_name    = getattr(config, "INSIGHTFACE_MODEL",       "buffalo_l")
        self._threshold     = getattr(config, "IDENTITY_MATCH_THRESHOLD", 0.55)
        self._det_score_min = getattr(config, "FACE_DET_SCORE_MIN",       0.60)
        self._rw            = _RWLock()
        self._cache: dict   = {}
        self._ready         = False

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
        Returns [{"embedding": ndarray(512), "det_score": float, "bbox": list}]
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
            emb = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            results.append({
                "embedding": emb,
                "det_score": float(face.det_score),
                "bbox":      face.bbox.tolist() if face.bbox is not None else [],
            })
        return results

    # ── Cache (full rebuild — startup + post-clustering only) ─────────────────
    def reload_cache(self) -> int:
        """
        Full DB rebuild. Expensive. Call ONLY on startup and after clustering.
        NOT called per violation.
        """
        from face.models import FaceIdentity

        identities = FaceIdentity.query.all()
        new: dict = {}
        for identity in identities:
            embs = [e.embedding for e in identity.embeddings.all()]
            if not embs:
                continue
            mean = np.mean(embs, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean)
            centroid = mean / norm if norm > 0 else mean
            new[identity.id] = {
                "label":    identity.label,
                "centroid": centroid,
                "count":    len(embs),
            }

        with self._rw.write():
            self._cache = new

        logger.info("Cache rebuilt: %d identities.", len(new))
        return len(new)

    # ── Cache (incremental — called per violation) ────────────────────────────
    def _patch_cache(self, identity_id: str, label: str, new_embedding: np.ndarray) -> None:
        """
        Update one identity's centroid in-place using a running mean.
        O(1) — no DB reads. Called from match_or_create().
        """
        with self._rw.write():
            entry = self._cache.get(identity_id)
            if entry is None:
                # New identity — seed the cache entry
                self._cache[identity_id] = {
                    "label":    label,
                    "centroid": new_embedding.copy(),
                    "count":    1,
                }
                return

            n       = entry["count"]
            old_c   = entry["centroid"]
            # Running mean: new_c = normalise(old_c * n + new_emb)
            combined = old_c * n + new_embedding
            norm     = np.linalg.norm(combined)
            entry["centroid"] = combined / norm if norm > 0 else combined
            entry["count"]    = n + 1
            entry["label"]    = label   # in case it was renamed

    def _snapshot(self) -> dict:
        with self._rw.read():
            return {k: dict(v) for k, v in self._cache.items()}

    # ── Match / create ────────────────────────────────────────────────────────
    def match_or_create(
        self,
        embedding: np.ndarray,
        stream_id: Optional[str] = None,
    ) -> tuple[str, str, float]:
        """
        Match embedding against centroid cache.
        On hit  → add embedding to existing identity (incremental cache update).
        On miss → create new identity (incremental cache update).

        Returns (identity_id, label, similarity_score).
        score = 0.0 on new identity creation.

        NO full reload_cache() is called here.
        """
        from face.models import FaceIdentity, FaceEmbedding
        from extensions import db

        cache     = self._snapshot()
        best_id   : Optional[str] = None
        best_label: str            = ""
        best_score: float          = 0.0

        for identity_id, data in cache.items():
            score = float(np.dot(embedding, data["centroid"]))
            if score > best_score:
                best_score = score
                best_id    = identity_id
                best_label = data["label"]

        if best_id and best_score >= self._threshold:
            emb_row = FaceEmbedding(identity_id=best_id, stream_id=stream_id)
            emb_row.embedding = embedding
            db.session.add(emb_row)
            db.session.commit()
            self._patch_cache(best_id, best_label, embedding)
            logger.debug("Matched '%s' score=%.4f", best_label, best_score)
            return best_id, best_label, best_score

        # No match → create new identity
        label    = FaceIdentity.next_label()
        identity = FaceIdentity(label=label, is_confirmed=False)
        db.session.add(identity)
        db.session.flush()

        emb_row = FaceEmbedding(identity_id=identity.id, stream_id=stream_id)
        emb_row.embedding = embedding
        db.session.add(emb_row)
        db.session.commit()
        self._patch_cache(identity.id, label, embedding)
        logger.info("New identity: '%s'", label)
        return identity.id, label, 0.0