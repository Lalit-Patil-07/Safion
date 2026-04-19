"""
InsightFacePipeline  — v3  (multi-prototype)
=============================================
Model  : buffalo_l (InsightFace)
Detect : RetinaFace
Embed  : ArcFace 512-dim L2-normalised

Changes from v2 (single centroid)
-----------------------------------
MULTI-PROTOTYPE CACHE
  Each identity now stores up to MAX_PROTOTYPES representative vectors instead
  of a single weighted centroid.  Prototypes capture distinct appearance modes
  (frontal / profile, with helmet / without, day / night lighting).

  Prototype update rules (incremental, O(K) per violation):
    1. Compute similarity to every existing prototype.
    2. If best_sim >= PROTO_MERGE_THRESHOLD (same viewing condition):
         weighted-average-merge into that prototype.
    3. Else if len(prototypes) < MAX_PROTOTYPES:
         add as a new prototype.
    4. Else (at capacity, genuinely novel view):
         replace the weakest prototype only if new quality is higher.

  reload_cache() rebuilds prototypes from stored embeddings using the same
  greedy algorithm, seeded from highest-quality embeddings first.

MATCHING
  match_or_create() scores each identity by the MAX similarity across all its
  prototypes.  One strong match against any prototype is sufficient.
  Outlier check and centroid-update guard are preserved.

SIMILARITY ENGINE SUPPORT
  _snapshot() now includes "is_confirmed" and "is_archived" flags so the
  similarity engine (face/similarity.py) can filter without a DB round-trip.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import time
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


# ── Quality scoring (unchanged from v2) ──────────────────────────────────────
def compute_quality_score(det_score: float, bbox: list, image_shape: tuple) -> float:
    """
    Combined embedding quality: det_score × 0.7 + face_area_ratio × 0.3
    Range [0, 1].
    """
    if not bbox or len(bbox) < 4 or not image_shape:
        return float(det_score)

    h, w = image_shape[:2]
    bx1, by1, bx2, by2 = bbox[:4]
    face_area  = max((bx2 - bx1) * (by2 - by1), 0)
    area_ratio = min(face_area / max(h * w, 1), 1.0)
    area_score = min(area_ratio / 0.15, 1.0)
    return float(np.clip(det_score * 0.7 + area_score * 0.3, 0.0, 1.0))


# ── Prototype helpers ─────────────────────────────────────────────────────────
def _build_prototypes(
    rows,
    max_k: int,
    merge_threshold: float,
) -> list[dict]:
    """
    Build up to max_k representative prototypes from FaceEmbedding rows.

    Algorithm
    ---------
    1. Sort rows by quality_score descending.
    2. For each embedding, check similarity against current prototypes.
       - If any prototype is within merge_threshold: weighted-merge.
       - Else if under capacity: add new prototype.
       - Else: skip (at capacity, this embedding doesn't add a new view).

    Returns list of {"vec": ndarray(512), "weight": float, "count": int}.
    """
    rows_sorted = sorted(rows, key=lambda r: r.quality_score or 0.0, reverse=True)
    prototypes: list[dict] = []

    for row in rows_sorted:
        emb = row.embedding
        q   = float(row.quality_score or 0.5)

        if not prototypes:
            prototypes.append({"vec": emb, "weight": q, "count": 1})
            continue

        vecs     = np.stack([p["vec"] for p in prototypes])
        sims     = (vecs @ emb).ravel()
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= merge_threshold:
            p     = prototypes[best_idx]
            w_old = p["weight"] * p["count"]
            w_new = q
            combined = p["vec"] * w_old + emb * w_new
            norm     = np.linalg.norm(combined)
            p["vec"]    = combined / norm if norm > 0 else combined
            p["count"] += 1
            p["weight"]  = (w_old + w_new) / p["count"]
        elif len(prototypes) < max_k:
            prototypes.append({"vec": emb, "weight": q, "count": 1})
        # else: at capacity and genuinely different — skip

    return prototypes


def _proto_max_similarity(
    prototypes: list[dict],
    query: np.ndarray,
) -> float:
    """
    Return the maximum cosine similarity between query and any prototype.
    """
    if not prototypes:
        return 0.0
    vecs = np.stack([p["vec"] for p in prototypes])
    return float(np.max(vecs @ query))


def _update_prototypes(
    prototypes: list[dict],
    new_emb: np.ndarray,
    quality: float,
    max_k: int,
    merge_threshold: float,
) -> None:
    """
    In-place incremental prototype update.  Called from _patch_cache().
    """
    if not prototypes:
        prototypes.append({"vec": new_emb.copy(), "weight": quality, "count": 1})
        return

    vecs     = np.stack([p["vec"] for p in prototypes])
    sims     = (vecs @ new_emb).ravel()
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim >= merge_threshold:
        p     = prototypes[best_idx]
        w_old = p["weight"] * p["count"]
        w_new = quality
        combined = p["vec"] * w_old + new_emb * w_new
        norm     = np.linalg.norm(combined)
        p["vec"]    = combined / norm if norm > 0 else combined
        p["count"] += 1
        p["weight"]  = (w_old + w_new) / p["count"]

    elif len(prototypes) < max_k:
        prototypes.append({"vec": new_emb.copy(), "weight": quality, "count": 1})

    else:
        # At capacity — replace weakest prototype if incoming quality is higher
        weights  = [p["weight"] for p in prototypes]
        min_idx  = int(np.argmin(weights))
        if quality > prototypes[min_idx]["weight"]:
            prototypes[min_idx] = {"vec": new_emb.copy(), "weight": quality, "count": 1}


def _temporal_bonus(data: dict, recent_window: float, boost: float) -> float:
    """
    Return `boost` if this identity was seen within `recent_window` seconds,
    else 0.0.  Runs in O(1) from in-memory cache — no DB reads.
    """
    last_seen = data.get("last_seen", 0.0)
    if last_seen and (time.time() - last_seen) < recent_window:
        return boost
    return 0.0


# ── Pipeline ──────────────────────────────────────────────────────────────────
class InsightFacePipeline:
    """
    Cache entry structure (v3):
    {
        identity_id: {
            "label":        str,
            "prototypes":   [{"vec": ndarray(512), "weight": float, "count": int}, ...],
            "confidence":   float,   # running mean of match scores
            "n_matches":    int,
            "is_confirmed": bool,    # for similarity engine — no DB round-trip needed
            "is_archived":  bool,
        }
    }
    """

    def __init__(self, config) -> None:
        self._model_name             = config["INSIGHTFACE_MODEL"]
        self._threshold              = config["IDENTITY_MATCH_THRESHOLD"]
        self._det_score_min          = config["FACE_DET_SCORE_MIN"]
        self._quality_min            = config["EMBEDDING_QUALITY_MIN"]
        self._outlier_min_sim        = config["OUTLIER_MIN_SIMILARITY"]
        self._max_prototypes         = config["MAX_PROTOTYPES"]
        self._proto_merge            = config["PROTO_MERGE_THRESHOLD"]
        self._ema_alpha              = config["EMA_ALPHA"]
        self._temporal_boost         = config["TEMPORAL_BOOST"]
        self._recent_window          = config["RECENT_WINDOW"]
        self._strong_match_threshold = config["STRONG_MATCH_THRESHOLD"]
        self._rw              = _RWLock()
        self._cache: dict     = {}
        self._ready           = False

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
        Returns [{"embedding", "det_score", "quality_score", "bbox"}].
        Faces below quality_min are dropped.
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
            bbox = face.bbox.tolist() if face.bbox is not None else []
            q    = compute_quality_score(float(face.det_score), bbox, person_crop_bgr.shape)
            if q < self._quality_min:
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
        Rebuild multi-prototype cache from DB.  Call at startup and post-clustering.
        """
        from face.models import FaceIdentity

        identities = FaceIdentity.query.filter_by(is_archived=False).all()
        new: dict  = {}

        for identity in identities:
            rows = identity.embeddings.all()
            if not rows:
                continue
            prototypes = _build_prototypes(rows, self._max_prototypes, self._proto_merge)
            if not prototypes:
                continue
            new[identity.id] = {
                "label":        identity.label,
                "prototypes":   prototypes,
                "confidence":   identity.identity_confidence or 0.0,
                "n_matches":    len(rows),
                "is_confirmed": identity.is_confirmed,
                "is_archived":  False,   # filtered above
                "last_seen":    identity.last_seen.timestamp() if identity.last_seen else 0.0,
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
        update_prototypes: bool = True,
        is_confirmed: bool = False,
    ) -> None:
        """
        Incremental prototype update.  O(K) — no DB reads.
        If update_prototypes=False (outlier), only confidence/n_matches are updated.
        """
        with self._rw.write():
            entry = self._cache.get(identity_id)

            if entry is None:
                self._cache[identity_id] = {
                    "label":        label,
                    "prototypes":   [{"vec": new_embedding.copy(), "weight": quality, "count": 1}],
                    "confidence":   match_score,
                    "n_matches":    1,
                    "is_confirmed": is_confirmed,
                    "is_archived":  False,
                    "last_seen":    time.time(),
                }
                return

            entry["label"]       = label
            entry["is_confirmed"] = is_confirmed
            entry["last_seen"]    = time.time()
            entry["n_matches"]  += 1
            if match_score >= self._strong_match_threshold:
                entry["confidence"] = (self._ema_alpha * match_score
                                       + (1 - self._ema_alpha) * entry["confidence"])

            if update_prototypes:
                _update_prototypes(
                    entry["prototypes"],
                    new_embedding, quality,
                    self._max_prototypes,
                    self._proto_merge,
                )

    # ── Track-level match + store ─────────────────────────────────────────────
    def match_and_store_track(
        self,
        embeddings: list,
        qualities: list,
        stream_id: Optional[str] = None,
    ) -> tuple[str, str, float]:
        """
        Called at track confirmation.  Two-step operation:

        MATCH using quality-weighted mean
          Mean is more stable than any individual frame — it averages out
          per-frame pose jitter and produces a more representative vector.
          Individual frames score LOWER than means (confirmed empirically).

        STORE all individual embeddings (not just the mean)
          Each frame is a separate DB row, allowing _build_prototypes() to
          reconstruct the full appearance diversity for future matching and
          clustering.  This also gives DBSCAN more data points to work with.

        Returns (identity_id, label, match_score).
        """
        from face.models import FaceIdentity, FaceEmbedding
        from extensions import db

        if not embeddings or not qualities:
            return self.match_or_create(embeddings[0] if embeddings else None,
                                        qualities[0] if qualities else 0.0,
                                        stream_id=stream_id)

        # ── Step 1: compute quality-weighted mean for matching ────────────────
        emb_matrix = np.stack(embeddings).astype(np.float32)
        w          = np.array(qualities, dtype=np.float32)
        w_sum      = w.sum()
        if w_sum > 0:
            mean_emb = (emb_matrix * (w / w_sum)[:, None]).sum(axis=0)
        else:
            mean_emb = emb_matrix.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        best_quality = float(max(qualities))

        # ── Step 2: match mean against cache ─────────────────────────────────
        cache      = self._snapshot()
        best_id    : Optional[str] = None
        best_label : str           = ""
        best_score : float         = 0.0

        for identity_id, data in cache.items():
            score = (_proto_max_similarity(data["prototypes"], mean_emb)
                     + _temporal_bonus(data, self._recent_window, self._temporal_boost))
            if score > best_score:
                best_score = score
                best_id    = identity_id
                best_label = data["label"]

        matched = best_id and best_score >= self._threshold

        if not matched:
            # Create new identity — atomic label generation + flush
            identity   = FaceIdentity.create_auto()
            best_label = identity.label
            best_id    = identity.id
            best_score = 0.0
            logger.info("New identity from track: '%s' (mean_quality=%.3f, n_embs=%d)",
                        best_label, best_quality, len(embeddings))
        else:
            logger.debug("Track matched '%s' (mean_score=%.4f, n_embs=%d)",
                         best_label, best_score, len(embeddings))

        # ── Step 3: store ALL individual embeddings ───────────────────────────
        # Filter: keep those above quality_min to avoid polluting the gallery
        quality_threshold = self._quality_min
        stored = 0
        for emb, q in zip(embeddings, qualities):
            if q < quality_threshold:
                continue
            row               = FaceEmbedding(identity_id=best_id, stream_id=stream_id)
            row.embedding     = emb
            row.det_score     = q
            row.quality_score = q
            db.session.add(row)
            stored += 1

        # Fallback: if all frames were below quality, store the best one
        if stored == 0:
            best_idx      = int(np.argmax(qualities))
            row           = FaceEmbedding(identity_id=best_id, stream_id=stream_id)
            row.embedding = embeddings[best_idx]
            row.det_score = qualities[best_idx]
            row.quality_score = qualities[best_idx]
            db.session.add(row)
            stored = 1

        db.session.commit()

        # ── Step 4: patch cache using mean embedding ──────────────────────────
        is_outlier = matched and best_score < self._outlier_min_sim
        self._patch_cache(
            best_id, best_label, mean_emb, best_quality,
            match_score=best_score,
            update_prototypes=(not is_outlier) and (best_score >= self._strong_match_threshold),
            is_confirmed=cache.get(best_id, {}).get("is_confirmed", False) if matched else False,
        )

        logger.debug("match_and_store_track: identity='%s' score=%.4f stored=%d embs",
                     best_label, best_score, stored)
        return best_id, best_label, best_score

    # ── Match / create ────────────────────────────────────────────────────────
    def match_or_create(
        self,
        embedding: np.ndarray,
        quality: float,
        stream_id: Optional[str] = None,
    ) -> tuple[str, str, float]:
        """
        Match embedding against multi-prototype cache.

        Scoring: best_score = max cosine similarity across all prototypes of each identity.
        Outlier guard: if match >= threshold but < outlier_min_sim, store embedding
        in DB (helps clustering) but do NOT update prototypes.

        Returns (identity_id, label, match_score).
        score = 0.0 on new identity creation.
        """
        from face.models import FaceIdentity, FaceEmbedding
        from extensions import db

        cache      = self._snapshot()
        best_id    : Optional[str] = None
        best_label : str           = ""
        best_score : float         = 0.0

        for identity_id, data in cache.items():
            score = (_proto_max_similarity(data["prototypes"], embedding)
                     + _temporal_bonus(data, self._recent_window, self._temporal_boost))
            if score > best_score:
                best_score = score
                best_id    = identity_id
                best_label = data["label"]

        if best_id and best_score >= self._threshold:
            is_outlier = best_score < self._outlier_min_sim
            if is_outlier:
                logger.debug(
                    "Outlier for '%s': score=%.4f < outlier_min=%.4f — stored, prototypes unchanged",
                    best_label, best_score, self._outlier_min_sim,
                )

            emb_row               = FaceEmbedding(identity_id=best_id, stream_id=stream_id)
            emb_row.embedding     = embedding
            emb_row.det_score     = quality
            emb_row.quality_score = quality
            db.session.add(emb_row)
            db.session.commit()

            self._patch_cache(
                best_id, best_label, embedding, quality,
                match_score=best_score,
                update_prototypes=(not is_outlier) and (best_score >= self._strong_match_threshold),
                is_confirmed=cache[best_id].get("is_confirmed", False),
            )
            logger.debug("Matched '%s' score=%.4f (prototypes=%d)",
                         best_label, best_score, len(cache[best_id]["prototypes"]))
            return best_id, best_label, best_score

        # No match → create new identity (atomic label generation + flush)
        identity = FaceIdentity.create_auto()
        label    = identity.label

        emb_row               = FaceEmbedding(identity_id=identity.id, stream_id=stream_id)
        emb_row.embedding     = embedding
        emb_row.det_score     = quality
        emb_row.quality_score = quality
        db.session.add(emb_row)
        db.session.commit()

        self._patch_cache(identity.id, label, embedding, quality, match_score=0.0)
        logger.info("New identity: '%s' quality=%.3f", label, quality)
        return identity.id, label, 0.0


    def _snapshot(self) -> dict:
        """Thread-safe shallow copy.  Prototype lists are shared read-only."""
        with self._rw.read():
            return {k: {**v} for k, v in self._cache.items()}