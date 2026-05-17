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


def _unit(vec: np.ndarray) -> np.ndarray:
    arr  = np.asarray(vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


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


def compute_quality_score(det_score: float, bbox: list, image_shape: tuple) -> float:
    if not bbox or len(bbox) < 4 or not image_shape:
        return float(det_score)

    h, w = image_shape[:2]
    bx1, by1, bx2, by2 = bbox[:4]
    face_area  = max((bx2 - bx1) * (by2 - by1), 0)
    area_ratio = min(face_area / max(h * w, 1), 1.0)
    area_score = min(area_ratio / 0.15, 1.0)
    return float(np.clip(det_score * 0.7 + area_score * 0.3, 0.0, 1.0))


def _build_prototypes(
    rows,
    max_k: int,
    merge_threshold: float,
) -> list[dict]:
    rows_sorted = sorted(rows, key=lambda r: r.quality_score or 0.0, reverse=True)
    prototypes: list[dict] = []

    for row in rows_sorted:
        emb = _unit(row.embedding)
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
            p["vec"]    = _unit(combined)
            p["count"] += 1
            p["weight"]  = (w_old + w_new) / p["count"]
        elif len(prototypes) < max_k:
            prototypes.append({"vec": emb, "weight": q, "count": 1})

    return prototypes


def _proto_max_similarity(
    prototypes: list[dict],
    query: np.ndarray,
    label: str = "",
    identity_id: str = "",
) -> float:
    if not prototypes:
        return 0.0

    query = np.asarray(query, dtype=np.float32).ravel()
    vecs  = np.stack([
        np.asarray(p["vec"], dtype=np.float32).ravel() for p in prototypes
    ])
    sims  = (vecs @ query).ravel()
    best  = float(np.max(sims))

    logger.debug(
        "  [proto_sim] identity=%s label=%s scores=%s max=%.4f",
        identity_id[:8] if identity_id else "?",
        label,
        [f"{s:.4f}" for s in sims.tolist()],
        best,
    )
    return best


def _update_prototypes(
    prototypes: list[dict],
    new_emb: np.ndarray,
    quality: float,
    max_k: int,
    merge_threshold: float,
) -> None:
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
        p["vec"]    = _unit(combined)
        p["count"] += 1
        p["weight"]  = (w_old + w_new) / p["count"]

    elif len(prototypes) < max_k:
        prototypes.append({"vec": new_emb.copy(), "weight": quality, "count": 1})

    else:
        weights  = [p["weight"] for p in prototypes]
        min_idx  = int(np.argmin(weights))
        if quality > prototypes[min_idx]["weight"]:
            prototypes[min_idx] = {"vec": new_emb.copy(), "weight": quality, "count": 1}


def _temporal_bonus(data: dict, recent_window: float, boost: float) -> float:
    last_seen = data.get("last_seen", 0.0)
    if last_seen and (time.time() - last_seen) < recent_window:
        return boost
    return 0.0


def _log_top_candidates(
    cache: dict,
    query: np.ndarray,
    threshold: float,
    n: int = 3,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG) or not cache:
        return

    scores = [
        (iid, data["label"],
         _proto_max_similarity(data["prototypes"], query,
                               label=data["label"], identity_id=iid),
         len(data["prototypes"]))
        for iid, data in cache.items()
    ]
    scores.sort(key=lambda x: x[2], reverse=True)

    logger.debug("  ── top-%d candidates (threshold=%.4f) ──", n, threshold)
    for rank, (iid, label, sc, n_proto) in enumerate(scores[:n], 1):
        verdict = "MATCH ✓" if sc >= threshold else "below threshold"
        logger.debug(
            "  [%d] '%s' score=%.4f n_prototypes=%d  %s",
            rank, label, sc, n_proto, verdict,
        )
    if not scores:
        logger.debug("  (cache is empty — no candidates)")


class InsightFacePipeline:

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
        self._prefer_gpu             = config["PREFER_GPU"]
        self._rw              = _RWLock()
        self._cache: dict     = {}
        self._ready           = False

    @staticmethod
    def _select_providers(prefer_gpu: bool) -> list[str]:
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            available = ["CPUExecutionProvider"]

        if prefer_gpu:
            if "CUDAExecutionProvider" in available:
                logger.info("InsightFace provider: CUDAExecutionProvider (GPU)")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.warning(
                "PREFER_GPU=true but CUDAExecutionProvider is not available. "
                "Falling back to CPUExecutionProvider."
            )

        logger.info("InsightFace provider: CPUExecutionProvider")
        return ["CPUExecutionProvider"]

    def init_app(self, app) -> None:
        global _insight_app
        try:
            try:
                import torch  # noqa: F401
                logger.info("torch %s preloaded (CUDA available: %s)",
                            torch.__version__, torch.cuda.is_available())
            except ImportError:
                logger.warning("torch not installed — CUDA DLL preloading skipped.")

            from insightface.app import FaceAnalysis
            providers = self._select_providers(self._prefer_gpu)

            try:
                import onnxruntime as _ort
                logger.info("ONNXRuntime %s — available providers: %s",
                            _ort.__version__, _ort.get_available_providers())
            except ImportError:
                logger.warning("ONNXRuntime not importable — InsightFace may fail.")

            with _insight_lock:
                if _insight_app is None:
                    logger.info("Loading InsightFace '%s' …", self._model_name)
                    _insight_app = FaceAnalysis(
                        name=self._model_name,
                        providers=providers,
                    )
                    ctx_id = 0 if self._prefer_gpu and "CUDAExecutionProvider" in providers else -1
                    _insight_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

                    active = _insight_app.models[list(_insight_app.models.keys())[0]].session.get_providers()
                    logger.info("InsightFace loaded — active session providers: %s", active)
                    if self._prefer_gpu and "CUDAExecutionProvider" not in active:
                        logger.error(
                            "PREFER_GPU=true but InsightFace session is using %s, not GPU.",
                            active,
                        )
            self._ready = True
        except Exception as exc:
            logger.error("InsightFace load failed: %s", exc)
            self._ready = False

        logger.info(
            "Pipeline config: IDENTITY_MATCH_THRESHOLD=%.4f  "
            "STRONG_MATCH_THRESHOLD=%.4f  OUTLIER_MIN_SIMILARITY=%.4f  "
            "MAX_PROTOTYPES=%d  PROTO_MERGE_THRESHOLD=%.4f  "
            "TEMPORAL_BOOST=%.4f  RECENT_WINDOW=%.1fs",
            self._threshold,
            self._strong_match_threshold,
            self._outlier_min_sim,
            self._max_prototypes,
            self._proto_merge,
            self._temporal_boost,
            self._recent_window,
        )

    @property
    def is_ready(self) -> bool:
        return self._ready and _insight_app is not None

    def embed_crop(self, person_crop_bgr: np.ndarray) -> list[dict]:
        """
        RetinaFace + ArcFace on a BGR crop.
        Input scale determines alignment quality — callers should normalize
        face height to ~250px via embed_person_crop() for runtime video frames.
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

        # Log number of detected faces before filtering
        logger.debug("embed_crop: detected %d faces", len(faces))

        results = []
        for face in faces:
            if face.det_score < self._det_score_min or face.embedding is None:
                logger.debug("embed_crop: face det_score=%.4f skipped due to low quality", face.det_score)
                continue
            bbox = face.bbox.tolist() if face.bbox is not None else []
            q    = compute_quality_score(float(face.det_score), bbox, person_crop_bgr.shape)
            if q < self._quality_min:
                logger.debug("embed_crop: face quality_score=%.4f below threshold", q)
                continue
            emb = _unit(np.asarray(face.embedding, dtype=np.float32))
            results.append({
                "embedding":     emb,
                "det_score":     float(face.det_score),
                "quality_score": q,
                "bbox":          bbox,
            })
        logger.debug("embed_crop: %d faces passed filters", len(results))
        return results

    def embed_person_crop(self, person_crop_bgr: np.ndarray) -> list[dict]:
        """
        Normalize face scale before embedding — matches importer Stage-2 exactly.

        Problem with raw embed_crop() at runtime:
          YOLO person crop size varies with camera distance.  Face height ranges
          from ~30px (distant) to ~300px (close-up).  InsightFace at det_size=640
          internally rescales the crop; variable input scale produces different
          landmark alignment quality per frame, leading to ArcFace embeddings
          that are mutually incompatible and incompatible with importer embeddings
          (observed cosine similarity: -0.01 to 0.08).

        This method replicates the importer's two-stage normalization:
          Stage 1 — detect face bbox in the person crop.
          Stage 2 — crop person-scale region (pad: 30% x, 15% top, 150% bottom),
                    resize so face height == _TARGET_FACE_PX (250px), embed.

        At 250px face height the person-scale crop is ~662px tall.
        InsightFace scales it DOWN 0.97× (lossless); face lands at ~241px in
        the 640px detection window — the P4 anchor centre for buffalo_l.
        Both runtime and importer embeddings are now generated from the same
        effective detection scale, making cosine similarity valid for matching.

        Falls back to raw embed_crop(person_crop_bgr) only when Stage-1 detects
        no face in the person crop (extreme occlusion / too-small crop).
        """
        _TARGET_FACE_PX = 250

        if not self.is_ready or person_crop_bgr is None or person_crop_bgr.size == 0:
            return []

        # ── Stage 1: detect face bbox inside the person crop ──────────────────
        img_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        try:
            with _insight_lock:
                stage1_faces = _insight_app.get(img_rgb)
        except Exception as exc:
            logger.warning("embed_person_crop Stage-1 error: %s", exc)
            return []

        # Log number of Stage-1 faces detected
        logger.debug("embed_person_crop: Stage-1 detection found %d faces", len(stage1_faces))

        if not stage1_faces:
            logger.debug(
                "embed_person_crop: no face in person crop - falling back to raw embed_crop")
            result = self.embed_crop(person_crop_bgr)
            logger.debug("embed_person_crop: fallback to raw embed_crop returned %d faces", len(result))
            return result

        # Highest det_score face — person crop should contain exactly one.
        best_face = max(stage1_faces, key=lambda f: float(f.det_score))
        x1, y1, x2, y2 = [int(v) for v in best_face.bbox]
        ph, pw = person_crop_bgr.shape[:2]

        face_h = max(1, y2 - y1)
        face_w = max(1, x2 - x1)

        # ── Stage 2: person-scale crop (same padding as importer) ─────────────
        pad_x        = max(1, int(face_w * 0.30))
        pad_y_top    = max(1, int(face_h * 0.15))
        pad_y_bottom = max(1, int(face_h * 1.50))

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y_top)
        cx2 = min(pw, x2 + pad_x)
        cy2 = min(ph, y2 + pad_y_bottom)

        normalized_crop = person_crop_bgr[cy1:cy2, cx1:cx2]
        if normalized_crop.size == 0:
            logger.debug("embed_person_crop: normalized crop empty - falling back to raw embed_crop")
            result = self.embed_crop(person_crop_bgr)
            logger.debug("embed_person_crop: fallback returned %d faces", len(result))
            return result

        # Resize so face == _TARGET_FACE_PX tall.
        scale  = _TARGET_FACE_PX / face_h
        new_h  = max(1, int(normalized_crop.shape[0] * scale))
        new_w  = max(1, int(normalized_crop.shape[1] * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        normalized_crop = cv2.resize(normalized_crop, (new_w, new_h), interpolation=interp)

        logger.debug(
            "embed_person_crop: face_h=%dpx  scale=%.3f  "
            "crop=%dx%d  face_in_det≈%dpx",
            face_h, scale, new_w, new_h,
            int(_TARGET_FACE_PX * (640.0 / new_h)),
        )

        # ── Stage 2: embed the normalized crop ──────────────────────────────────────
        stage2_results = self.embed_crop(normalized_crop)
        logger.debug("embed_person_crop: Stage-2 embedding returned %d faces", len(stage2_results))

        # If Stage-2 embedding returns no results, fallback to raw embedding
        if not stage2_results:
            logger.debug("embed_person_crop: Stage-2 embedding failed - falling back to raw embed_crop")
            result = self.embed_crop(person_crop_bgr)
            logger.debug("embed_person_crop: fallback returned %d faces", len(result))
            return result

        logger.debug("embed_person_crop: returning %d faces", len(stage2_results))
        return stage2_results

    def reload_cache(self) -> int:
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
                "is_archived":  False,
                "last_seen":    identity.last_seen.timestamp() if identity.last_seen else 0.0,
            }

        with self._rw.write():
            self._cache = new

        logger.info(
            "Cache rebuilt: %d identities loaded  "
            "(IDENTITY_MATCH_THRESHOLD=%.4f  STRONG_MATCH_THRESHOLD=%.4f)",
            len(new), self._threshold, self._strong_match_threshold,
        )
        return len(new)

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
        emb = _unit(new_embedding)

        with self._rw.write():
            entry = self._cache.get(identity_id)

            if entry is None:
                self._cache[identity_id] = {
                    "label":        label,
                    "prototypes":   [{"vec": emb.copy(), "weight": quality, "count": 1}],
                    "confidence":   match_score,
                    "n_matches":    1,
                    "is_confirmed": is_confirmed,
                    "is_archived":  False,
                    "last_seen":    time.time(),
                }
                return

            entry["label"]        = label
            entry["is_confirmed"] = is_confirmed
            entry["last_seen"]    = time.time()
            entry["n_matches"]   += 1
            if match_score >= self._strong_match_threshold:
                entry["confidence"] = (self._ema_alpha * match_score
                                       + (1 - self._ema_alpha) * entry["confidence"])

            if update_prototypes:
                _update_prototypes(
                    entry["prototypes"],
                    emb, quality,
                    self._max_prototypes,
                    self._proto_merge,
                )

    def match_and_store_track(
        self,
        embeddings: list,
        qualities: list,
        stream_id: Optional[str] = None,
    ) -> tuple[str, str, float]:
        from face.models import FaceIdentity, FaceEmbedding
        from extensions import db

        if not embeddings or not qualities:
            return self.match_or_create(
                embeddings[0] if embeddings else None,
                qualities[0] if qualities else 0.0,
                stream_id=stream_id,
            )

        normed_embs  = [_unit(e) for e in embeddings]
        best_quality = float(max(qualities))

        emb_matrix = np.stack(normed_embs).astype(np.float32)
        w          = np.array(qualities, dtype=np.float32)
        w_sum      = w.sum()
        mean_emb   = _unit(
            (emb_matrix * (w / w_sum)[:, None]).sum(axis=0)
            if w_sum > 0 else emb_matrix.mean(axis=0)
        )

        cache      = self._snapshot()
        best_id    : Optional[str] = None
        best_label : str           = ""
        best_score : float         = 0.0
        best_raw   : float         = 0.0

        for identity_id, data in cache.items():
            raw = _proto_max_similarity(
                data["prototypes"], mean_emb,
                label=data["label"], identity_id=identity_id,
            )
            score = raw + _temporal_bonus(data, self._recent_window, self._temporal_boost)
            logger.debug(
                "  [match_track] candidate='%s' raw=%.4f bonus=%.4f total=%.4f threshold=%.4f",
                data["label"], raw, score - raw, score, self._threshold,
            )
            if score > best_score:
                best_score = score
                best_raw   = raw
                best_id    = identity_id
                best_label = data["label"]

        matched = best_id and best_score >= self._threshold

        logger.debug(
            "match_and_store_track: n_embeddings=%d  best='%s'  "
            "raw_similarity=%.4f  temporal_bonus=%.4f  total_score=%.4f  "
            "threshold=%.4f  matched=%s  cache_size=%d",
            len(embeddings), best_label or "(none)",
            best_raw, best_score - best_raw, best_score,
            self._threshold, matched, len(cache),
        )
        _log_top_candidates(cache, mean_emb, self._threshold)

        if not matched:
            identity   = FaceIdentity.create_auto()
            best_label = identity.label
            best_id    = identity.id
            best_score = 0.0
            logger.info(
                "New identity from track: '%s'  (best_quality=%.3f  n_embs=%d  "
                "best_rejected_score=%.4f  threshold=%.4f)",
                best_label, best_quality, len(embeddings), best_raw, self._threshold,
            )
        else:
            logger.debug(
                "Track matched '%s'  score=%.4f  n_embs=%d",
                best_label, best_score, len(embeddings),
            )

        quality_threshold = self._quality_min
        stored = 0
        for emb, q in zip(normed_embs, qualities):
            if q < quality_threshold:
                continue
            row               = FaceEmbedding(identity_id=best_id, stream_id=stream_id)
            row.embedding     = emb
            row.det_score     = q
            row.quality_score = q
            db.session.add(row)
            stored += 1

        if stored == 0:
            best_idx      = int(np.argmax(qualities))
            row           = FaceEmbedding(identity_id=best_id, stream_id=stream_id)
            row.embedding = normed_embs[best_idx]
            row.det_score = qualities[best_idx]
            row.quality_score = qualities[best_idx]
            db.session.add(row)
            stored = 1

        db.session.commit()

        is_outlier = matched and best_raw < self._outlier_min_sim
        self._patch_cache(
            best_id, best_label, mean_emb, best_quality,
            match_score=best_score,
            update_prototypes=(not is_outlier) and (best_score >= self._strong_match_threshold),
            is_confirmed=cache.get(best_id, {}).get("is_confirmed", False) if matched else False,
        )

        logger.debug(
            "match_and_store_track done: identity='%s'  score=%.4f  stored=%d embs",
            best_label, best_score, stored,
        )
        return best_id, best_label, best_score

    def match_or_create(
        self,
        embedding: np.ndarray,
        quality: float,
        stream_id: Optional[str] = None,
    ) -> tuple[str, str, float]:
        from face.models import FaceIdentity, FaceEmbedding
        from extensions import db

        emb = _unit(np.asarray(embedding, dtype=np.float32))

        cache      = self._snapshot()
        best_id    : Optional[str] = None
        best_label : str           = ""
        best_score : float         = 0.0
        best_raw   : float         = 0.0

        for identity_id, data in cache.items():
            raw   = _proto_max_similarity(
                data["prototypes"], emb,
                label=data["label"], identity_id=identity_id,
            )
            score = raw + _temporal_bonus(data, self._recent_window, self._temporal_boost)
            logger.debug(
                "  [match_or_create] candidate='%s' raw=%.4f bonus=%.4f total=%.4f",
                data["label"], raw, score - raw, score,
            )
            if score > best_score:
                best_score = score
                best_raw   = raw
                best_id    = identity_id
                best_label = data["label"]

        logger.debug(
            "match_or_create: best='%s'  raw_similarity=%.4f  "
            "temporal_bonus=%.4f  total_score=%.4f  threshold=%.4f  cache_size=%d",
            best_label or "(none)", best_raw,
            best_score - best_raw, best_score,
            self._threshold, len(cache),
        )
        _log_top_candidates(cache, emb, self._threshold)

        if best_id and best_score >= self._threshold:
            is_outlier = best_raw < self._outlier_min_sim
            if is_outlier:
                logger.debug(
                    "Outlier for '%s': raw_score=%.4f < outlier_min=%.4f — "
                    "stored in DB, prototypes unchanged",
                    best_label, best_raw, self._outlier_min_sim,
                )

            emb_row               = FaceEmbedding(identity_id=best_id, stream_id=stream_id)
            emb_row.embedding     = emb
            emb_row.det_score     = quality
            emb_row.quality_score = quality
            db.session.add(emb_row)
            db.session.commit()

            self._patch_cache(
                best_id, best_label, emb, quality,
                match_score=best_score,
                update_prototypes=(not is_outlier) and (best_score >= self._strong_match_threshold),
                is_confirmed=cache[best_id].get("is_confirmed", False),
            )
            logger.debug(
                "Matched '%s'  score=%.4f  n_prototypes=%d",
                best_label, best_score,
                len(cache.get(best_id, {}).get("prototypes", [])),
            )
            return best_id, best_label, best_score

        logger.info(
            "No match: best_raw=%.4f  threshold=%.4f  gap=%.4f  — creating new identity",
            best_raw, self._threshold, self._threshold - best_raw,
        )

        identity = FaceIdentity.create_auto()
        label    = identity.label

        emb_row               = FaceEmbedding(identity_id=identity.id, stream_id=stream_id)
        emb_row.embedding     = emb
        emb_row.det_score     = quality
        emb_row.quality_score = quality
        db.session.add(emb_row)
        db.session.commit()

        self._patch_cache(identity.id, label, emb, quality, match_score=0.0)
        logger.info("New identity: '%s'  quality=%.3f", label, quality)
        return identity.id, label, 0.0

    def _snapshot(self) -> dict:
        with self._rw.read():
            return {k: {**v} for k, v in self._cache.items()}