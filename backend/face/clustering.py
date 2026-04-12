"""
Identity consolidation — quality-weighted DBSCAN
=================================================

Changes from v1
---------------
1. ARCHIVE instead of delete
   Merged source identities are archived (is_archived=True, merged_into_id set).
   Audit trail is preserved.  Routes.py "delete" already archives — clustering
   now does the same.

2. QUALITY-WEIGHTED distance matrix
   Low-quality embeddings (quality_score < QUALITY_ANCHOR_MIN) are included in
   the clustering run but get their distances up-weighted — effectively pushing
   them toward the edges of clusters rather than anchoring cluster centroids.
   This prevents blurry detections from mislocating cluster centres.

3. IDENTITY CONFIDENCE update after clustering
   After merging, the canonical identity's confidence is updated from the mean
   match_score of its violation history.

4. NOISE POINT handling
   Noise points are left assigned to their current identity.
   Unassigned noise points get a new solo identity (unchanged from v1).
"""

import logging
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from extensions import db
from face.models import FaceEmbedding, FaceIdentity, Violation

logger = logging.getLogger(__name__)

QUALITY_ANCHOR_MIN = 0.55   # embeddings below this are soft-included (distance inflated)
QUALITY_INFLATE    = 1.25   # inflate distances for low-quality embeddings by this factor


def run_clustering(
    eps: float = 0.40,
    min_samples: int = 2,
    only_unconfirmed: bool = True,
) -> dict:
    """
    Consolidate fragmented identities using quality-weighted DBSCAN.

    Returns dict:
        clusters_found, identities_merged, embeddings_reassigned, noise_count
    """
    # ── Load embeddings ───────────────────────────────────────────────────────
    query = FaceEmbedding.query

    if only_unconfirmed:
        confirmed_ids = [
            i.id for i in FaceIdentity.query.filter_by(is_confirmed=True, is_archived=False).all()
        ]
        if confirmed_ids:
            query = query.filter(~FaceEmbedding.identity_id.in_(confirmed_ids))

    all_embs: list[FaceEmbedding] = query.all()

    if len(all_embs) < 2:
        logger.info("Consolidation skipped — fewer than 2 eligible embeddings.")
        return {"clusters_found": 0, "identities_merged": 0,
                "embeddings_reassigned": 0, "noise_count": 0}

    # ── Build quality-weighted distance matrix ────────────────────────────────
    X = np.stack([e.embedding for e in all_embs]).astype(np.float32)
    D = cosine_distances(X).astype(np.float64)
    np.clip(D, 0, 2, out=D)

    # Inflate distances for low-quality embeddings in both directions
    qualities = np.array([
        e.quality_score if e.quality_score is not None else 0.5
        for e in all_embs
    ], dtype=np.float64)

    low_q_mask = qualities < QUALITY_ANCHOR_MIN
    if low_q_mask.any():
        # Inflate rows AND columns for low-quality points
        D[low_q_mask, :] *= QUALITY_INFLATE
        D[:, low_q_mask] *= QUALITY_INFLATE
        np.clip(D, 0, 2, out=D)

    # ── DBSCAN ───────────────────────────────────────────────────────────────
    labels = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)\
             .fit_predict(D)

    unique_labels       = set(labels)
    clusters_found      = len(unique_labels - {-1})
    identities_merged   = 0
    embeddings_reassigned = 0

    logger.info(
        "DBSCAN: %d embeddings → %d clusters, %d noise",
        len(all_embs), clusters_found, (labels == -1).sum(),
    )

    # ── Process clusters ──────────────────────────────────────────────────────
    for cluster_id in unique_labels:
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embs    = [all_embs[i] for i in cluster_indices]

        if cluster_id == -1:
            # Noise: ensure each has an identity
            for emb in cluster_embs:
                if emb.identity_id is None:
                    label    = FaceIdentity.next_label()
                    identity = FaceIdentity(label=label, is_confirmed=False)
                    db.session.add(identity)
                    db.session.flush()
                    emb.identity_id = identity.id
                    embeddings_reassigned += 1
            continue

        identity_ids = {e.identity_id for e in cluster_embs if e.identity_id is not None}

        # Choose canonical: confirmed > oldest created_at
        canonical: Optional[FaceIdentity] = None
        for iid in identity_ids:
            candidate = FaceIdentity.query.get(iid)
            if candidate is None or candidate.is_archived:
                continue
            if canonical is None:
                canonical = candidate
            elif candidate.is_confirmed and not canonical.is_confirmed:
                canonical = candidate
            elif (not canonical.is_confirmed and not candidate.is_confirmed
                  and candidate.created_at < canonical.created_at):
                canonical = candidate

        if canonical is None:
            label     = FaceIdentity.next_label()
            canonical = FaceIdentity(label=label, is_confirmed=False)
            db.session.add(canonical)
            db.session.flush()

        # Re-assign embeddings
        for emb in cluster_embs:
            if emb.identity_id != canonical.id:
                emb.identity_id = canonical.id
                embeddings_reassigned += 1

        # Merge non-canonical identities → ARCHIVE (not delete)
        for iid in identity_ids:
            if iid == canonical.id:
                continue
            old = FaceIdentity.query.get(iid)
            if old is None or old.is_confirmed or old.is_archived:
                continue

            # Move violations and inherit thumbnail if needed
            Violation.query.filter_by(identity_id=iid).update(
                {"identity_id": canonical.id}
            )
            if old.thumbnail_filename and not canonical.thumbnail_filename:
                canonical.thumbnail_filename = old.thumbnail_filename

            # ARCHIVE the source — never hard delete
            old.is_archived    = True
            old.merged_into_id = canonical.id
            identities_merged += 1

        # Update canonical's identity_confidence from its violation history
        violation_scores = [
            v.match_score for v in canonical.violations.all()
            if v.match_score is not None
        ]
        if violation_scores:
            canonical.identity_confidence = float(np.mean(violation_scores))

    db.session.commit()
    noise_count = int((labels == -1).sum())
    logger.info(
        "Consolidation done: %d merged, %d reassigned, %d noise.",
        identities_merged, embeddings_reassigned, noise_count,
    )

    return {
        "clusters_found":        clusters_found,
        "identities_merged":     identities_merged,
        "embeddings_reassigned": embeddings_reassigned,
        "noise_count":           noise_count,
    }