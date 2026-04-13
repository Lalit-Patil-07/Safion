"""
Identity similarity engine — v2
================================
Computes pairwise identity similarity for the merge suggestion system.
Operates on the in-memory cache — no DB reads.

Why the v1 formula was broken for cross-track singleton identities
-----------------------------------------------------------------
v1 formula: 0.70 × max_cross + 0.30 × covered_at_soft=0.60

When each identity has exactly 1 prototype (freshly created from a single
confirmed track), and the cross-track cosine similarity is 0.38–0.45:

  covered_at_0.60 = (0.40 >= 0.60) = 0.0
  final = 0.70 × 0.40 + 0.30 × 0.0 = 0.28

The 0.30 × 0.0 term actively suppresses the score for the exact case that
needs detection.  With SUGGESTION_THRESHOLD=0.50, no cross-track same-person
pair could ever surface as a suggestion.

v2 fix
------
For singleton galleries (min(|A|, |B|) == 1):
  → pure cosine similarity.  No coverage penalty.

For larger galleries:
  → adaptive coverage weight that scales from 0 (size=2) to 0.25 (size=5+).
  → soft_threshold lowered from 0.60 to SIMILARITY_SOFT_THRESHOLD (default 0.35).

Correct threshold relationship
-------------------------------
IDENTITY_MATCH_THRESHOLD  ← auto-assign to existing identity
SUGGESTION_THRESHOLD       ← show as merge suggestion for operator review

MUST hold: SUGGESTION_THRESHOLD < IDENTITY_MATCH_THRESHOLD

The suggestion system exists to catch pairs that failed auto-matching.
If SUGGESTION_THRESHOLD >= IDENTITY_MATCH_THRESHOLD, suggestions only fire
for pairs that already matched — which is useless.

Time complexity: O(N² × K²) — ~50ms for 500 identities.
"""

from __future__ import annotations
import numpy as np


def identity_similarity(
    entry_a: dict,
    entry_b: dict,
    soft_threshold: float = 0.35,
) -> float:
    """
    Compute similarity score [0, 1] between two identity cache entries.

    Parameters
    ----------
    entry_a, entry_b : cache entries with "prototypes" list of {"vec": ndarray}
    soft_threshold   : similarity above which a prototype-pair counts as "covered"
                       (lowered from 0.60 to 0.35 to match surveillance video range)

    For singleton galleries (K=1 on either side): returns pure cosine similarity.
    For larger galleries: blends max_cross + adaptive coverage weight.
    """
    protos_a = entry_a.get("prototypes", [])
    protos_b = entry_b.get("prototypes", [])

    if not protos_a or not protos_b:
        return 0.0

    vecs_a = np.stack([p["vec"] for p in protos_a])  # (|A|, 512)
    vecs_b = np.stack([p["vec"] for p in protos_b])  # (|B|, 512)

    # Cross-similarity matrix (|A| × |B|)
    sim_matrix = vecs_a @ vecs_b.T

    # For each prototype in A: best match score in B
    best_per_a = sim_matrix.max(axis=1)   # (|A|,)
    max_cross  = float(np.mean(best_per_a))

    min_size = min(len(protos_a), len(protos_b))

    if min_size <= 1:
        # Singleton case: pure cosine, no coverage penalty.
        # Two singletons scoring 0.38 should surface as a suggestion.
        return float(max_cross)

    # Multi-prototype case: adaptive coverage weight.
    # Scales from 0.05 (size=2) to 0.25 (size=5+) so small galleries
    # still weight primarily on max_cross.
    cov_weight = min(0.25, 0.05 * (min_size - 1))
    covered    = float(np.mean(best_per_a >= soft_threshold))

    return float((1.0 - cov_weight) * max_cross + cov_weight * covered)


def generate_merge_suggestions(
    cache_snapshot: dict,
    threshold: float = 0.28,
    max_results: int = 30,
    soft_threshold: float = 0.35,
) -> list[dict]:
    """
    Compare all pairs of active (non-confirmed, non-archived) identities
    and return those above `threshold`, ranked by similarity descending.

    Parameters
    ----------
    cache_snapshot  : from InsightFacePipeline._snapshot()
    threshold       : minimum score to surface (default 0.28, below auto-match 0.35)
    max_results     : cap on results
    soft_threshold  : passed through to identity_similarity()

    Returns
    -------
    List of {"identity_a_id", "identity_a_label", "identity_b_id",
             "identity_b_label", "similarity"} sorted by similarity DESC.
    """
    # Only compare non-confirmed, non-archived identities with at least 1 prototype
    active = [
        (iid, data)
        for iid, data in cache_snapshot.items()
        if not data.get("is_confirmed", False)
        and not data.get("is_archived", False)
        and data.get("prototypes")
    ]

    suggestions: list[dict] = []

    for i, (id_a, data_a) in enumerate(active):
        for id_b, data_b in active[i + 1:]:
            sim = identity_similarity(data_a, data_b, soft_threshold=soft_threshold)
            if sim >= threshold:
                suggestions.append({
                    "identity_a_id":    id_a,
                    "identity_a_label": data_a["label"],
                    "identity_b_id":    id_b,
                    "identity_b_label": data_b["label"],
                    "similarity":       round(sim, 3),
                })

    suggestions.sort(key=lambda x: x["similarity"], reverse=True)
    return suggestions[:max_results]