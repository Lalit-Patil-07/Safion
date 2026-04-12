"""
Identity similarity engine
==========================
Computes pairwise similarity between identities and generates ranked
merge suggestions.  Operates entirely on the in-memory cache — no DB reads.

Design
------
Identity similarity uses three signals:

1. MAX cross-prototype similarity
   For each prototype in identity A, find its best match in identity B.
   Score = mean of top-K per-A matches.
   Captures "at least one strong match" between the two galleries.

2. OVERLAP fraction
   Fraction of A's prototypes that have a match ≥ a soft threshold (0.60) in B.
   Captures "how much of A's variation is covered by B".

3. SYMMETRIC blend
   Final score = 0.70 × max_cross + 0.30 × overlap_fraction
   Weights tuned so a single coincidental match between prototypes of
   different people (common when one has a very small prototype set) doesn't
   trigger a spurious suggestion.

Time complexity
---------------
O(N² × K²) where N = number of active identities and K = MAX_PROTOTYPES.
With N=500, K=5: 500×499/2 = 124,750 pairs × 25 dot products ≈ 3.1M ops.
In numpy this runs in <50ms.  Safe for a synchronous API call.

No DB writes.  Suggestions are generated on demand and never persisted —
they're always fresh.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def identity_similarity(
    entry_a: dict,
    entry_b: dict,
    soft_threshold: float = 0.60,
) -> float:
    """
    Compute similarity score [0, 1] between two identity cache entries.

    Parameters
    ----------
    entry_a, entry_b : cache entries from InsightFacePipeline._snapshot()
                       must have "prototypes" key with list of {"vec": ndarray, ...}
    soft_threshold   : minimum similarity for a prototype-pair to count as "overlap"

    Returns
    -------
    float in [0, 1].  Values ≥ 0.68 (SUGGESTION_THRESHOLD default) are surfaced
    to operators as merge suggestions.
    """
    protos_a = entry_a.get("prototypes", [])
    protos_b = entry_b.get("prototypes", [])

    if not protos_a or not protos_b:
        return 0.0

    vecs_a = np.stack([p["vec"] for p in protos_a])   # (|A|, 512)
    vecs_b = np.stack([p["vec"] for p in protos_b])   # (|B|, 512)

    # Cross-similarity matrix: (|A|, |B|)
    sim_matrix = vecs_a @ vecs_b.T

    # Signal 1: for each prototype in A, best match in B — then average
    best_per_a  = sim_matrix.max(axis=1)         # (|A|,)
    max_cross   = float(np.mean(best_per_a))

    # Signal 2: fraction of A's prototypes with ANY match ≥ soft_threshold in B
    covered     = float(np.mean(best_per_a >= soft_threshold))

    return float(0.70 * max_cross + 0.30 * covered)


def generate_merge_suggestions(
    cache_snapshot: dict,
    threshold: float = 0.68,
    max_results: int = 30,
) -> list[dict]:
    """
    Compare all pairs of active (non-confirmed, non-archived) identities and
    return those above `threshold`, ranked by similarity descending.

    Parameters
    ----------
    cache_snapshot : dict from InsightFacePipeline._snapshot()
    threshold      : minimum similarity score to include in results
    max_results    : cap on returned suggestions

    Returns
    -------
    List of dicts:
        {
            "identity_a_id":  str,
            "identity_a_label": str,
            "identity_b_id":  str,
            "identity_b_label": str,
            "similarity":     float,   # 0–1, two decimal places
        }
    Sorted by similarity DESC.
    """
    # Only compare unconfirmed, non-archived identities with at least 1 prototype
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
            sim = identity_similarity(data_a, data_b)
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
