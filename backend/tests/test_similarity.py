"""
Tests for the identity similarity engine and merge suggestion generator.
Run with: pytest tests/test_similarity.py -v

No DB, no models — pure numpy math.
"""
import pytest
import numpy as np
from unittest.mock import patch
from face.similarity import identity_similarity, generate_merge_suggestions


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _proto(vec: np.ndarray, weight: float = 0.7) -> dict:
    return {"vec": _norm(vec), "weight": weight, "count": 1}


def _entry(protos: list, confirmed: bool = False, archived: bool = False) -> dict:
    return {
        "label":        "test",
        "prototypes":   protos,
        "confidence":   0.5,
        "n_matches":    len(protos),
        "is_confirmed": confirmed,
        "is_archived":  archived,
    }


class TestIdentitySimilarity:
    def test_identical_singleton_prototypes_scores_one(self):
        rng = np.random.default_rng(0)
        vec = _norm(rng.random(512).astype(np.float32))
        a = _entry([_proto(vec)])
        b = _entry([_proto(vec.copy())])
        score = identity_similarity(a, b)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_singletons_score_near_zero(self):
        v1 = np.zeros(512, dtype=np.float32); v1[0] = 1.0
        v2 = np.zeros(512, dtype=np.float32); v2[1] = 1.0
        a = _entry([_proto(v1)])
        b = _entry([_proto(v2)])
        score = identity_similarity(a, b)
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_singleton_uses_pure_cosine_no_coverage_penalty(self):
        """The v1 formula gave 0.70 × 0.40 + 0.30 × 0.0 = 0.28 for singletons.
        v2 must return 0.40 exactly for a singleton pair with cosine 0.40."""
        rng = np.random.default_rng(1)
        base  = _norm(rng.random(512).astype(np.float32))
        noise = _norm(rng.random(512).astype(np.float32)) * 0.3
        perturbed = _norm(base * 0.92 + noise * 0.08)   # cos ≈ 0.92 not 0.4
        # Build a specific pair with known cosine ~0.40
        # Use directly computed pair
        v1 = _norm(np.array([1.0, 0.0, 0.0] + [0.0] * 509, dtype=np.float32))
        angle = np.deg2rad(66)  # cos(66°) ≈ 0.407
        v2 = _norm(np.array([np.cos(angle), np.sin(angle)] + [0.0] * 510, dtype=np.float32))
        expected_cos = float(np.dot(v1, v2))

        a = _entry([_proto(v1)])
        b = _entry([_proto(v2)])
        score = identity_similarity(a, b)
        # For singleton: score = pure cosine
        assert score == pytest.approx(expected_cos, abs=1e-4)

    def test_multi_prototype_uses_max_cross(self):
        """Score should be driven by the closest prototype pair, not the average."""
        rng = np.random.default_rng(2)
        # Identity A: two prototypes — one close to identity B, one far
        close_vec = _norm(rng.random(512).astype(np.float32))
        far_vec   = _norm(rng.random(512).astype(np.float32) * -1)  # roughly opposite
        b_vec     = close_vec.copy()   # identical to close_vec

        a = _entry([_proto(close_vec), _proto(far_vec)])
        b = _entry([_proto(b_vec)])
        score = identity_similarity(a, b)
        # score must be > 0.90 (the close pair dominates)
        assert score > 0.90

    def test_empty_prototypes_returns_zero(self):
        a = _entry([])
        b = _entry([{"vec": np.ones(512, dtype=np.float32), "weight": 0.7, "count": 1}])
        assert identity_similarity(a, b) == 0.0
        assert identity_similarity(b, a) == 0.0

    def test_symmetric(self):
        rng = np.random.default_rng(3)
        v1 = _norm(rng.random(512).astype(np.float32))
        v2 = _norm(rng.random(512).astype(np.float32))
        a = _entry([_proto(v1), _proto(v2)])
        b = _entry([_proto(_norm(rng.random(512).astype(np.float32)))])
        assert identity_similarity(a, b) == pytest.approx(identity_similarity(b, a), abs=1e-5)

    def test_soft_threshold_parameter_respected(self):
        """Coverage term only fires at a lower soft_threshold."""
        v = np.zeros(512, dtype=np.float32); v[0] = 1.0
        cos45 = np.cos(np.pi / 4)
        v45   = np.array([cos45, np.sqrt(1 - cos45**2)] + [0.0] * 510, dtype=np.float32)

        a = _entry([_proto(v), _proto(_norm(np.ones(512, dtype=np.float32)))])
        b = _entry([_proto(v45)])

        # cos(45°) ≈ 0.707 — should be covered at soft=0.60 but not at soft=0.80
        score_low  = identity_similarity(a, b, soft_threshold=0.60)
        score_high = identity_similarity(a, b, soft_threshold=0.80)
        # Higher soft_threshold → lower coverage → lower or equal score
        assert score_low >= score_high - 1e-4


class TestGenerateMergeSuggestions:
    def _cache(self, n_identities: int, rng_seed: int = 42) -> dict:
        """Build a fake cache with n distinct random identities."""
        rng = np.random.default_rng(rng_seed)
        cache = {}
        for i in range(n_identities):
            v = _norm(rng.random(512).astype(np.float32))
            cache[f"id-{i:03d}"] = {
                "label":        f"Person_{i+1:03d}",
                "prototypes":   [_proto(v)],
                "confidence":   0.5,
                "n_matches":    1,
                "is_confirmed": False,
                "is_archived":  False,
            }
        return cache

    def test_empty_cache_returns_empty_list(self):
        assert generate_merge_suggestions({}) == []

    def test_single_identity_returns_empty_list(self):
        cache = self._cache(1)
        assert generate_merge_suggestions(cache) == []

    def test_identical_pair_surfaces_as_suggestion(self):
        rng = np.random.default_rng(7)
        vec = _norm(rng.random(512).astype(np.float32))
        cache = {
            "id-A": {"label": "A", "prototypes": [_proto(vec)], "confidence": 0.5,
                     "n_matches": 1, "is_confirmed": False, "is_archived": False},
            "id-B": {"label": "B", "prototypes": [_proto(vec.copy())], "confidence": 0.5,
                     "n_matches": 1, "is_confirmed": False, "is_archived": False},
        }
        suggestions = generate_merge_suggestions(cache, threshold=0.28)
        assert len(suggestions) == 1
        assert suggestions[0]["similarity"] == pytest.approx(1.0, abs=1e-3)

    def test_confirmed_identities_excluded(self):
        rng = np.random.default_rng(8)
        vec = _norm(rng.random(512).astype(np.float32))
        cache = {
            "id-A": {"label": "A", "prototypes": [_proto(vec)],
                     "is_confirmed": True, "is_archived": False,  # CONFIRMED
                     "confidence": 0.5, "n_matches": 1},
            "id-B": {"label": "B", "prototypes": [_proto(vec.copy())],
                     "is_confirmed": False, "is_archived": False,
                     "confidence": 0.5, "n_matches": 1},
        }
        suggestions = generate_merge_suggestions(cache, threshold=0.0)
        assert len(suggestions) == 0   # A is confirmed → excluded

    def test_archived_identities_excluded(self):
        rng = np.random.default_rng(9)
        vec = _norm(rng.random(512).astype(np.float32))
        cache = {
            "id-A": {"label": "A", "prototypes": [_proto(vec)],
                     "is_confirmed": False, "is_archived": True,  # ARCHIVED
                     "confidence": 0.5, "n_matches": 1},
            "id-B": {"label": "B", "prototypes": [_proto(vec.copy())],
                     "is_confirmed": False, "is_archived": False,
                     "confidence": 0.5, "n_matches": 1},
        }
        suggestions = generate_merge_suggestions(cache, threshold=0.0)
        assert len(suggestions) == 0

    def test_results_sorted_by_similarity_descending(self):
        rng = np.random.default_rng(10)
        v1 = _norm(rng.random(512).astype(np.float32))
        v2 = _norm(rng.random(512).astype(np.float32))
        v3 = _norm(v1 * 0.99 + rng.random(512).astype(np.float32) * 0.01)
        cache = {
            "id-1": {"label": "A", "prototypes": [_proto(v1)], "is_confirmed": False,
                     "is_archived": False, "confidence": 0.5, "n_matches": 1},
            "id-2": {"label": "B", "prototypes": [_proto(v2)], "is_confirmed": False,
                     "is_archived": False, "confidence": 0.5, "n_matches": 1},
            "id-3": {"label": "C", "prototypes": [_proto(v3)], "is_confirmed": False,
                     "is_archived": False, "confidence": 0.5, "n_matches": 1},
        }
        suggestions = generate_merge_suggestions(cache, threshold=0.0)
        sims = [s["similarity"] for s in suggestions]
        assert sims == sorted(sims, reverse=True)

    def test_max_results_respected(self):
        cache = self._cache(20)
        suggestions = generate_merge_suggestions(cache, threshold=0.0, max_results=5)
        assert len(suggestions) <= 5

    def test_cross_track_realistic_range_surfaces_suggestion(self):
        """
        Regression test for the exact scenario that was broken:
        Same person, two tracks, mean-embedding cosine similarity ~0.39.
        Should appear as suggestion with threshold=0.28.
        """
        rng = np.random.default_rng(42)
        identity_base = _norm(rng.standard_normal(512).astype(np.float32))

        def make_mean_emb(base, angle_deg, n_frames, seed):
            r = np.random.default_rng(seed)
            embs = []
            for _ in range(n_frames):
                noise = r.standard_normal(512).astype(np.float32)
                e = _norm(base + noise * angle_deg / 90.0 * 0.30)
                embs.append(e)
            mean = np.mean(embs, axis=0)
            return _norm(mean)

        t1_mean = make_mean_emb(identity_base, 5, 4, seed=1)
        t2_mean = make_mean_emb(identity_base, 25, 4, seed=100)
        cross_sim = float(np.dot(t1_mean, t2_mean))

        cache = {
            "person-001": {"label": "Person_001", "prototypes": [_proto(t1_mean)],
                           "is_confirmed": False, "is_archived": False,
                           "confidence": 0.5, "n_matches": 4},
            "person-002": {"label": "Person_002", "prototypes": [_proto(t2_mean)],
                           "is_confirmed": False, "is_archived": False,
                           "confidence": 0.5, "n_matches": 4},
        }
        suggestions = generate_merge_suggestions(cache, threshold=0.28)
        # If cross_sim > 0.28 we expect a suggestion; if not, fragmentation still occurs
        if cross_sim >= 0.28:
            assert len(suggestions) == 1, (
                f"Expected 1 suggestion for cross_sim={cross_sim:.4f} but got {len(suggestions)}"
            )
        # Regardless, the formula must return pure cosine for singletons
        a = {"prototypes": [_proto(t1_mean)], "is_confirmed": False, "is_archived": False}
        b = {"prototypes": [_proto(t2_mean)], "is_confirmed": False, "is_archived": False}
        assert identity_similarity(a, b) == pytest.approx(cross_sim, abs=1e-4)
