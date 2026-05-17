"""
Tests for spatial violation association logic.
Run with: pytest tests/test_detection.py -v
"""
import pytest
from detection.association import check_association, split_detections


class TestCheckAssociation:
    # ── Direct overlap ────────────────────────────────────────────────────────
    def test_full_overlap_associates(self):
        person  = [100.0, 100.0, 300.0, 500.0]
        # Violation entirely inside person bbox
        violation = [150.0, 120.0, 250.0, 200.0]
        assert check_association(person, violation) is True

    def test_partial_overlap_associates(self):
        person    = [100.0, 100.0, 300.0, 500.0]
        violation = [50.0,  80.0,  200.0, 200.0]  # overlaps left edge
        assert check_association(person, violation) is True

    # ── Horizontally separated — should NOT associate ────────────────────────
    def test_no_horizontal_overlap_does_not_associate(self):
        person    = [100.0, 100.0, 300.0, 500.0]
        violation = [400.0, 100.0, 600.0, 300.0]  # completely to the right
        assert check_association(person, violation) is False

    # ── Floating above (helmet-off-head scenario) ─────────────────────────────
    def test_helmet_just_above_person_associates(self):
        person    = [100.0, 150.0, 300.0, 500.0]  # person height = 350
        # Helmet sits just above the person's head, centre-x within person x-range
        violation = [140.0,  80.0, 260.0, 145.0]  # gap = 150-145 = 5px
        assert check_association(person, violation) is True

    def test_helmet_too_far_above_does_not_associate(self):
        person    = [100.0, 300.0, 300.0, 600.0]  # person height = 300
        # Violation is 200px above (gap > 0.5 × height = 150)
        violation = [140.0,  50.0, 260.0, 100.0]  # gap = 300-100 = 200px
        assert check_association(person, violation) is False

    def test_above_but_horizontally_misaligned_does_not_associate(self):
        person    = [100.0, 200.0, 300.0, 500.0]
        # Centre-x = 350 — outside person x-range [100, 300]
        violation = [300.0, 100.0, 400.0, 195.0]
        assert check_association(person, violation) is False

    # ── Edge cases ────────────────────────────────────────────────────────────
    def test_touching_edges_no_overlap(self):
        person    = [100.0, 100.0, 300.0, 500.0]
        violation = [300.0, 100.0, 500.0, 300.0]  # x1 == person x2, no overlap
        assert check_association(person, violation) is False


# ---------------------------------------------------------------------------
# split_detections
# ---------------------------------------------------------------------------
class TestSplitDetections:
    def _det(self, class_name, safe):
        return {
            "class_name": class_name,
            "safe": safe,
            "confidence": 0.9,
            "bbox": [0.0, 0.0, 100.0, 200.0],
            "color": "#ffffff",
        }

    def test_splits_correctly(self):
        dets = [
            self._det("Person",         safe=True),
            self._det("NO-Hardhat",     safe=False),
            self._det("NO-Mask",        safe=False),
            self._det("Hardhat",        safe=True),
            self._det("Person",         safe=True),
        ]
        persons, violations = split_detections(dets)
        assert len(persons) == 2
        assert len(violations) == 2
        assert all(p["class_name"] == "Person" for p in persons)
        assert all(not v["safe"] for v in violations)

    def test_empty_input(self):
        persons, violations = split_detections([])
        assert persons == []
        assert violations == []

    def test_no_persons(self):
        dets = [self._det("NO-Hardhat", safe=False)]
        persons, violations = split_detections(dets)
        assert persons == []
        assert violations == [dets[0]]

    def test_no_violations(self):
        dets = [self._det("Person", safe=True), self._det("Hardhat", safe=True)]
        persons, violations = split_detections(dets)
        assert len(persons) == 1
        assert violations == []
