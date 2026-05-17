"""
PPE violation association
=========================
Determines whether a violation detection (e.g. NO-Hardhat) belongs to
a specific person detection, using spatial geometry.

Rules (in priority order):
1. Overlap — if the violation bbox overlaps the person bbox, they are associated.
2. Proximity above — if the violation center-x falls within the person x-range
   AND the violation is above the person AND the vertical gap is small
   relative to the person's height, they are associated.
   This handles helmets floating just above the person bbox top edge.
"""


def check_association(
    person_bbox: list[float],
    violation_bbox: list[float],
    proximity_ratio: float = 0.5,
) -> bool:
    """
    Return True if `violation_bbox` is associated with `person_bbox`.

    Parameters
    ----------
    person_bbox     : [px1, py1, px2, py2]
    violation_bbox  : [vx1, vy1, vx2, vy2]
    proximity_ratio : fraction of the person's height used as the vertical
                      proximity window for the "floating above" check
    """
    px1, py1, px2, py2 = person_bbox
    vx1, vy1, vx2, vy2 = violation_bbox

    person_height = py2 - py1

    # ── Rule 1: direct overlap ────────────────────────────────────────────────
    x_overlap_left  = max(px1, vx1)
    x_overlap_right = min(px2, vx2)
    y_overlap_top   = max(py1, vy1)
    y_overlap_bot   = min(py2, vy2)

    if x_overlap_right > x_overlap_left and y_overlap_bot > y_overlap_top:
        return True

    # ── Rule 2: violation is above and horizontally within the person ─────────
    v_center_x = (vx1 + vx2) / 2.0
    is_horizontally_aligned = px1 <= v_center_x <= px2
    is_above_person = vy2 <= py1
    vertical_gap = py1 - vy2
    is_close_vertically = vertical_gap <= person_height * proximity_ratio

    return is_horizontally_aligned and is_above_person and is_close_vertically


def split_detections(
    detections: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Partition a list of detection dicts into:
        (person_detections, violation_detections)
    """
    persons: list[dict] = []
    violations: list[dict] = []

    for det in detections:
        if det["class_name"] == "Person":
            persons.append(det)
        elif not det["safe"]:
            violations.append(det)

    return persons, violations
