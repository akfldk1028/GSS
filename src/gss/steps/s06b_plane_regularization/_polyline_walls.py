"""Module H: Merge collinear walls into multi-segment polyline walls.

When two walls share an endpoint and have similar directions, they can be
merged into a single wall with a 3+ point center-line (polyline).

This enables L-shaped walls, curved approximations, and other non-straight
wall geometries to be represented as single IFC entities.

Output: walls with wall_type="polyline" and N-point center_line_2d.
Two-point walls remain unchanged (backward compatible).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _endpoint_distance(w1: dict, w2: dict) -> tuple[float, int, int]:
    """Find the minimum distance between any pair of endpoints of two walls.

    Returns (min_distance, ep_idx_w1, ep_idx_w2) where ep_idx is 0 (start)
    or 1 (end) indicating which endpoint of each wall is closest.
    """
    cl1 = w1["center_line_2d"]
    cl2 = w2["center_line_2d"]

    best_dist = float("inf")
    best_i = 0
    best_j = 0

    for i, pi in enumerate([cl1[0], cl1[-1]]):
        for j, pj in enumerate([cl2[0], cl2[-1]]):
            d = np.linalg.norm(np.array(pi) - np.array(pj))
            if d < best_dist:
                best_dist = d
                best_i = i
                best_j = j

    return float(best_dist), best_i, best_j


def _wall_end_direction(wall: dict, ep_idx: int) -> np.ndarray:
    """Get the direction vector at the specified endpoint of a wall.

    For 2-point walls: single direction.
    For N-point polylines: direction of the first/last segment.

    ep_idx: 0 = start, 1 = end.
    """
    cl = wall["center_line_2d"]
    if ep_idx == 0:
        p1 = np.array(cl[0], dtype=float)
        p2 = np.array(cl[1], dtype=float)
    else:
        p1 = np.array(cl[-2], dtype=float)
        p2 = np.array(cl[-1], dtype=float)

    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-12:
        return np.array([1.0, 0.0])
    return d / length


def _angle_between(d1: np.ndarray, d2: np.ndarray) -> float:
    """Angle between two 2D direction vectors in degrees (0-180)."""
    cos_val = float(np.dot(d1, d2))
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_val))))


def _merge_two_walls(
    w1: dict, w2: dict, ep1: int, ep2: int,
) -> dict:
    """Merge two walls into a single polyline wall.

    The shared endpoint becomes an interior point of the new polyline.

    Args:
        w1, w2: Wall dicts.
        ep1: Endpoint index of w1 closest to w2 (0=start, 1=end).
        ep2: Endpoint index of w2 closest to w1 (0=start, 1=end).

    Returns:
        New wall dict with merged center_line_2d.
    """
    cl1 = [list(p) for p in w1["center_line_2d"]]
    cl2 = [list(p) for p in w2["center_line_2d"]]

    # Orient both chains so the shared endpoint is at the junction
    # We want: chain1 → shared_point → chain2
    if ep1 == 0:
        cl1 = cl1[::-1]  # reverse so shared point is at end
    # cl1 now ends at the shared point

    if ep2 == 1:
        cl2 = cl2[::-1]  # reverse so shared point is at start
    # cl2 now starts at the shared point

    # Merge: cl1[:-1] + [shared] + cl2[1:]
    # (avoid duplicating the shared point)
    shared = [(cl1[-1][0] + cl2[0][0]) / 2.0, (cl1[-1][1] + cl2[0][1]) / 2.0]
    merged_cl = cl1[:-1] + [shared] + cl2[1:]

    # Average properties
    t1 = w1.get("thickness", 0.2)
    t2 = w2.get("thickness", 0.2)
    hr1 = w1.get("height_range", [0, 3])
    hr2 = w2.get("height_range", [0, 3])

    return {
        "id": w1["id"],
        "center_line_2d": merged_cl,
        "wall_type": "polyline",
        "thickness": (t1 + t2) / 2.0,
        "height_range": [
            min(hr1[0], hr2[0]),
            max(hr1[1], hr2[1]),
        ],
        "normal_axis": w1.get("normal_axis", ""),
        "normal_vector": w1.get("normal_vector"),
        "synthetic": w1.get("synthetic", False) and w2.get("synthetic", False),
        "plane_ids": w1.get("plane_ids", []) + w2.get("plane_ids", []),
        "is_exterior": w1.get("is_exterior", True) or w2.get("is_exterior", True),
        "merged_from": [w1.get("id"), w2.get("id")],
    }


def merge_collinear_walls(
    walls: list[dict],
    angle_tolerance_deg: float = 10.0,
    endpoint_tolerance: float | None = None,
) -> list[dict]:
    """Merge collinear wall pairs sharing an endpoint into polyline walls.

    Two walls are candidates for merging when:
    1. They share an endpoint (within endpoint_tolerance)
    2. Their directions at the shared endpoint differ by < angle_tolerance_deg
    3. They have similar thickness (within 50%)

    Args:
        walls: List of wall dicts with center_line_2d.
        angle_tolerance_deg: Max angle between wall directions to merge.
        endpoint_tolerance: Max distance between endpoints to consider shared.
            If None, uses the minimum wall thickness as tolerance.

    Returns:
        New list of walls with merged polylines (originals removed).
    """
    if len(walls) < 2:
        return walls

    # Default endpoint tolerance from minimum wall thickness
    if endpoint_tolerance is None:
        thicknesses = [w.get("thickness", 0.2) for w in walls]
        endpoint_tolerance = min(thicknesses) if thicknesses else 0.2

    # Work with mutable copies
    remaining = [dict(w) for w in walls]
    merged_count = 0

    # Greedy pair merging
    changed = True
    while changed:
        changed = False
        n = len(remaining)

        best_pair = None
        best_dist = float("inf")

        for i in range(n):
            for j in range(i + 1, n):
                dist, ep_i, ep_j = _endpoint_distance(remaining[i], remaining[j])
                if dist > endpoint_tolerance:
                    continue

                # Check direction similarity at shared endpoint
                dir_i = _wall_end_direction(remaining[i], ep_i)
                dir_j = _wall_end_direction(remaining[j], ep_j)

                # When endpoints meet (ep_i=1, ep_j=0), directions should be similar
                # When ends meet (both ep=1 or both ep=0), one dir should be negated
                if (ep_i == 1 and ep_j == 0) or (ep_i == 0 and ep_j == 1):
                    # Sequential: w1 end → w2 start, directions should be similar
                    angle = _angle_between(dir_i, dir_j)
                else:
                    # Both starts or both ends meeting: negate one
                    angle = _angle_between(dir_i, -dir_j)

                if angle > angle_tolerance_deg:
                    continue

                # Check thickness compatibility (within 50%)
                t_i = remaining[i].get("thickness", 0.2)
                t_j = remaining[j].get("thickness", 0.2)
                t_ratio = max(t_i, t_j) / max(min(t_i, t_j), 1e-6)
                if t_ratio > 1.5:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j, ep_i, ep_j)

        if best_pair is not None:
            i, j, ep_i, ep_j = best_pair
            merged = _merge_two_walls(remaining[i], remaining[j], ep_i, ep_j)
            # Remove old walls, add merged
            remaining = [w for k, w in enumerate(remaining) if k not in (i, j)]
            remaining.append(merged)
            merged_count += 1
            changed = True

    if merged_count > 0:
        logger.info(
            f"Polyline merging: {merged_count} pairs merged, "
            f"{len(walls)} walls -> {len(remaining)} walls"
        )

    return remaining
