"""Module G: Classify walls as interior/exterior.

Uses the convex hull of wall center-line midpoints to determine which walls
form the building perimeter (exterior) versus internal partitions (interior).

Algorithm:
1. Collect all wall center-line midpoints
2. Compute convex hull
3. For each wall, check if its midpoint is near a hull edge → exterior
4. Validate: exterior wall's outward normal should point away from hull center
5. Extract building footprint from ordered exterior walls
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _wall_midpoint(wall: dict) -> np.ndarray:
    """Get center-line midpoint in XZ plane."""
    cl = wall["center_line_2d"]
    return np.array([(cl[0][0] + cl[1][0]) / 2.0, (cl[0][1] + cl[1][1]) / 2.0])


def _point_to_segment_distance(
    pt: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray,
) -> float:
    """Compute minimum distance from a point to a line segment."""
    d = seg_b - seg_a
    seg_len_sq = np.dot(d, d)
    if seg_len_sq < 1e-12:
        return float(np.linalg.norm(pt - seg_a))
    t = float(np.dot(pt - seg_a, d) / seg_len_sq)
    t = max(0.0, min(1.0, t))
    proj = seg_a + t * d
    return float(np.linalg.norm(pt - proj))


def classify_walls(
    walls: list[dict],
    hull_distance_tolerance: float | None = None,
) -> dict:
    """Classify each wall as interior or exterior.

    A wall is exterior if its center-line midpoint lies on or very near
    the convex hull of all wall midpoints.

    Args:
        walls: list of wall dicts (modified in-place: adds "is_exterior" field).
        hull_distance_tolerance: max distance from hull edge to classify as exterior.
            If None, auto-computed as 5% of hull perimeter / num_edges.

    Returns:
        stats dict with counts.
    """
    if len(walls) < 3:
        for w in walls:
            w["is_exterior"] = True
        return {"exterior": len(walls), "interior": 0}

    # Collect midpoints
    midpoints = np.array([_wall_midpoint(w) for w in walls])

    # Compute convex hull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(midpoints)
    except Exception as e:
        logger.warning(f"ConvexHull failed ({e}), marking all walls as exterior")
        for w in walls:
            w["is_exterior"] = True
        return {"exterior": len(walls), "interior": 0}

    hull_vertices = hull.vertices
    hull_pts = midpoints[hull_vertices]

    # Auto-compute tolerance if not provided
    if hull_distance_tolerance is None:
        perimeter = 0.0
        n_hull = len(hull_pts)
        for i in range(n_hull):
            perimeter += np.linalg.norm(hull_pts[(i + 1) % n_hull] - hull_pts[i])
        hull_distance_tolerance = perimeter * 0.05 / max(n_hull, 1)
        hull_distance_tolerance = max(hull_distance_tolerance, 0.1)

    # Hull center
    hull_center = hull_pts.mean(axis=0)

    # Build hull edges
    n_hull = len(hull_pts)
    hull_edges = []
    for i in range(n_hull):
        hull_edges.append((hull_pts[i], hull_pts[(i + 1) % n_hull]))

    exterior_count = 0
    interior_count = 0

    for wi, w in enumerate(walls):
        mid = midpoints[wi]

        # Check distance to nearest hull edge
        min_dist = float("inf")
        for seg_a, seg_b in hull_edges:
            d = _point_to_segment_distance(mid, seg_a, seg_b)
            min_dist = min(min_dist, d)

        is_exterior = min_dist <= hull_distance_tolerance

        # Validate normal direction for exterior walls
        if is_exterior and "normal_vector" in w:
            nv = np.array(w["normal_vector"])
            to_outside = mid - hull_center
            if np.dot(nv, to_outside) < 0:
                # Normal points inward — might be interior wall near hull
                # Reduce confidence but still classify as exterior
                logger.debug(
                    f"Wall {w['id']}: exterior candidate but normal points inward"
                )

        w["is_exterior"] = bool(is_exterior)

        if is_exterior:
            exterior_count += 1
        else:
            interior_count += 1

    logger.info(
        f"Exterior classification: {exterior_count} exterior, "
        f"{interior_count} interior (tolerance={hull_distance_tolerance:.2f})"
    )
    return {"exterior": exterior_count, "interior": interior_count}


def extract_building_footprint(walls: list[dict]) -> list[list[float]] | None:
    """Extract building footprint polygon from exterior walls.

    Orders exterior wall center-line endpoints into a closed polygon.

    Returns:
        List of [x, z] coordinates forming the footprint polygon, or None.
    """
    exterior_walls = [w for w in walls if w.get("is_exterior", False)]
    if len(exterior_walls) < 3:
        return None

    # Collect all exterior wall endpoints
    pts = []
    for w in exterior_walls:
        cl = w["center_line_2d"]
        pts.append(cl[0])
        pts.append(cl[1])

    pts_arr = np.array(pts)

    # Use convex hull to order points into a polygon
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_arr)
        footprint = pts_arr[hull.vertices].tolist()
        # Close the polygon
        footprint.append(footprint[0])
        logger.info(f"Building footprint: {len(footprint) - 1} vertices")
        return footprint
    except Exception as e:
        logger.warning(f"Footprint extraction failed: {e}")
        return None
