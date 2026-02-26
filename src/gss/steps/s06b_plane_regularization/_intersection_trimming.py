"""Module D: Extend/trim wall center-lines to meet at corners.

For non-parallel wall center-lines, compute 2D intersection points in the
XZ plane. Extend or trim endpoints to the intersection point if it lies
along the wall's direction (or close to an endpoint).

Ref: Cloud2BIM adjust_intersections + extend_to_centerline patterns.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _line_intersection_2d(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> np.ndarray | None:
    """Compute intersection of two infinite lines through (p1,p2) and (p3,p4).

    Returns intersection point or None if parallel.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross) < 1e-10:
        return None  # parallel

    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def _nearest_endpoint_info(
    ix: np.ndarray, p1: np.ndarray, p2: np.ndarray,
) -> tuple[int, float, float]:
    """For intersection point ix and segment p1-p2, find the nearest endpoint.

    Returns (endpoint_index, distance_to_endpoint, t_parameter).
    t=0 at p1, t=1 at p2. t<0 means ix is before p1, t>1 means after p2.
    """
    d = p2 - p1
    seg_len_sq = np.dot(d, d)
    if seg_len_sq < 1e-12:
        return 0, float(np.linalg.norm(ix - p1)), 0.0

    t = float(np.dot(ix - p1, d) / seg_len_sq)
    d0 = float(np.linalg.norm(ix - p1))
    d1 = float(np.linalg.norm(ix - p2))

    if d0 <= d1:
        return 0, d0, t
    return 1, d1, t


def trim_intersections(walls: list[dict], snap_tolerance: float = 0.5) -> dict:
    """Extend/trim wall center-line endpoints to meet at intersection points.

    For each pair of non-parallel walls:
    1. Compute their infinite-line intersection
    2. For each wall, check if the intersection is a valid corner:
       - The intersection is near an endpoint (within snap_tolerance), OR
       - The intersection is along the wall's extension direction and
         the extension is within 50% of the wall's length
    3. If valid, extend/snap the nearest endpoint to the intersection

    Args:
        walls: list of wall dicts with center_line_2d (modified in-place).
        snap_tolerance: base tolerance for snapping nearby endpoints.

    Returns:
        stats dict.
    """
    snapped_count = 0
    extended_count = 0
    n = len(walls)

    for i in range(n):
        for j in range(i + 1, n):
            if walls[i]["normal_axis"] == walls[j]["normal_axis"]:
                continue

            cl_i = walls[i]["center_line_2d"]
            cl_j = walls[j]["center_line_2d"]
            p1 = np.array(cl_i[0], dtype=float)
            p2 = np.array(cl_i[1], dtype=float)
            p3 = np.array(cl_j[0], dtype=float)
            p4 = np.array(cl_j[1], dtype=float)

            ix = _line_intersection_2d(p1, p2, p3, p4)
            if ix is None:
                continue

            # Max extension = 50% of wall length (scale-independent)
            len_i = float(np.linalg.norm(p2 - p1))
            len_j = float(np.linalg.norm(p4 - p3))
            max_ext_i = max(snap_tolerance, len_i * 0.5)
            max_ext_j = max(snap_tolerance, len_j * 0.5)

            # Process wall i
            ep_idx_i, dist_i, t_i = _nearest_endpoint_info(ix, p1, p2)
            should_snap_i = False

            if dist_i <= snap_tolerance:
                should_snap_i = True
            elif dist_i <= max_ext_i:
                # Extend if ix is beyond the endpoint (t < 0 or t > 1)
                if (ep_idx_i == 0 and t_i < 0) or (ep_idx_i == 1 and t_i > 1):
                    should_snap_i = True

            if should_snap_i:
                cl_i[ep_idx_i] = ix.tolist()
                if dist_i <= snap_tolerance:
                    snapped_count += 1
                else:
                    extended_count += 1
                logger.debug(
                    f"Wall {walls[i]['id']} ep{ep_idx_i}: "
                    f"{'snapped' if dist_i <= snap_tolerance else 'extended'} "
                    f"{dist_i:.2f} to corner with wall {walls[j]['id']}"
                )

            # Process wall j
            ep_idx_j, dist_j, t_j = _nearest_endpoint_info(ix, p3, p4)
            should_snap_j = False

            if dist_j <= snap_tolerance:
                should_snap_j = True
            elif dist_j <= max_ext_j:
                if (ep_idx_j == 0 and t_j < 0) or (ep_idx_j == 1 and t_j > 1):
                    should_snap_j = True

            if should_snap_j:
                cl_j[ep_idx_j] = ix.tolist()
                if dist_j <= snap_tolerance:
                    snapped_count += 1
                else:
                    extended_count += 1
                logger.debug(
                    f"Wall {walls[j]['id']} ep{ep_idx_j}: "
                    f"{'snapped' if dist_j <= snap_tolerance else 'extended'} "
                    f"{dist_j:.2f} to corner with wall {walls[i]['id']}"
                )

    total = snapped_count + extended_count
    logger.info(
        f"Intersection trimming: {snapped_count} snapped, "
        f"{extended_count} extended ({total} total)"
    )
    return {"snapped_endpoints": snapped_count, "extended_endpoints": extended_count}
