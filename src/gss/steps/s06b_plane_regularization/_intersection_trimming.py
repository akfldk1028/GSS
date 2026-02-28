"""Module D: Extend/trim wall center-lines to meet at corners.

For non-parallel wall center-lines, compute 2D intersection points in the
XZ plane. Extend or trim endpoints to the intersection point if it lies
along the wall's direction (or close to an endpoint).

Supports both Manhattan (±X/±Z) and arbitrary-angle walls.

Includes T-junction detection and multi-wall junction clustering.

Ref: Cloud2BIM adjust_intersections + extend_to_centerline patterns.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _is_manhattan(wall: dict) -> bool:
    """Check if wall is Manhattan-aligned (normal_axis is "x" or "z")."""
    axis = wall.get("normal_axis", "")
    return axis in ("x", "z")


def _walls_are_parallel(w1: dict, w2: dict, threshold: float = 0.98) -> bool:
    """Check if two walls are parallel by comparing their directions.

    For Manhattan walls: same normal_axis means parallel.
    For oblique walls: compare direction vectors.
    """
    # Manhattan shortcut
    a1 = w1.get("normal_axis", "")
    a2 = w2.get("normal_axis", "")
    if a1 in ("x", "z") and a2 in ("x", "z"):
        return a1 == a2

    # General case: compare wall directions
    d1 = _wall_direction(w1)
    d2 = _wall_direction(w2)
    return abs(float(np.dot(d1, d2))) > threshold


def _wall_direction(wall: dict) -> np.ndarray:
    """Get wall's direction as a unit vector (along center-line)."""
    cl = wall["center_line_2d"]
    p1, p2 = np.array(cl[0], dtype=float), np.array(cl[1], dtype=float)
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-12:
        return np.array([1.0, 0.0])
    return d / length


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


def _is_t_junction(
    ix: np.ndarray,
    p1: np.ndarray, p2: np.ndarray,
    snap_tolerance: float,
) -> bool:
    """Check if intersection point is a T-junction for segment p1-p2.

    A T-junction means the intersection is near the interior of the segment
    (not at an endpoint), and one wall's endpoint meets the middle of another.
    """
    d = p2 - p1
    seg_len_sq = np.dot(d, d)
    if seg_len_sq < 1e-12:
        return False

    t = float(np.dot(ix - p1, d) / seg_len_sq)
    # T-junction: intersection is in the interior of the segment (not near endpoints)
    margin = snap_tolerance / max(np.sqrt(seg_len_sq), 1e-6)
    return margin < t < (1.0 - margin)


def _cluster_nearby_endpoints(walls: list[dict], snap_tolerance: float) -> int:
    """Cluster nearby wall endpoints and snap them to centroids.

    When 3+ walls meet at roughly the same point, their endpoints
    should converge to a single junction point.

    Returns number of endpoints clustered.
    """
    # Collect all endpoints with references
    endpoints: list[tuple[np.ndarray, int, int]] = []  # (point, wall_idx, ep_idx)
    for wi, w in enumerate(walls):
        cl = w["center_line_2d"]
        endpoints.append((np.array(cl[0], dtype=float), wi, 0))
        endpoints.append((np.array(cl[1], dtype=float), wi, 1))

    if len(endpoints) < 3:
        return 0

    # Greedy clustering
    used = set()
    clustered = 0

    for i in range(len(endpoints)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(endpoints)):
            if j in used:
                continue
            if np.linalg.norm(endpoints[i][0] - endpoints[j][0]) <= snap_tolerance:
                cluster.append(j)
                used.add(j)

        if len(cluster) < 2:
            continue

        # Compute centroid
        centroid = np.mean([endpoints[k][0] for k in cluster], axis=0)

        # Snap all endpoints in cluster to centroid
        for k in cluster:
            _, wi, ei = endpoints[k]
            walls[wi]["center_line_2d"][ei] = centroid.tolist()
            clustered += 1

    if clustered > 0:
        logger.info(f"Junction clustering: {clustered} endpoints snapped to centroids")
    return clustered


def _should_snap_endpoint(
    ix: np.ndarray,
    p1: np.ndarray, p2: np.ndarray,
    snap_tolerance: float,
    max_ext: float,
    other_snapping: bool,
) -> tuple[bool, int, float, str]:
    """Decide whether to snap a wall endpoint to an intersection point.

    Returns (should_snap, endpoint_index, distance, reason).
    Reasons: "close", "extend", "trim", "no".

    The key insight: if the OTHER wall is snapping to this intersection,
    then this wall should also trim its excess endpoint to meet there,
    even if the intersection is in the interior of this segment.
    """
    ep_idx, dist, t = _nearest_endpoint_info(ix, p1, p2)

    # Case 1: intersection is very close to an endpoint → always snap
    if dist <= snap_tolerance:
        return True, ep_idx, dist, "close"

    # Case 2: intersection is beyond the endpoint → extend
    is_beyond = (ep_idx == 0 and t < 0) or (ep_idx == 1 and t > 1)
    if is_beyond and dist <= max_ext:
        return True, ep_idx, dist, "extend"

    # Case 3: intersection is inside the segment, but the other wall
    # is trying to meet here → trim the nearest endpoint to the intersection.
    # This handles T-junctions where a wall overshoots past a corner.
    # Only trim up to 20% of wall length to avoid destroying walls.
    if other_snapping and 0 < t < 1:
        wall_len = float(np.linalg.norm(p2 - p1))
        max_trim = wall_len * 0.2
        if dist <= max_trim:
            return True, ep_idx, dist, "trim"

    return False, ep_idx, dist, "no"


def _constrained_snap(
    ix: np.ndarray, wall: dict, ep_idx: int,
) -> list[float]:
    """Snap endpoint to intersection, constrained to wall's direction line.

    For Manhattan walls:
    - X-normal wall: keep X constant, only change Z (wall extends along Z)
    - Z-normal wall: keep Z constant, only change X (wall extends along X)

    For arbitrary-angle walls:
    - Project ix onto the infinite line through the wall's center-line
    """
    axis = wall.get("normal_axis", "")
    cl = wall["center_line_2d"]

    if axis == "x":
        # X-normal wall: compute wall's center X from both endpoints, keep it
        wall_x = (cl[0][0] + cl[1][0]) / 2.0
        return [wall_x, ix[1]]
    elif axis == "z":
        # Z-normal wall: compute wall's center Z from both endpoints, keep it
        wall_z = (cl[0][1] + cl[1][1]) / 2.0
        return [ix[0], wall_z]
    else:
        # General case: project ix onto the wall's center-line (infinite line)
        p1 = np.array(cl[0], dtype=float)
        p2 = np.array(cl[1], dtype=float)
        d = p2 - p1
        d_len_sq = np.dot(d, d)
        if d_len_sq < 1e-12:
            return ix.tolist()
        # Anchor is the other endpoint (the one not being snapped)
        anchor = np.array(cl[1 - ep_idx], dtype=float)
        t = np.dot(ix - anchor, d / d_len_sq * np.linalg.norm(d))
        d_norm = d / np.sqrt(d_len_sq)
        return (anchor + t * d_norm).tolist()


def _enforce_wall_straightness(walls: list[dict]) -> int:
    """Post-process: enforce that walls maintain straight center-lines.

    For Manhattan walls:
    - X-normal walls → both endpoints get same X (average)
    - Z-normal walls → both endpoints get same Z (average)

    For oblique walls:
    - Project endpoints onto the line defined by the wall's normal_vector
    - Ensures the wall stays on its plane

    Returns number of walls corrected.
    """
    corrected = 0
    for w in walls:
        cl = w["center_line_2d"]
        axis = w.get("normal_axis", "")

        if axis == "x":
            avg_x = (cl[0][0] + cl[1][0]) / 2.0
            if abs(cl[0][0] - cl[1][0]) > 1e-6:
                cl[0][0] = avg_x
                cl[1][0] = avg_x
                corrected += 1
        elif axis == "z":
            avg_z = (cl[0][1] + cl[1][1]) / 2.0
            if abs(cl[0][1] - cl[1][1]) > 1e-6:
                cl[0][1] = avg_z
                cl[1][1] = avg_z
                corrected += 1
        elif "normal_vector" in w:
            # Oblique wall: ensure both endpoints lie on the same line
            # perpendicular to normal_vector
            nv = np.array(w["normal_vector"])  # 2D normal in XZ
            p1 = np.array(cl[0], dtype=float)
            p2 = np.array(cl[1], dtype=float)
            # Project both endpoints onto the normal direction
            d1 = np.dot(p1, nv)
            d2 = np.dot(p2, nv)
            avg_d = (d1 + d2) / 2.0
            if abs(d1 - d2) > 1e-6:
                # Move each point to avg_d along normal
                cl[0] = (p1 + (avg_d - d1) * nv).tolist()
                cl[1] = (p2 + (avg_d - d2) * nv).tolist()
                corrected += 1

    return corrected


def _reconnect_corners(walls: list[dict], snap_tolerance: float) -> int:
    """Reconnect wall endpoints at corners after straightness enforcement.

    For Manhattan walls:
    - The corner of X-normal and Z-normal walls is at (X_wall_x, Z_wall_z)

    For general walls:
    - Compute the intersection of the two wall lines and snap nearby endpoints

    Returns number of endpoints reconnected.
    """
    reconnected = 0
    n = len(walls)

    for i in range(n):
        for j in range(i + 1, n):
            if _walls_are_parallel(walls[i], walls[j]):
                continue

            a1 = walls[i].get("normal_axis", "")
            a2 = walls[j].get("normal_axis", "")

            # Manhattan fast path
            if a1 in ("x", "z") and a2 in ("x", "z") and a1 != a2:
                if a1 == "x":
                    w_x, w_z = walls[i], walls[j]
                else:
                    w_x, w_z = walls[j], walls[i]

                cl_x = w_x["center_line_2d"]
                cl_z = w_z["center_line_2d"]

                wall_x = cl_x[0][0]
                wall_z = cl_z[0][1]
                corner = np.array([wall_x, wall_z])

                for cl in [cl_x, cl_z]:
                    d0 = np.linalg.norm(np.array(cl[0]) - corner)
                    d1 = np.linalg.norm(np.array(cl[1]) - corner)
                    min_d = min(d0, d1)
                    ep = 0 if d0 <= d1 else 1

                    if min_d <= snap_tolerance:
                        cl[ep] = corner.tolist()
                        reconnected += 1
            else:
                # General case: compute line-line intersection
                cl_i = walls[i]["center_line_2d"]
                cl_j = walls[j]["center_line_2d"]
                p1 = np.array(cl_i[0], dtype=float)
                p2 = np.array(cl_i[1], dtype=float)
                p3 = np.array(cl_j[0], dtype=float)
                p4 = np.array(cl_j[1], dtype=float)

                ix = _line_intersection_2d(p1, p2, p3, p4)
                if ix is None:
                    continue

                # Check if any endpoint is close enough to reconnect
                for cl in [cl_i, cl_j]:
                    ep0 = np.array(cl[0], dtype=float)
                    ep1 = np.array(cl[1], dtype=float)
                    d0 = float(np.linalg.norm(ep0 - ix))
                    d1 = float(np.linalg.norm(ep1 - ix))
                    min_d = min(d0, d1)
                    ep = 0 if d0 <= d1 else 1

                    if min_d <= snap_tolerance:
                        cl[ep] = ix.tolist()
                        reconnected += 1

    if reconnected > 0:
        logger.info(f"Corner reconnection: {reconnected} endpoints snapped to ideal corners")
    return reconnected


def trim_intersections(walls: list[dict], snap_tolerance: float = 0.5) -> dict:
    """Extend/trim wall center-line endpoints to meet at intersection points.

    For each pair of non-parallel walls:
    1. Compute their infinite-line intersection
    2. For each wall, check if the intersection is a valid corner:
       - The intersection is near an endpoint (within snap_tolerance), OR
       - The intersection is along the wall's extension direction and
         the extension is within 50% of the wall's length
       - The intersection is inside the segment (trim) and the other
         wall is also snapping (T-junction trim, max 20% of length)
    3. If valid, extend/trim/snap the nearest endpoint to the intersection,
       constrained to the wall's axis / direction line

    Also handles:
    - Wall straightness enforcement: ensures walls stay straight after snapping
    - Multi-wall junctions: cluster nearby endpoints to a centroid

    Args:
        walls: list of wall dicts with center_line_2d (modified in-place).
        snap_tolerance: tolerance for snapping nearby endpoints (already scaled).

    Returns:
        stats dict.
    """
    snapped_count = 0
    extended_count = 0
    trimmed_count = 0
    n = len(walls)

    for i in range(n):
        for j in range(i + 1, n):
            if _walls_are_parallel(walls[i], walls[j]):
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

            len_i = float(np.linalg.norm(p2 - p1))
            len_j = float(np.linalg.norm(p4 - p3))
            max_ext_i = max(snap_tolerance, len_i * 0.5)
            max_ext_j = max(snap_tolerance, len_j * 0.5)

            # First pass: check both walls without trim (to know if other is snapping)
            snap_i_no_trim, ep_i, dist_i, reason_i = _should_snap_endpoint(
                ix, p1, p2, snap_tolerance, max_ext_i, other_snapping=False,
            )
            snap_j_no_trim, ep_j, dist_j, reason_j = _should_snap_endpoint(
                ix, p3, p4, snap_tolerance, max_ext_j, other_snapping=False,
            )

            # Second pass: if one side is already snapping, allow trim on the other
            snap_i, ep_i, dist_i, reason_i = _should_snap_endpoint(
                ix, p1, p2, snap_tolerance, max_ext_i, other_snapping=snap_j_no_trim,
            )
            snap_j, ep_j, dist_j, reason_j = _should_snap_endpoint(
                ix, p3, p4, snap_tolerance, max_ext_j, other_snapping=snap_i_no_trim or snap_i,
            )

            if snap_i:
                cl_i[ep_i] = _constrained_snap(ix, walls[i], ep_i)
                if reason_i == "close":
                    snapped_count += 1
                elif reason_i == "extend":
                    extended_count += 1
                elif reason_i == "trim":
                    trimmed_count += 1
                logger.debug(
                    f"Wall {walls[i]['id']} ep{ep_i}: "
                    f"{reason_i} {dist_i:.2f} to corner with wall {walls[j]['id']}"
                )

            if snap_j:
                cl_j[ep_j] = _constrained_snap(ix, walls[j], ep_j)
                if reason_j == "close":
                    snapped_count += 1
                elif reason_j == "extend":
                    extended_count += 1
                elif reason_j == "trim":
                    trimmed_count += 1
                logger.debug(
                    f"Wall {walls[j]['id']} ep{ep_j}: "
                    f"{reason_j} {dist_j:.2f} to corner with wall {walls[i]['id']}"
                )

    # Enforce wall straightness (fix any remaining drift from snapping)
    axis_corrected = _enforce_wall_straightness(walls)
    if axis_corrected > 0:
        logger.info(f"Wall straightness: corrected {axis_corrected} walls")

    # Reconnect corners
    _reconnect_corners(walls, snap_tolerance * 2.0)

    # Multi-wall junction clustering (use 1.5x tolerance for clustering)
    junction_clustered = _cluster_nearby_endpoints(walls, snap_tolerance * 1.5)

    # Re-enforce wall straightness after clustering
    _enforce_wall_straightness(walls)

    # Final corner reconnection
    _reconnect_corners(walls, snap_tolerance * 2.0)

    total = snapped_count + extended_count + trimmed_count
    logger.info(
        f"Intersection trimming: {snapped_count} snapped, "
        f"{extended_count} extended, {trimmed_count} trimmed, "
        f"{junction_clustered} junction-clustered ({total} total)"
    )
    return {
        "snapped_endpoints": snapped_count,
        "extended_endpoints": extended_count,
        "trimmed_endpoints": trimmed_count,
        "junction_clustered": junction_clustered,
    }
