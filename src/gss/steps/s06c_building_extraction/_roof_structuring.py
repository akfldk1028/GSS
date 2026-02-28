"""Module E: Roof structure reconstruction.

Goes beyond simple roof type classification (s06b Module H) to reconstruct
the actual geometric structure: ridge lines, eave lines, and valley lines.

Algorithm:
1. Identify roof planes (inclined planes above ceiling)
2. Compute pairwise plane-plane intersections → ridge/valley lines
3. Compute roof plane ∩ wall top → eave lines
4. Build roof face graph
5. Output: type + faces + ridges + eaves + valleys
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _plane_centroid(plane: dict) -> np.ndarray | None:
    """Get 3D centroid from boundary."""
    bnd = plane.get("boundary_3d")
    if bnd is not None and len(bnd) > 0:
        pts = np.asarray(bnd)
        if pts.ndim == 2:
            return pts.mean(axis=0)
    return None


def _plane_intersect_line(
    n1: np.ndarray, d1: float, n2: np.ndarray, d2: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute intersection line of two planes.

    Each plane: n·x + d = 0
    Intersection direction = n1 × n2
    A point on the line is found by solving the 2-plane system.

    Returns:
        (point_on_line, direction) or None if planes are parallel.
    """
    direction = np.cross(n1, n2)
    dir_len = np.linalg.norm(direction)
    if dir_len < 1e-6:
        return None  # parallel
    direction = direction / dir_len

    # Find a point on the line:
    # We solve [n1; n2; direction] @ p = [-d1; -d2; 0]
    # Use pseudoinverse for robustness
    A = np.vstack([n1, n2, direction])
    b = np.array([-d1, -d2, 0.0])
    try:
        point = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    return point, direction


def _clip_line_to_boundaries(
    point: np.ndarray,
    direction: np.ndarray,
    plane1: dict,
    plane2: dict,
) -> list[list[float]] | None:
    """Clip an infinite intersection line to the overlapping extent of two planes.

    Projects both plane boundaries onto the line direction, finds overlap.
    """
    def project_extent(plane):
        bnd = plane.get("boundary_3d")
        if bnd is None or len(bnd) == 0:
            return None
        pts = np.asarray(bnd)
        if pts.ndim != 2:
            return None
        # Project onto line
        t_vals = (pts - point) @ direction
        return float(t_vals.min()), float(t_vals.max())

    ext1 = project_extent(plane1)
    ext2 = project_extent(plane2)

    if ext1 is None or ext2 is None:
        return None

    # Overlap
    t_min = max(ext1[0], ext2[0])
    t_max = min(ext1[1], ext2[1])

    if t_max - t_min < 0.01:
        return None  # no overlap

    p1 = (point + t_min * direction).tolist()
    p2 = (point + t_max * direction).tolist()
    return [p1, p2]


def _roof_slope_deg(normal: np.ndarray) -> float:
    """Compute roof slope angle from horizontal (0=flat, 90=vertical)."""
    ny = abs(normal[1])
    if ny > 1.0 - 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(ny, 0.0, 1.0))))


def _roof_aspect(normal: np.ndarray) -> str:
    """Determine which direction the roof faces (the downslope direction)."""
    # XZ component of normal = downslope direction
    xz = np.array([normal[0], normal[2]])
    norm = np.linalg.norm(xz)
    if norm < 0.05:
        return "flat"
    xz = xz / norm
    # Map to compass
    angle = float(np.degrees(np.arctan2(xz[1], xz[0])))  # -180 to 180
    if -45 <= angle < 45:
        return "east"
    elif 45 <= angle < 135:
        return "south"
    elif angle >= 135 or angle < -135:
        return "west"
    else:
        return "north"


def structure_roof(
    planes: list[dict],
    footprint: dict | None = None,
    ceiling_heights: list[float] | None = None,
    *,
    ridge_snap_tolerance: float = 0.3,
    min_roof_tilt: float = 0.15,
    max_roof_tilt: float = 0.85,
    scale: float = 1.0,
) -> dict:
    """Reconstruct roof structure from planes.

    Args:
        planes: All planes (roof planes identified by label or tilt).
        footprint: Building footprint dict (optional).
        ceiling_heights: Ceiling Y values (for height threshold).
        ridge_snap_tolerance: Snap tolerance for ridge endpoints (meters).
        min_roof_tilt: Min |ny| for roof plane.
        max_roof_tilt: Max |ny| for inclined roof (above = flat).
        scale: Coordinate scale.

    Returns:
        Dict with roof_type, faces, ridges, eaves, valleys.
    """
    # Determine ceiling threshold
    if ceiling_heights:
        max_ceiling = max(ceiling_heights)
    else:
        # Estimate from planes: highest floor/ceiling
        max_ceiling = float("-inf")
        for p in planes:
            if p.get("label") in ("ceiling",):
                c = _plane_centroid(p)
                if c is not None:
                    max_ceiling = max(max_ceiling, c[1])
        if max_ceiling == float("-inf"):
            max_ceiling = 0.0

    # 1. Identify roof planes
    roof_planes: list[dict] = []
    flat_count = 0
    inclined_count = 0

    for p in planes:
        n = np.asarray(p["normal"], dtype=float)
        ny = abs(n[1])

        # Must have some vertical component (not a wall)
        if ny < min_roof_tilt:
            continue

        # Check height
        centroid = _plane_centroid(p)
        if centroid is not None and centroid[1] < max_ceiling - 0.5 * scale:
            continue  # below ceiling

        # Already labeled as floor/ceiling → skip unless above ceiling
        if p.get("label") in ("floor",):
            continue
        if p.get("label") == "ceiling":
            if centroid is not None and centroid[1] < max_ceiling + 0.3 * scale:
                continue  # actual ceiling, not roof

        # It's a roof plane
        if ny > max_roof_tilt:
            roof_type = "flat"
            flat_count += 1
        else:
            roof_type = "inclined"
            inclined_count += 1

        roof_planes.append({
            **p,
            "roof_sub_type": roof_type,
            "_normal": n,
        })

    # 2. Build face list
    faces = []
    for rp in roof_planes:
        n = rp["_normal"]
        faces.append({
            "id": len(faces),
            "plane_id": rp["id"],
            "slope_deg": _roof_slope_deg(n),
            "aspect": _roof_aspect(n),
            "sub_type": rp["roof_sub_type"],
        })

    # 3. Compute ridge lines (intersection of inclined roof plane pairs)
    ridges = []
    valleys = []
    inclined_roofs = [rp for rp in roof_planes if rp["roof_sub_type"] == "inclined"]

    for i in range(len(inclined_roofs)):
        for j in range(i + 1, len(inclined_roofs)):
            rp1, rp2 = inclined_roofs[i], inclined_roofs[j]
            n1, n2 = rp1["_normal"], rp2["_normal"]

            result = _plane_intersect_line(n1, rp1["d"], n2, rp2["d"])
            if result is None:
                continue

            point, direction = result
            segment = _clip_line_to_boundaries(point, direction, rp1, rp2)
            if segment is None:
                continue

            # Ridge vs valley: if both normals point "upward" from the line,
            # it's a ridge; if they point "downward", it's a valley
            # Heuristic: ridge → intersection is at top, valley → at bottom
            mid = np.array([(s1 + s2) / 2 for s1, s2 in zip(segment[0], segment[1])])
            c1 = _plane_centroid(rp1)
            c2 = _plane_centroid(rp2)

            if c1 is not None and c2 is not None:
                avg_centroid_y = (c1[1] + c2[1]) / 2.0
                if mid[1] >= avg_centroid_y:
                    ridges.append(segment)
                else:
                    valleys.append(segment)
            else:
                ridges.append(segment)  # default to ridge

    # 4. Compute eave lines (roof plane ∩ vertical walls at their top)
    eaves = []
    wall_planes = [p for p in planes if p.get("label") == "wall"]

    for rp in roof_planes:
        n_roof = rp["_normal"]
        for wp in wall_planes:
            n_wall = np.asarray(wp["normal"], dtype=float)
            result = _plane_intersect_line(n_roof, rp["d"], n_wall, wp["d"])
            if result is None:
                continue

            point, direction = result
            segment = _clip_line_to_boundaries(point, direction, rp, wp)
            if segment is not None:
                eaves.append(segment)

    # 5. Classify overall roof type
    total = flat_count + inclined_count
    if total == 0:
        roof_type = "none"
    elif flat_count > 0 and inclined_count == 0:
        roof_type = "flat"
    elif inclined_count == 1:
        roof_type = "shed"
    elif inclined_count == 2:
        roof_type = "gable"
    elif inclined_count >= 4:
        roof_type = "hip"
    else:
        roof_type = "mixed"

    logger.info(
        f"Roof structuring: {total} roof planes "
        f"({flat_count} flat, {inclined_count} inclined) → {roof_type}, "
        f"{len(ridges)} ridges, {len(eaves)} eaves, {len(valleys)} valleys"
    )

    return {
        "roof_type": roof_type,
        "faces": faces,
        "ridges": ridges,
        "eaves": eaves,
        "valleys": valleys,
    }
