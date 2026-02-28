"""Module C: Facade detection — group vertical planes into building faces.

Groups exterior vertical planes by normal direction, then merges coplanar
planes within each group into facade segments.

Algorithm:
1. Select vertical planes (|ny| < threshold) as wall candidates
2. Project normals to XZ plane, cluster by angle
3. Within each normal group, cluster by d-value → coplanar sets
4. Each cluster = one facade; compute boundary union, area, orientation
5. Determine orientation label (N/S/E/W/NE/NW/SE/SW)
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

_ORIENTATION_LABELS = [
    ("north", np.array([0.0, 0.0, -1.0])),
    ("south", np.array([0.0, 0.0, 1.0])),
    ("east", np.array([1.0, 0.0, 0.0])),
    ("west", np.array([-1.0, 0.0, 0.0])),
]


def _project_xz(normal: np.ndarray) -> np.ndarray | None:
    """Project normal to XZ and normalize."""
    xz = np.array([normal[0], 0.0, normal[2]])
    norm = np.linalg.norm(xz)
    if norm < 0.1:
        return None
    return xz / norm


def _determine_orientation(normal_xz: np.ndarray) -> str:
    """Map a facade normal to a compass orientation."""
    best_label = "unknown"
    best_dot = -2.0
    for label, axis in _ORIENTATION_LABELS:
        d = float(np.dot(normal_xz, axis))
        if d > best_dot:
            best_dot = d
            best_label = label
    # Check for diagonal orientations (dot < 0.85 for primary → diagonal)
    if best_dot < 0.85:
        # It's between two primary directions
        dots = [(label, float(np.dot(normal_xz, axis))) for label, axis in _ORIENTATION_LABELS]
        dots.sort(key=lambda x: -x[1])
        if dots[0][1] > 0.3 and dots[1][1] > 0.3:
            best_label = dots[0][0][:1] + dots[1][0][:1]  # e.g. "ne"
            best_label = best_label.upper()
            # Canonicalize
            canon = {"NE": "NE", "EN": "NE", "NW": "NW", "WN": "NW",
                     "SE": "SE", "ES": "SE", "SW": "SW", "WS": "SW"}
            best_label = canon.get(best_label, best_label).lower()
    return best_label


def _plane_height_range(plane: dict) -> tuple[float, float]:
    """Get Y height range from boundary."""
    bnd = plane.get("boundary_3d")
    if bnd is None or len(bnd) == 0:
        return (0.0, 0.0)
    pts = np.asarray(bnd)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return (0.0, 0.0)
    return (float(pts[:, 1].min()), float(pts[:, 1].max()))


def _plane_area(plane: dict) -> float:
    """Approximate plane area from boundary (XZ projection for walls)."""
    bnd = plane.get("boundary_3d")
    if bnd is None or len(bnd) < 3:
        return 0.0
    pts = np.asarray(bnd)
    if pts.ndim != 2:
        return 0.0
    # Use shoelace on XZ projection * height range
    y_range = pts[:, 1].max() - pts[:, 1].min()
    x_range = pts[:, 0].max() - pts[:, 0].min()
    z_range = pts[:, 2].max() - pts[:, 2].min()
    # Heuristic: wall area = max(x_range, z_range) * y_range
    return float(max(x_range, z_range) * y_range)


def _plane_d_value(plane: dict, axis_vec: np.ndarray) -> float:
    """Project plane position onto the axis direction."""
    n = np.asarray(plane["normal"], dtype=float)
    return float(-plane["d"] * np.dot(n, axis_vec) / (np.dot(n, n) + 1e-12))


def detect_facades(
    planes: list[dict],
    building_centroid: np.ndarray | None = None,
    *,
    normal_threshold: float = 0.3,
    coplanar_dist_threshold: float = 0.5,
    coplanar_angle_threshold: float = 15.0,
    min_facade_area: float = 2.0,
    scale: float = 1.0,
) -> list[dict]:
    """Detect and group facade planes.

    Args:
        planes: List of plane dicts.
        building_centroid: XYZ centroid of building (for outward normal validation).
        normal_threshold: Max |ny| for a vertical (wall) plane.
        coplanar_dist_threshold: Max d-difference (meters) for coplanar merging.
        coplanar_angle_threshold: Max angle (degrees) for normal grouping.
        min_facade_area: Min area (sq meters) for a valid facade.
        scale: Coordinate scale.

    Returns:
        List of facade dicts with id, normal, plane_ids, height_range, area, orientation.
    """
    dist_thresh_scaled = coplanar_dist_threshold * scale
    min_area_scaled = min_facade_area * scale * scale

    # 1. Select vertical planes
    vertical_planes: list[tuple[int, dict, np.ndarray]] = []
    for i, p in enumerate(planes):
        n = np.asarray(p["normal"], dtype=float)
        if abs(n[1]) > normal_threshold:
            continue  # floor/ceiling/roof, not a wall
        xz = _project_xz(n)
        if xz is None:
            continue
        vertical_planes.append((i, p, xz))

    if not vertical_planes:
        logger.info("No vertical planes found for facade detection")
        return []

    # 2. Cluster by normal direction (greedy angle clustering)
    cos_thresh = np.cos(np.radians(coplanar_angle_threshold))
    normal_groups: list[list[int]] = []  # indices into vertical_planes
    group_normals: list[np.ndarray] = []

    for vi, (pi, p, xz) in enumerate(vertical_planes):
        assigned = False
        for gi, gn in enumerate(group_normals):
            if float(np.dot(xz, gn)) >= cos_thresh:
                normal_groups[gi].append(vi)
                # Update group normal (running mean)
                members = [vertical_planes[j][2] for j in normal_groups[gi]]
                mean = np.mean(members, axis=0)
                mean_norm = np.linalg.norm(mean)
                if mean_norm > 1e-6:
                    group_normals[gi] = mean / mean_norm
                assigned = True
                break
        if not assigned:
            normal_groups.append([vi])
            group_normals.append(xz.copy())

    # 3. Within each normal group, cluster by d-value
    facades: list[dict] = []
    facade_id = 0

    for gi, members in enumerate(normal_groups):
        axis_vec = group_normals[gi]

        # Get d-values for each member
        d_values = []
        for vi in members:
            pi, p, xz = vertical_planes[vi]
            d_val = _plane_d_value(p, axis_vec)
            d_values.append((vi, d_val))

        # Sort by d-value and cluster
        d_values.sort(key=lambda x: x[1])
        d_clusters: list[list[int]] = []  # list of vi indices

        for vi, d_val in d_values:
            placed = False
            for dc in d_clusters:
                last_vi = dc[-1]
                last_d = next(dv for v, dv in d_values if v == last_vi)
                if abs(d_val - last_d) <= dist_thresh_scaled:
                    dc.append(vi)
                    placed = True
                    break
            if not placed:
                d_clusters.append([vi])

        # 4. Each d-cluster = one facade
        for dc in d_clusters:
            plane_ids = [vertical_planes[vi][1]["id"] for vi in dc]
            member_planes = [vertical_planes[vi][1] for vi in dc]

            # Compute aggregate properties
            total_area = sum(_plane_area(p) for p in member_planes)
            if total_area < min_area_scaled:
                continue

            # Height range
            all_y_min = float("inf")
            all_y_max = float("-inf")
            for p in member_planes:
                y_lo, y_hi = _plane_height_range(p)
                all_y_min = min(all_y_min, y_lo)
                all_y_max = max(all_y_max, y_hi)

            # Mean normal
            mean_n = np.mean([vertical_planes[vi][2] for vi in dc], axis=0)
            mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-12)

            # Orientation
            orientation = _determine_orientation(mean_n)

            facade = {
                "id": facade_id,
                "normal": mean_n.tolist(),
                "plane_ids": plane_ids,
                "height_range": [float(all_y_min), float(all_y_max)],
                "area": float(total_area / (scale * scale)),  # back to meters²
                "orientation": orientation,
            }
            facades.append(facade)
            facade_id += 1

    logger.info(
        f"Facade detection: {len(vertical_planes)} vertical planes → "
        f"{len(facades)} facades"
    )
    return facades
