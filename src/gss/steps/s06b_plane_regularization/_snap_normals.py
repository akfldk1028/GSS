"""Module A: Snap plane normals to dominant axes.

Supports two modes:
- Manhattan: walls → nearest ±X or ±Z axis (original behavior)
- Cluster: discover dominant wall directions from data → snap to those

In Manhattan-aligned Y-up space:
- Wall normals → nearest dominant axis (XZ-plane projected)
- Floor/ceiling normals → nearest ±Y axis
- Recompute d to keep plane passing through same region
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Candidate axes for horizontal planes (Manhattan Y-up)
_HORIZ_AXES = [
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
]

# Default Manhattan wall axes
_MANHATTAN_WALL_AXES = [
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
]


def _project_to_xz(normal: np.ndarray) -> np.ndarray | None:
    """Project normal to XZ plane and normalize. Returns None if degenerate."""
    xz = np.array([normal[0], 0.0, normal[2]])
    norm = np.linalg.norm(xz)
    if norm < 0.1:
        return None
    return xz / norm


def _discover_wall_axes(
    planes: list[dict], threshold_deg: float = 15.0,
) -> list[np.ndarray]:
    """Cluster wall normals to discover dominant directions.

    1. Collect wall normals, project to XZ plane, normalize
    2. Greedy angle clustering (threshold_deg tolerance)
    3. Each cluster's mean normal → candidate axis
    4. Opposite direction automatically added (±)

    Returns:
        list of unit-length axis vectors (always in ± pairs).
    """
    # Collect XZ-projected wall normals
    xz_normals = []
    for p in planes:
        if p["label"] != "wall":
            continue
        n = np.asarray(p["normal"], dtype=float)
        proj = _project_to_xz(n)
        if proj is not None:
            xz_normals.append(proj)

    if not xz_normals:
        logger.warning("No wall normals found for clustering, falling back to Manhattan axes")
        return list(_MANHATTAN_WALL_AXES)

    # Canonicalize: ensure first non-zero component is positive (so ±pairs collapse)
    canonical = []
    for n in xz_normals:
        if n[0] < -1e-6 or (abs(n[0]) < 1e-6 and n[2] < -1e-6):
            canonical.append(-n)
        else:
            canonical.append(n)

    threshold_cos = np.cos(np.radians(threshold_deg))

    # Greedy clustering
    clusters: list[list[np.ndarray]] = []
    for n in canonical:
        assigned = False
        for cluster in clusters:
            mean_n = np.mean(cluster, axis=0)
            mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-12)
            if np.dot(n, mean_n) >= threshold_cos:
                cluster.append(n)
                assigned = True
                break
        if not assigned:
            clusters.append([n])

    # Build axes from clusters (both + and - directions)
    axes = []
    for cluster in clusters:
        mean_n = np.mean(cluster, axis=0)
        mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-12)
        axes.append(mean_n.copy())
        axes.append(-mean_n)

    logger.info(
        f"Normal clustering: {len(xz_normals)} wall normals → "
        f"{len(clusters)} dominant directions → {len(axes)} axes"
    )
    return axes


def _snap_one(normal: np.ndarray, candidates: list[np.ndarray], threshold_deg: float):
    """Find best axis candidate and return (snapped_normal, angle) or (None, angle)."""
    best_dot = -2.0
    best_axis = None
    for axis in candidates:
        d = np.dot(normal, axis)
        if d > best_dot:
            best_dot = d
            best_axis = axis
    angle_deg = np.degrees(np.arccos(np.clip(best_dot, -1.0, 1.0)))
    if angle_deg <= threshold_deg:
        return best_axis.copy(), angle_deg
    return None, angle_deg


def _plane_centroid(plane: dict) -> np.ndarray:
    """Get a representative point on the plane.

    Uses boundary centroid if available, otherwise the closest point
    on the plane to the origin: p0 = -d * n.
    """
    bnd = plane.get("boundary_3d")
    if bnd is not None and len(bnd) > 0:
        pts = np.asarray(bnd)
        if pts.ndim == 2 and len(pts) > 0:
            return pts.mean(axis=0)
    return -plane["d"] * plane["normal"]


def snap_normals(
    planes: list[dict],
    threshold_deg: float = 20.0,
    normal_mode: Literal["manhattan", "cluster"] = "manhattan",
    cluster_angle_tolerance: float = 15.0,
) -> dict:
    """Snap plane normals to dominant axes.

    Args:
        planes: list of plane dicts (modified in-place), in Manhattan space.
        threshold_deg: max angle deviation to allow snapping.
        normal_mode: "manhattan" uses ±X/±Z, "cluster" discovers axes from data.
        cluster_angle_tolerance: angle tolerance for clustering (cluster mode only).

    Returns:
        stats dict with counts and discovered axes.
    """
    # Determine wall axes
    if normal_mode == "cluster":
        wall_axes = _discover_wall_axes(planes, threshold_deg=cluster_angle_tolerance)
    else:
        wall_axes = list(_MANHATTAN_WALL_AXES)

    snapped_walls = 0
    snapped_horiz = 0
    skipped = 0

    for p in planes:
        label = p["label"]
        normal = np.asarray(p["normal"], dtype=float)

        if label == "wall":
            candidates = wall_axes
        elif label in ("floor", "ceiling"):
            candidates = _HORIZ_AXES
        else:
            continue

        snapped, angle = _snap_one(normal, candidates, threshold_deg)
        if snapped is None:
            logger.debug(
                f"Plane {p['id']} ({label}): angle {angle:.1f}° > threshold, not snapped"
            )
            skipped += 1
            continue

        centroid = _plane_centroid(p)
        p["normal"] = snapped
        p["d"] = float(-np.dot(snapped, centroid))

        if label == "wall":
            snapped_walls += 1
        else:
            snapped_horiz += 1
        logger.debug(
            f"Plane {p['id']} ({label}): snapped {angle:.1f}° to {snapped.tolist()}"
        )

    logger.info(
        f"Normal snapping ({normal_mode}): {snapped_walls} walls, {snapped_horiz} floor/ceiling, "
        f"{skipped} skipped, {len(wall_axes)} candidate axes"
    )
    return {
        "snapped_walls": snapped_walls,
        "snapped_horiz": snapped_horiz,
        "skipped": skipped,
        "normal_mode": normal_mode,
        "num_axes": len(wall_axes),
        "wall_axes": [a.tolist() for a in wall_axes],
    }
