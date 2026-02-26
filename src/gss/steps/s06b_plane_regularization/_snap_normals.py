"""Module A: Snap plane normals to Manhattan axes.

In Manhattan-aligned Y-up space:
- Wall normals → nearest ±X or ±Z axis
- Floor/ceiling normals → nearest ±Y axis
- Recompute d to keep plane passing through same region
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Candidate axes for each plane type (Manhattan Y-up)
_WALL_AXES = [
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
]
_HORIZ_AXES = [
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
]


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


def snap_normals(planes: list[dict], threshold_deg: float = 20.0) -> dict:
    """Snap plane normals to nearest Manhattan axis.

    Args:
        planes: list of plane dicts (modified in-place), in Manhattan space.
        threshold_deg: max angle deviation to allow snapping.

    Returns:
        stats dict with counts.
    """
    snapped_walls = 0
    snapped_horiz = 0
    skipped = 0

    for p in planes:
        label = p["label"]
        normal = np.asarray(p["normal"], dtype=float)

        if label == "wall":
            candidates = _WALL_AXES
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
        f"Normal snapping: {snapped_walls} walls, {snapped_horiz} floor/ceiling, "
        f"{skipped} skipped"
    )
    return {"snapped_walls": snapped_walls, "snapped_horiz": snapped_horiz, "skipped": skipped}
