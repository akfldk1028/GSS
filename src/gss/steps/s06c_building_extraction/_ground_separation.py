"""Module A: Ground plane separation.

Identifies the ground plane as the lowest, widest horizontal plane.
Labels it in planes.json and optionally separates ground points from
building points.

Algorithm:
1. Select horizontal planes (|ny| > threshold)
2. Compute each plane's XZ extent (bounding box area)
3. The lowest horizontal plane with extent > min_ground_extent = ground
4. Tag it label="ground"
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _plane_centroid_y(plane: dict) -> float | None:
    """Get centroid Y coordinate from boundary or normal/d."""
    bnd = plane.get("boundary_3d")
    if bnd is not None and len(bnd) > 0:
        pts = np.asarray(bnd)
        if pts.ndim == 2 and pts.shape[1] >= 2:
            return float(pts[:, 1].mean())
    ny = plane["normal"][1] if not isinstance(plane["normal"], np.ndarray) else plane["normal"][1]
    if abs(ny) > 0.1:
        return float(-plane["d"] / ny)
    return None


def _plane_xz_extent(plane: dict) -> float:
    """Compute XZ bounding-box area of a plane's boundary."""
    bnd = plane.get("boundary_3d")
    if bnd is None or len(bnd) == 0:
        return 0.0
    pts = np.asarray(bnd)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return 0.0
    x_range = pts[:, 0].max() - pts[:, 0].min()
    z_range = pts[:, 2].max() - pts[:, 2].min()
    return float(max(x_range, z_range))


def detect_ground_plane(
    planes: list[dict],
    *,
    normal_threshold: float = 0.8,
    min_ground_extent: float = 10.0,
    scale: float = 1.0,
) -> dict | None:
    """Detect the ground plane from a list of planes.

    The ground plane is the lowest horizontal plane whose XZ extent
    exceeds min_ground_extent (in meters, scaled).

    Args:
        planes: List of plane dicts (modified in-place: label → "ground").
        normal_threshold: Min |ny| to consider horizontal.
        min_ground_extent: Min XZ extent in meters.
        scale: Coordinate scale (scene_units / meter).

    Returns:
        The ground plane dict, or None if not found.
    """
    min_extent_scaled = min_ground_extent * scale

    # Find horizontal planes
    candidates: list[tuple[int, float, float]] = []  # (index, y, extent)
    for i, p in enumerate(planes):
        n = np.asarray(p["normal"], dtype=float)
        if abs(n[1]) < normal_threshold:
            continue
        y = _plane_centroid_y(p)
        if y is None:
            continue
        extent = _plane_xz_extent(p)
        candidates.append((i, y, extent))

    if not candidates:
        logger.warning("No horizontal planes found for ground detection")
        return None

    # Sort by Y (lowest first), then by extent (largest first)
    candidates.sort(key=lambda c: (c[1], -c[2]))

    # Find lowest horizontal plane with sufficient extent
    for idx, y, extent in candidates:
        if extent >= min_extent_scaled:
            planes[idx]["label"] = "ground"
            logger.info(
                f"Ground plane detected: plane {planes[idx]['id']}, "
                f"y={y:.2f}, extent={extent:.1f} (min={min_extent_scaled:.1f})"
            )
            return planes[idx]

    # Fallback: if no plane is wide enough, use the lowest horizontal plane
    # but only if extent is at least 20% of the minimum
    idx, y, extent = candidates[0]
    if extent >= min_extent_scaled * 0.2:
        planes[idx]["label"] = "ground"
        logger.info(
            f"Ground plane (fallback): plane {planes[idx]['id']}, "
            f"y={y:.2f}, extent={extent:.1f} (below min {min_extent_scaled:.1f})"
        )
        return planes[idx]

    logger.warning(
        f"No ground plane found (best candidate extent={candidates[0][2]:.1f}, "
        f"min required={min_extent_scaled:.1f})"
    )
    return None


def separate_ground_points(
    points: np.ndarray,
    ground_plane: dict,
    tolerance: float = 0.3,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate points into building and ground sets.

    Args:
        points: (N, 3) point cloud.
        ground_plane: Plane dict with normal and d.
        tolerance: Height tolerance in meters.
        scale: Coordinate scale.

    Returns:
        (building_points, ground_points) — both (M, 3) arrays.
    """
    n = np.asarray(ground_plane["normal"], dtype=float)
    d = float(ground_plane["d"])
    tol_scaled = tolerance * scale

    # Signed distance from each point to ground plane
    dists = points @ n + d
    ground_mask = np.abs(dists) <= tol_scaled

    ground_pts = points[ground_mask]
    building_pts = points[~ground_mask]

    logger.info(
        f"Ground separation: {len(ground_pts)} ground, {len(building_pts)} building "
        f"(tolerance={tol_scaled:.2f})"
    )
    return building_pts, ground_pts
