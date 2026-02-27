"""Module B: Snap floor/ceiling heights to consistent clusters.

In Manhattan Y-up space, horizontal planes have height = centroid_y.
Cluster nearby heights and snap each plane to the cluster mean.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _cluster_1d(values: list[float], tolerance: float) -> list[list[int]]:
    """Greedy 1D clustering: group values within tolerance of each other.

    Returns list of clusters, each cluster is a list of indices into `values`.
    """
    if not values:
        return []

    indexed = sorted(enumerate(values), key=lambda x: x[1])
    clusters: list[list[int]] = []
    current_cluster = [indexed[0][0]]
    current_mean = indexed[0][1]

    for i in range(1, len(indexed)):
        idx, val = indexed[i]
        if abs(val - current_mean) <= tolerance:
            current_cluster.append(idx)
            # Update running mean
            current_mean = np.mean([values[j] for j in current_cluster])
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
            current_mean = val
    clusters.append(current_cluster)
    return clusters


def _plane_height(plane: dict) -> float:
    """Get the Y-coordinate (height) of a horizontal plane in Manhattan space.

    For a plane with normal ≈ [0, ±1, 0] and d:
        n·p + d = 0  →  ±y + d = 0  →  y = ∓d
    More generally: height = -d / normal_y
    """
    ny = plane["normal"][1]
    if abs(ny) < 1e-6:
        # Not truly horizontal; use boundary centroid fallback
        bnd = plane.get("boundary_3d")
        if bnd is not None and len(bnd) > 0:
            return float(np.asarray(bnd)[:, 1].mean())
        return 0.0
    return -plane["d"] / ny


def snap_heights(planes: list[dict], tolerance: float = 0.5) -> dict:
    """Cluster floor/ceiling planes by height and snap to cluster means.

    Args:
        planes: list of plane dicts (modified in-place), in Manhattan space.
        tolerance: max height difference within a cluster.

    Returns:
        stats dict with floor_height, ceiling_height, etc.
    """
    # Separate floor and ceiling planes
    floor_indices = [i for i, p in enumerate(planes) if p["label"] == "floor"]
    ceiling_indices = [i for i, p in enumerate(planes) if p["label"] == "ceiling"]

    stats: dict = {"floor_heights": [], "ceiling_heights": []}

    for label, indices in [("floor", floor_indices), ("ceiling", ceiling_indices)]:
        if not indices:
            continue

        heights = [_plane_height(planes[i]) for i in indices]
        clusters = _cluster_1d(heights, tolerance)

        for cluster_member_indices in clusters:
            actual_indices = [indices[j] for j in cluster_member_indices]
            cluster_heights = [heights[j] for j in cluster_member_indices]
            mean_height = float(np.mean(cluster_heights))

            for j, pi in zip(cluster_member_indices, actual_indices):
                old_h = heights[j]
                if abs(old_h - mean_height) > 1e-6:
                    p = planes[pi]
                    ny = p["normal"][1]
                    if abs(ny) > 1e-6:
                        p["d"] = -mean_height * ny
                    logger.debug(
                        f"Plane {p['id']} ({label}): height {old_h:.3f} → {mean_height:.3f}"
                    )

            key = f"{label}_heights"
            stats[key].append(mean_height)
            logger.info(
                f"Height cluster ({label}): {len(actual_indices)} planes → "
                f"height={mean_height:.3f}"
            )

    return stats
