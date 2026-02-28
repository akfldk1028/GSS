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

    # Group into storeys
    stats["storeys"] = _group_storeys(
        stats.get("floor_heights", []),
        stats.get("ceiling_heights", []),
    )

    return stats


def _group_storeys(
    floor_heights: list[float],
    ceiling_heights: list[float],
) -> list[dict]:
    """Pair floor and ceiling heights into storey definitions.

    Algorithm:
    1. Sort all floor and ceiling heights
    2. Greedily match each floor to the nearest ceiling above it
    3. Name storeys sequentially (Ground Floor, 1st Floor, ...)

    Returns:
        List of storey dicts: [{name, floor_height, ceiling_height, elevation}, ...]
        Sorted by elevation (lowest first).
    """
    if not floor_heights and not ceiling_heights:
        return []

    # Handle single-storey (most common case)
    if len(floor_heights) <= 1 and len(ceiling_heights) <= 1:
        floor_h = floor_heights[0] if floor_heights else 0.0
        ceiling_h = ceiling_heights[0] if ceiling_heights else floor_h + 3.0
        return [{
            "name": "Ground Floor",
            "floor_height": float(floor_h),
            "ceiling_height": float(ceiling_h),
            "elevation": float(floor_h),
        }]

    # Multi-storey: pair floor_i with ceiling_j where ceiling_j > floor_i
    floors_sorted = sorted(floor_heights)
    ceilings_sorted = sorted(ceiling_heights)

    storeys = []
    used_ceilings: set[int] = set()

    for floor_h in floors_sorted:
        best_ceiling_idx = None
        best_ceiling_h = None
        best_gap = float("inf")

        for ci, ceiling_h in enumerate(ceilings_sorted):
            if ci in used_ceilings:
                continue
            gap = ceiling_h - floor_h
            if gap > 0.5 and gap < best_gap:  # ceiling must be above floor by >0.5
                best_gap = gap
                best_ceiling_idx = ci
                best_ceiling_h = ceiling_h

        if best_ceiling_idx is not None:
            used_ceilings.add(best_ceiling_idx)
            storeys.append({
                "floor_height": float(floor_h),
                "ceiling_height": float(best_ceiling_h),
                "elevation": float(floor_h),
            })

    # If we couldn't pair floors, try to infer from ceiling gaps
    if not storeys:
        # Fallback: treat all as single storey
        floor_h = min(floors_sorted) if floors_sorted else 0.0
        ceiling_h = max(ceilings_sorted) if ceilings_sorted else floor_h + 3.0
        storeys = [{
            "floor_height": float(floor_h),
            "ceiling_height": float(ceiling_h),
            "elevation": float(floor_h),
        }]

    # Sort by elevation and assign names
    storeys.sort(key=lambda s: s["elevation"])

    storey_names = ["Ground Floor"] + [f"Floor {i}" for i in range(1, 100)]
    for i, s in enumerate(storeys):
        s["name"] = storey_names[min(i, len(storey_names) - 1)]

    logger.info(
        f"Storey grouping: {len(storeys)} storeys detected "
        f"({', '.join(s['name'] for s in storeys)})"
    )
    return storeys
