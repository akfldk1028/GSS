"""Module B: Building vs vegetation/other segmentation.

Uses density-based clustering (DBSCAN) to separate dense building surfaces
from sparse vegetation and noise. No ML required.

Algorithm:
1. Filter points above ground and below max_building_height
2. DBSCAN clustering → building = dense clusters
3. Normal consistency filter: buildings have coherent normals, vegetation is random
4. Plane proximity: points near RANSAC planes are likely building
5. Largest qualifying cluster = primary building
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _try_dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray | None:
    """Try DBSCAN with sklearn, fall back to scipy if not available."""
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
        return labels
    except (ImportError, ValueError):
        pass

    # Fallback: scipy-based simple clustering
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        n = len(points)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            neighbors = tree.query_ball_point(points[i], eps)
            if len(neighbors) < min_samples:
                continue
            # BFS expansion
            queue = list(neighbors)
            visited[i] = True
            labels[i] = cluster_id
            head = 0
            while head < len(queue):
                j = queue[head]
                head += 1
                if visited[j]:
                    continue
                visited[j] = True
                labels[j] = cluster_id
                j_neighbors = tree.query_ball_point(points[j], eps)
                if len(j_neighbors) >= min_samples:
                    queue.extend(j_neighbors)
            cluster_id += 1
            if cluster_id > 100:  # safety limit
                break

        return labels
    except ImportError:
        logger.warning("Neither sklearn nor scipy available for DBSCAN")
        return None


def segment_building_points(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    planes: list[dict] | None = None,
    ground_plane: dict | None = None,
    *,
    max_building_height: float = 30.0,
    dbscan_eps: float = 0.5,
    min_cluster_size: int = 100,
    scale: float = 1.0,
) -> np.ndarray:
    """Segment points into building / vegetation / other.

    Args:
        points: (N, 3) point cloud.
        normals: (N, 3) per-point normals (optional).
        planes: RANSAC planes (optional, for proximity scoring).
        ground_plane: Ground plane dict (for height filtering).
        max_building_height: Max height in meters above ground.
        dbscan_eps: DBSCAN neighborhood radius in meters.
        min_cluster_size: Min cluster points.
        scale: Coordinate scale.

    Returns:
        Labels array (N,): 0=building, 1=vegetation, 2=other.
    """
    n = len(points)
    labels = np.full(n, 2, dtype=int)  # default: other

    # Height filtering
    if ground_plane is not None:
        gn = np.asarray(ground_plane["normal"], dtype=float)
        gd = float(ground_plane["d"])
        heights = points @ gn + gd  # signed distance from ground
        height_mask = (heights > 0) & (heights < max_building_height * scale)
    else:
        # Use Y coordinate as height proxy
        y = points[:, 1]
        if len(y) == 0:
            return labels
        y_min = y.min()
        height_mask = (y - y_min) < max_building_height * scale

    candidate_idx = np.where(height_mask)[0]
    if len(candidate_idx) < min_cluster_size:
        logger.warning(f"Too few candidate points ({len(candidate_idx)}) for building segmentation")
        return labels

    candidate_pts = points[candidate_idx]

    # DBSCAN clustering
    eps_scaled = dbscan_eps * scale
    cluster_labels = _try_dbscan(candidate_pts, eps_scaled, min_cluster_size)

    if cluster_labels is None:
        # No clustering available — mark all candidates as building
        labels[candidate_idx] = 0
        return labels

    # Find cluster sizes
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)  # noise

    if not unique_labels:
        logger.warning("No clusters found by DBSCAN")
        return labels

    cluster_sizes = {
        lbl: np.sum(cluster_labels == lbl)
        for lbl in unique_labels
    }

    # Score each cluster: building vs vegetation
    for lbl, size in cluster_sizes.items():
        cluster_mask = cluster_labels == lbl
        cluster_pts_idx = candidate_idx[cluster_mask]

        # Score components
        score = 0.0

        # Size score: larger = more likely building
        if size > min_cluster_size * 5:
            score += 0.3

        # Plane proximity: if planes exist, check how many cluster points
        # are near a RANSAC plane
        if planes:
            near_plane = 0
            for p in planes:
                pn = np.asarray(p["normal"], dtype=float)
                pd = float(p["d"])
                dists = np.abs(points[cluster_pts_idx] @ pn + pd)
                near_plane += np.sum(dists < 0.5 * scale)
            proximity_ratio = near_plane / max(size, 1)
            score += 0.4 * min(proximity_ratio, 1.0)

        # Normal consistency: building surfaces have coherent normals
        if normals is not None and len(cluster_pts_idx) > 10:
            cluster_normals = normals[cluster_pts_idx]
            # Compute normal variance (low = coherent = building)
            mean_normal = cluster_normals.mean(axis=0)
            mean_norm = np.linalg.norm(mean_normal)
            if mean_norm > 0.1:
                score += 0.3 * mean_norm  # high consistency → high score

        # Classify: score >= 0.3 → building, else vegetation
        if score >= 0.3:
            labels[cluster_pts_idx] = 0  # building
        else:
            labels[cluster_pts_idx] = 1  # vegetation

    # Noise points from DBSCAN
    noise_mask = cluster_labels == -1
    labels[candidate_idx[noise_mask]] = 2  # other

    building_count = np.sum(labels == 0)
    veg_count = np.sum(labels == 1)
    other_count = np.sum(labels == 2)
    logger.info(
        f"Building segmentation: {building_count} building, "
        f"{veg_count} vegetation, {other_count} other"
    )
    return labels
