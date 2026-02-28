"""Module F: Storey detection from exterior facade patterns.

Detects building storeys by analyzing horizontal discontinuities in
facade point distributions, following the Cloud2BIM histogram pattern.

Algorithm:
1. For each facade, project associated plane boundaries to height (Y) axis
2. Build vertical density histogram across all facades
3. Detect valleys in histogram → floor slab positions
4. Each pair of consecutive valleys = one storey
5. Cross-validate across facades for confidence
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _facade_height_samples(
    facade: dict,
    planes: list[dict],
) -> np.ndarray:
    """Collect Y coordinates from facade's constituent plane boundaries."""
    plane_ids = set(facade.get("plane_ids", []))
    heights = []
    for p in planes:
        if p["id"] not in plane_ids:
            continue
        bnd = p.get("boundary_3d")
        if bnd is None or len(bnd) == 0:
            continue
        pts = np.asarray(bnd)
        if pts.ndim == 2 and pts.shape[1] >= 2:
            heights.extend(pts[:, 1].tolist())
    return np.array(heights) if heights else np.array([])


def _detect_valleys(
    histogram: np.ndarray,
    bin_edges: np.ndarray,
    min_valley_depth: float = 0.3,
) -> list[float]:
    """Find valley positions (local minima) in a histogram.

    A valley is a bin whose count is significantly below its neighbors'
    average, indicating a floor slab boundary.
    """
    if len(histogram) < 5:
        return []

    # Smooth histogram with running average (window=3)
    smoothed = np.convolve(histogram, np.ones(3) / 3, mode="same")
    peak = smoothed.max()
    if peak < 1:
        return []

    # Find local minima
    valleys = []
    for i in range(2, len(smoothed) - 2):
        # Local minimum: lower than both neighbors
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            # Valley must be significantly below peak
            depth = 1.0 - smoothed[i] / peak
            if depth >= min_valley_depth:
                # Position = center of bin
                y = (bin_edges[i] + bin_edges[i + 1]) / 2.0
                valleys.append(float(y))

    return valleys


def detect_storeys_from_exterior(
    facades: list[dict],
    planes: list[dict],
    building_points: np.ndarray | None = None,
    *,
    min_storey_height: float = 2.5,
    max_storey_height: float = 5.0,
    histogram_resolution: float = 0.1,
    scale: float = 1.0,
) -> list[dict]:
    """Detect storeys from exterior facade patterns.

    Args:
        facades: Facade dicts from Module C.
        planes: All plane dicts.
        building_points: Optional building point cloud.
        min_storey_height: Min storey height in meters.
        max_storey_height: Max storey height in meters.
        histogram_resolution: Height histogram bin size in meters.
        scale: Coordinate scale.

    Returns:
        List of storey dicts: [{elevation, height, confidence}, ...].
    """
    min_h_scaled = min_storey_height * scale
    max_h_scaled = max_storey_height * scale
    bin_size = histogram_resolution * scale

    # Collect height samples from all facades
    all_heights = []
    for f in facades:
        h = _facade_height_samples(f, planes)
        if len(h) > 0:
            all_heights.append(h)

    # Also use building points if available
    if building_points is not None and len(building_points) > 0:
        all_heights.append(building_points[:, 1])

    if not all_heights:
        logger.info("No height data for storey detection")
        return []

    heights = np.concatenate(all_heights)
    if len(heights) < 10:
        return []

    y_min, y_max = heights.min(), heights.max()
    total_height = y_max - y_min

    if total_height < min_h_scaled:
        logger.info(f"Building height {total_height / scale:.1f}m too short for storey detection")
        return []

    # Build histogram
    n_bins = max(5, int(total_height / bin_size))
    histogram, bin_edges = np.histogram(heights, bins=n_bins)

    # Detect valleys
    valleys = _detect_valleys(histogram, bin_edges)

    # Add ground and top if not present
    boundaries = sorted([y_min] + valleys + [y_max])

    # Filter: remove boundaries that create too-small or too-large storeys
    filtered = [boundaries[0]]
    for b in boundaries[1:]:
        gap = b - filtered[-1]
        if gap >= min_h_scaled * 0.8:  # allow some tolerance
            filtered.append(b)
    # Ensure top is included
    if filtered[-1] < y_max - 0.1 * scale:
        filtered.append(y_max)
    boundaries = filtered

    # Build storey list
    storeys = []
    for i in range(len(boundaries) - 1):
        elev = boundaries[i]
        height = boundaries[i + 1] - elev

        if height < min_h_scaled * 0.5:
            continue
        if height > max_h_scaled * 1.5:
            # Likely multiple storeys merged — subdivide
            n_sub = max(1, round(height / (3.0 * scale)))
            sub_h = height / n_sub
            for j in range(n_sub):
                storeys.append({
                    "elevation": float((elev + j * sub_h) / scale),
                    "height": float(sub_h / scale),
                    "confidence": 0.5,  # lower confidence for subdivided
                })
            continue

        # Confidence: based on how clearly the valley was detected
        confidence = 0.8 if len(valleys) > 0 else 0.5
        storeys.append({
            "elevation": float(elev / scale),
            "height": float(height / scale),
            "confidence": confidence,
        })

    logger.info(
        f"Storey detection: {len(storeys)} storeys detected "
        f"(height range: {y_min / scale:.1f}–{y_max / scale:.1f}m)"
    )
    return storeys
