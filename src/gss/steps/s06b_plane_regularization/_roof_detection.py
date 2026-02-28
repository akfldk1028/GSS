"""Module H: Roof plane detection and classification.

Detects roof planes from RANSAC-extracted planes based on position and normal.

Algorithm:
1. Find the maximum ceiling height from height stats
2. Identify planes above the ceiling that have inclined normals
3. Classify roof type based on normal direction:
   - Flat roof: normal ≈ [0, ±1, 0] (horizontal, above ceiling)
   - Pitched roof: normal has significant Y component (0.3-0.9) + XZ component
4. Group pitched planes by ridge direction

Roof types:
  - flat: Single horizontal plane above ceiling → IfcSlab(ROOF)
  - shed: Single inclined plane → IfcSlab(ROOF)
  - gable: 2 inclined planes meeting at a ridge
  - hip: 4 inclined planes
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Normal Y component thresholds for roof classification
_MIN_ROOF_TILT = 0.15  # min |ny| to consider as a roof (vs vertical wall)
_MAX_WALL_TILT = 0.85  # max |ny| for inclined roof (above = flat roof)


def detect_roof_planes(
    planes: list[dict],
    ceiling_heights: list[float],
    *,
    height_margin: float = 0.5,
) -> dict:
    """Detect and classify roof planes.

    A plane is classified as a roof if:
    1. Its centroid is above max(ceiling_heights) - height_margin
    2. Its normal has a significant Y component (not a pure wall)
    3. It wasn't already labeled as floor/ceiling

    Args:
        planes: List of plane dicts (modified in-place: label set to "roof").
        ceiling_heights: Ceiling height values from height snapping.
        height_margin: Tolerance below ceiling to still consider as roof.

    Returns:
        Stats dict with roof plane info.
    """
    if not ceiling_heights:
        return {"num_roof_planes": 0, "roof_type": "unknown"}

    max_ceiling = max(ceiling_heights)
    roof_threshold = max_ceiling - height_margin

    roof_indices: list[int] = []
    flat_count = 0
    inclined_count = 0

    for i, p in enumerate(planes):
        if p["label"] in ("floor", "ceiling"):
            continue

        # Check plane height (centroid Y)
        height = _plane_centroid_y(p)
        if height is None or height < roof_threshold:
            continue

        # Check normal: roof planes have significant Y component
        ny = abs(p["normal"][1])
        if ny < _MIN_ROOF_TILT:
            continue  # Too vertical — it's a wall, not a roof

        # This is a roof plane
        roof_indices.append(i)
        p["label"] = "roof"

        if ny > _MAX_WALL_TILT:
            p["roof_type"] = "flat"
            flat_count += 1
        else:
            p["roof_type"] = "inclined"
            inclined_count += 1

    # Classify overall roof type
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
        f"Roof detection: {total} roof planes "
        f"({flat_count} flat, {inclined_count} inclined) → {roof_type}"
    )

    return {
        "num_roof_planes": total,
        "num_flat": flat_count,
        "num_inclined": inclined_count,
        "roof_type": roof_type,
        "roof_plane_ids": [planes[i]["id"] for i in roof_indices],
    }


def _plane_centroid_y(plane: dict) -> float | None:
    """Get centroid Y coordinate of a plane from its boundary."""
    bnd = plane.get("boundary_3d")
    if bnd is not None and len(bnd) > 0:
        pts = np.asarray(bnd)
        if pts.ndim == 2 and pts.shape[1] >= 2:
            return float(pts[:, 1].mean())

    # Fallback: compute from normal and d
    ny = plane["normal"][1]
    if abs(ny) > 0.1:
        return float(-plane["d"] / ny)

    return None
