"""Module I: Detect columns from narrow wall-like plane pairs.

Reclassifies walls as columns when:
1. Both XZ extents are < max_column_width (narrow in both directions)
2. Aspect ratio min(length, thickness) / height < column_aspect_ratio
3. Nearly square cross-section → round column; otherwise rectangular

Output: columns.json + filtered walls list.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def detect_columns(
    walls: list[dict],
    scale: float = 1.0,
    *,
    max_column_width: float = 1.0,
    column_aspect_ratio: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    """Detect and reclassify narrow walls as columns.

    Args:
        walls: List of wall dicts from wall_thickness module.
        scale: Scene units / meter.
        max_column_width: Max extent in both axes to classify as column (meters).
        column_aspect_ratio: Max min(length, thickness) / height for columns.

    Returns:
        (columns, remaining_walls) where columns is a list of column dicts
        and remaining_walls are walls that were NOT reclassified.
    """
    columns: list[dict] = []
    remaining: list[dict] = []
    max_width_scene = max_column_width * scale

    for w in walls:
        cl = w["center_line_2d"]
        if len(cl) < 2:
            remaining.append(w)
            continue

        hr = w.get("height_range", [0, 0])
        height = hr[1] - hr[0]
        thickness = w.get("thickness", 0.2 * scale)

        # Compute wall length in XZ plane
        p1 = np.array(cl[0], dtype=float)
        p2 = np.array(cl[-1], dtype=float)
        wall_length = float(np.linalg.norm(p2 - p1))

        # Check extent criteria (both dimensions must be small)
        if wall_length > max_width_scene or thickness > max_width_scene:
            remaining.append(w)
            continue

        # Check aspect ratio
        min_dim = min(wall_length, thickness) / max(scale, 1e-6)
        height_m = height / max(scale, 1e-6)
        if height_m < 0.1:
            remaining.append(w)
            continue

        ratio = min_dim / height_m
        if ratio > column_aspect_ratio:
            remaining.append(w)
            continue

        # It's a column! Determine type
        length_m = wall_length / max(scale, 1e-6)
        thickness_m = thickness / max(scale, 1e-6)
        squareness = min(length_m, thickness_m) / max(length_m, thickness_m, 1e-6)

        if squareness > 0.7:
            column_type = "round"
            center = ((p1 + p2) / 2.0).tolist()
            radius = (length_m + thickness_m) / 4.0  # average of half-dimensions
        else:
            column_type = "rectangular"
            center = ((p1 + p2) / 2.0).tolist()
            radius = None

        col = {
            "id": len(columns),
            "column_type": column_type,
            "center_2d": center,
            "height_range": list(hr),
            "source_wall_id": w.get("id"),
            "plane_ids": w.get("plane_ids", []),
        }
        if column_type == "round":
            col["radius"] = radius
        else:
            col["width"] = length_m
            col["depth"] = thickness_m
            # Direction for rectangular orientation
            if wall_length > 1e-6:
                direction = ((p2 - p1) / wall_length).tolist()
            else:
                direction = [1.0, 0.0]
            col["direction"] = direction

        columns.append(col)
        logger.debug(
            f"Wall {w.get('id')} reclassified as {column_type} column "
            f"(length={length_m:.2f}m, thickness={thickness_m:.2f}m, "
            f"height={height_m:.2f}m, ratio={ratio:.2f})"
        )

    if columns:
        logger.info(
            f"Column detection: {len(columns)} columns detected "
            f"({len(walls)} walls -> {len(remaining)} walls + {len(columns)} columns)"
        )

    return columns, remaining
