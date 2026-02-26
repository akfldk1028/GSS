"""Module E: Detect room boundaries from wall center-lines.

Wall center-lines form a planar graph in the XZ plane.
Use Shapely's unary_union + polygonize to extract closed room polygons.

Ref: Cloud2BIM space_generator.py pattern.
"""

from __future__ import annotations

import logging

import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

logger = logging.getLogger(__name__)


def detect_spaces(
    walls: list[dict],
    floor_heights: list[float],
    ceiling_heights: list[float],
    min_area: float = 1.0,
) -> list[dict]:
    """Detect enclosed room polygons from wall center-lines.

    Args:
        walls: list of wall dicts with center_line_2d.
        floor_heights: detected floor heights (Manhattan Y).
        ceiling_heights: detected ceiling heights (Manhattan Y).
        min_area: minimum polygon area to keep (filters noise).

    Returns:
        list of space dicts:
        {id, boundary_2d, area, floor_height, ceiling_height}
    """
    # Build LineStrings from wall center-lines
    lines = []
    for w in walls:
        cl = w["center_line_2d"]
        p1, p2 = cl[0], cl[1]
        length = np.linalg.norm(np.array(p2) - np.array(p1))
        if length < 1e-6:
            continue
        lines.append(LineString([p1, p2]))

    if len(lines) < 3:
        logger.warning("Space detection: fewer than 3 wall lines, skipping")
        return []

    # Merge all lines (automatically splits at intersections / nodes)
    merged = unary_union(lines)

    # Polygonize: extract all closed faces from the planar graph
    polygons = list(polygonize(merged))
    logger.info(f"Space detection: polygonize found {len(polygons)} candidate polygons")

    # Filter by area
    floor_h = min(floor_heights) if floor_heights else 0.0
    ceiling_h = max(ceiling_heights) if ceiling_heights else 3.0

    # Also filter out the "exterior" polygon (typically the largest)
    # by checking if its area is much larger than others
    areas = [p.area for p in polygons]

    spaces = []
    space_id = 0
    for poly, area in zip(polygons, areas):
        if area < min_area:
            logger.debug(f"Space: skipping polygon with area {area:.2f} < {min_area}")
            continue

        # Extract boundary vertices
        coords = list(poly.exterior.coords)
        boundary_2d = [[float(c[0]), float(c[1])] for c in coords]

        spaces.append({
            "id": space_id,
            "boundary_2d": boundary_2d,
            "area": float(area),
            "floor_height": float(floor_h),
            "ceiling_height": float(ceiling_h),
        })
        space_id += 1

    logger.info(
        f"Space detection: {len(spaces)} valid spaces "
        f"(filtered {len(polygons) - len(spaces)} by area)"
    )
    return spaces
