"""Module E: Detect room boundaries from wall center-lines.

Wall center-lines form a planar graph in the XZ plane.
Use Shapely's unary_union + polygonize to extract closed room polygons.

Includes snap+buffer gap closing and convex hull fallback.

Ref: Cloud2BIM space_generator.py pattern.
"""

from __future__ import annotations

import logging

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

logger = logging.getLogger(__name__)


def _snap_endpoints(lines: list[LineString], tolerance: float) -> list[LineString]:
    """Snap nearby line endpoints to close small gaps.

    When two endpoints are within tolerance, move both to their midpoint.
    This helps polygonize succeed when walls almost-but-don't-quite meet.
    """
    if tolerance <= 0 or len(lines) < 2:
        return lines

    # Collect all endpoints
    endpoints = []
    for i, line in enumerate(lines):
        coords = list(line.coords)
        endpoints.append((np.array(coords[0]), i, 0))
        endpoints.append((np.array(coords[-1]), i, 1))

    # Find close pairs and snap
    snapped = set()
    coords_lists = [list(line.coords) for line in lines]

    for a in range(len(endpoints)):
        if a in snapped:
            continue
        pa, li_a, ei_a = endpoints[a]
        cluster = [(pa, li_a, ei_a)]

        for b in range(a + 1, len(endpoints)):
            if b in snapped:
                continue
            pb, li_b, ei_b = endpoints[b]
            if li_a == li_b:
                continue  # same line
            if np.linalg.norm(pa - pb) <= tolerance:
                cluster.append((pb, li_b, ei_b))
                snapped.add(b)

        if len(cluster) > 1:
            centroid = np.mean([c[0] for c in cluster], axis=0)
            for _, li, ei in cluster:
                idx = 0 if ei == 0 else -1
                coords_lists[li][idx] = tuple(centroid)
            snapped.add(a)

    return [LineString(c) for c in coords_lists if len(c) >= 2]


def _convex_hull_fallback(
    lines: list[LineString], floor_h: float, ceiling_h: float, min_area: float = 0.0,
) -> list[dict]:
    """Fallback: use convex hull of all wall endpoints as single room boundary."""
    all_pts = []
    for line in lines:
        all_pts.extend(line.coords)

    if len(all_pts) < 3:
        return []

    from shapely.geometry import MultiPoint
    hull = MultiPoint(all_pts).convex_hull

    if isinstance(hull, Polygon) and hull.area > min_area:
        coords = list(hull.exterior.coords)
        boundary_2d = [[float(c[0]), float(c[1])] for c in coords]
        logger.info(f"Convex hull fallback: area={hull.area:.2f}")
        return [{
            "id": 0,
            "boundary_2d": boundary_2d,
            "area": float(hull.area),
            "floor_height": float(floor_h),
            "ceiling_height": float(ceiling_h),
        }]
    return []


def detect_spaces(
    walls: list[dict],
    floor_heights: list[float],
    ceiling_heights: list[float],
    min_area: float = 1.0,
    snap_tolerance: float = 0.0,
    scale: float = 1.0,
) -> list[dict]:
    """Detect enclosed room polygons from wall center-lines.

    Args:
        walls: list of wall dicts with center_line_2d.
        floor_heights: detected floor heights (Manhattan Y).
        ceiling_heights: detected ceiling heights (Manhattan Y).
        min_area: minimum polygon area to keep (already scaled).
        snap_tolerance: tolerance for snapping nearby endpoints (already scaled).
        scale: scene_units / meter (for fallback ceiling height).

    Returns:
        list of space dicts:
        {id, boundary_2d, area, floor_height, ceiling_height}
    """
    # Build LineStrings from wall center-lines (supports N-point polylines)
    lines = []
    for w in walls:
        cl = w["center_line_2d"]
        if len(cl) < 2:
            continue
        # Compute total length
        total = 0.0
        for k in range(len(cl) - 1):
            total += np.linalg.norm(np.array(cl[k + 1]) - np.array(cl[k]))
        if total < 1e-6:
            continue
        lines.append(LineString(cl))

    if len(lines) < 3:
        logger.warning("Space detection: fewer than 3 wall lines, skipping")
        return []

    floor_h = min(floor_heights) if floor_heights else 0.0
    ceiling_h = max(ceiling_heights) if ceiling_heights else floor_h + 3.0 * scale

    # Snap nearby endpoints to close small gaps
    if snap_tolerance > 0:
        lines = _snap_endpoints(lines, snap_tolerance)

    # Merge all lines (automatically splits at intersections / nodes)
    merged = unary_union(lines)

    # Polygonize: extract all closed faces from the planar graph
    polygons = list(polygonize(merged))
    logger.info(f"Space detection: polygonize found {len(polygons)} candidate polygons")

    # If polygonize found nothing, try convex hull fallback
    if not polygons:
        logger.info("Space detection: polygonize found no polygons, trying convex hull fallback")
        return _convex_hull_fallback(lines, floor_h, ceiling_h, min_area)

    # Filter by area
    spaces = []
    space_id = 0
    for poly in polygons:
        area = poly.area
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

    # If all polygons were filtered by area, try convex hull
    if not spaces and polygons:
        logger.info("Space detection: all polygons too small, trying convex hull fallback")
        return _convex_hull_fallback(lines, floor_h, ceiling_h, min_area)

    logger.info(
        f"Space detection: {len(spaces)} valid spaces "
        f"(filtered {len(polygons) - len(spaces)} by area)"
    )
    return spaces
