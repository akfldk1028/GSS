"""Module D: Building footprint extraction.

Extracts a 2D building footprint polygon using alpha shapes (concave hull)
instead of convex hull, correctly handling L-shaped, T-shaped, and irregular
building forms.

Algorithm:
1. Collect XZ coordinates from facade base-lines or wall center-lines
2. Compute concave hull (alpha shape) for non-convex footprints
3. Simplify with Douglas-Peucker
4. Compute oriented bounding box
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _collect_footprint_points(
    facades: list[dict] | None = None,
    walls: list[dict] | None = None,
    planes: list[dict] | None = None,
    building_points: np.ndarray | None = None,
) -> np.ndarray | None:
    """Collect 2D (XZ) points for footprint computation.

    Priority: building_points > facades > walls > planes.
    """
    # 1. From building points (project to XZ)
    if building_points is not None and len(building_points) >= 3:
        return building_points[:, [0, 2]]  # XZ

    # 2. From facades (use plane boundaries)
    if facades:
        pts = []
        for f in facades:
            # Use plane_ids to find corresponding planes
            if "boundary_xz" in f:
                pts.extend(f["boundary_xz"])
        if pts:
            return np.array(pts)

    # 3. From walls (center-line endpoints)
    if walls:
        pts = []
        for w in walls:
            cl = w.get("center_line_2d")
            if cl and len(cl) == 2:
                pts.append(cl[0])
                pts.append(cl[1])
        if pts:
            return np.array(pts)

    # 4. From vertical planes (boundary XZ projection)
    if planes:
        pts = []
        for p in planes:
            if p.get("label") not in ("wall",):
                continue
            bnd = p.get("boundary_3d")
            if bnd is None or len(bnd) == 0:
                continue
            arr = np.asarray(bnd)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                pts.append(arr[:, [0, 2]])  # XZ
        if pts:
            return np.vstack(pts)

    return None


def _compute_concave_hull(points_2d: np.ndarray, alpha: float | None = None):
    """Compute concave hull using shapely or alphashape.

    Falls back to convex hull if concave hull fails.
    """
    try:
        from shapely.geometry import MultiPoint
        mp = MultiPoint(points_2d.tolist())

        # Convex hull as baseline
        convex = mp.convex_hull
        if convex.geom_type != "Polygon" or convex.is_empty:
            return None

        # Try Shapely 2.0+ concave_hull
        try:
            from shapely import concave_hull
            if alpha is not None:
                ratio = min(1.0, max(0.0, 1.0 / (alpha + 0.01)))
            else:
                ratio = 0.3  # moderate concavity
            hull = concave_hull(mp, ratio=ratio)
            if hull.is_valid and not hull.is_empty and hull.geom_type == "Polygon":
                # Guard: if concave hull is too small compared to convex hull,
                # the input points likely form a perimeter (wall boundaries)
                # rather than a filled region. Fall back to convex hull.
                if hull.area >= convex.area * 0.3:
                    return hull
        except (ImportError, Exception):
            pass

        # Try alphashape package
        try:
            import alphashape
            if alpha is None:
                alpha = alphashape.optimizealpha(points_2d)
            hull = alphashape.alphashape(points_2d, alpha)
            if hasattr(hull, "exterior") and not hull.is_empty:
                if hull.area >= convex.area * 0.3:
                    return hull
        except (ImportError, Exception):
            pass

        # Fallback: convex hull
        return convex

    except ImportError:
        pass

    # Pure numpy convex hull fallback
    try:
        from scipy.spatial import ConvexHull
        ch = ConvexHull(points_2d)
        vertices = points_2d[ch.vertices]
        from shapely.geometry import Polygon
        return Polygon(vertices)
    except (ImportError, Exception):
        pass

    return None


def _oriented_bbox(polygon) -> dict | None:
    """Compute oriented bounding box of a shapely polygon."""
    try:
        obb = polygon.minimum_rotated_rectangle
        coords = np.array(obb.exterior.coords)
        # Compute dimensions
        d1 = np.linalg.norm(coords[1] - coords[0])
        d2 = np.linalg.norm(coords[2] - coords[1])
        width, height = min(d1, d2), max(d1, d2)
        center = np.array(obb.centroid.coords[0])
        # Angle of longer edge
        if d1 > d2:
            edge = coords[1] - coords[0]
        else:
            edge = coords[2] - coords[1]
        angle_rad = float(np.arctan2(edge[1], edge[0]))
        angle_deg = float(np.degrees(angle_rad))
        return {
            "center": center.tolist(),
            "dimensions": [float(width), float(height)],
            "angle_deg": angle_deg,
        }
    except Exception:
        return None


def extract_footprint(
    building_points: np.ndarray | None = None,
    facades: list[dict] | None = None,
    walls: list[dict] | None = None,
    planes: list[dict] | None = None,
    *,
    alpha: float | None = None,
    simplify_tolerance: float = 0.1,
    scale: float = 1.0,
) -> dict | None:
    """Extract 2D building footprint.

    Uses concave hull (alpha shape) for non-convex buildings.

    Args:
        building_points: Optional (N,3) building point cloud.
        facades: Optional facade list from Module C.
        walls: Optional wall list.
        planes: Optional plane list.
        alpha: Alpha shape parameter (None = auto).
        simplify_tolerance: Douglas-Peucker tolerance in meters.
        scale: Coordinate scale.

    Returns:
        Dict with polygon_2d, area, oriented_bbox, or None.
    """
    points_2d = _collect_footprint_points(facades, walls, planes, building_points)
    if points_2d is None or len(points_2d) < 3:
        logger.warning("Insufficient points for footprint extraction")
        return None

    # Remove duplicates
    points_2d = np.unique(points_2d, axis=0)
    if len(points_2d) < 3:
        logger.warning("Too few unique points for footprint")
        return None

    # Compute concave hull
    polygon = _compute_concave_hull(points_2d, alpha=alpha)
    if polygon is None:
        logger.warning("Concave hull computation failed")
        return None

    # Simplify
    tol_scaled = simplify_tolerance * scale
    simplified = polygon.simplify(tol_scaled, preserve_topology=True)
    if simplified.is_valid and not simplified.is_empty and simplified.geom_type == "Polygon":
        polygon = simplified

    # Extract coordinates
    coords = list(polygon.exterior.coords)
    area = float(polygon.area) / (scale * scale)  # back to meters²

    # Oriented bounding box
    obb = _oriented_bbox(polygon)
    if obb:
        obb["dimensions"] = [d / scale for d in obb["dimensions"]]
        obb["center"] = [c for c in obb["center"]]  # keep in scene units

    result = {
        "polygon_2d": [list(c) for c in coords],
        "area": area,
        "oriented_bbox": obb,
    }

    logger.info(
        f"Footprint extracted: {len(coords) - 1} vertices, "
        f"area={area:.1f} m², convex={'convex' if polygon.convex_hull.area == polygon.area else 'concave'}"
    )
    return result
