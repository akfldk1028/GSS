"""3D geometry utilities: rotations, coordinate transforms, plane math."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def qvec2rotmat(qvec: list[float] | np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def make_w2c(qvec: list[float], tvec: list[float]) -> np.ndarray:
    """Build 4x4 world-to-camera matrix from COLMAP quaternion + translation."""
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(qvec)
    w2c[:3, 3] = tvec
    return w2c


def make_c2w(qvec: list[float], tvec: list[float]) -> np.ndarray:
    """Build 4x4 camera-to-world matrix from COLMAP quaternion + translation."""
    return np.linalg.inv(make_w2c(qvec, tvec))


def compute_concave_hull(
    points_2d: np.ndarray,
    alpha: float | None = None,
    min_area_ratio: float = 0.3,
):
    """Compute concave hull (alpha shape) of 2D points.

    3-stage fallback: shapely.concave_hull -> alphashape -> ConvexHull.
    Returns a shapely Polygon or None.

    Args:
        points_2d: (N, 2) array of 2D points.
        alpha: Alpha shape parameter (None = auto).
        min_area_ratio: Min ratio of concave/convex area to accept
            (prevents over-concavity).
    """
    if points_2d is None or len(points_2d) < 3:
        return None

    try:
        from shapely.geometry import MultiPoint
        mp = MultiPoint(points_2d.tolist())

        convex = mp.convex_hull
        if convex.geom_type != "Polygon" or convex.is_empty:
            return None

        # Stage 1: Shapely 2.0+ concave_hull
        try:
            from shapely import concave_hull
            if alpha is not None:
                ratio = min(1.0, max(0.0, 1.0 / (alpha + 0.01)))
            else:
                ratio = 0.3  # moderate concavity
            hull = concave_hull(mp, ratio=ratio)
            if hull.is_valid and not hull.is_empty and hull.geom_type == "Polygon":
                if hull.area >= convex.area * min_area_ratio:
                    return hull
        except (ImportError, Exception):
            pass

        # Stage 2: alphashape package
        try:
            import alphashape
            if alpha is None:
                alpha = alphashape.optimizealpha(points_2d)
            hull = alphashape.alphashape(points_2d, alpha)
            if hasattr(hull, "exterior") and not hull.is_empty:
                if hull.area >= convex.area * min_area_ratio:
                    return hull
        except (ImportError, Exception):
            pass

        # Stage 3: Convex hull fallback
        return convex

    except ImportError:
        pass

    # Pure scipy fallback
    try:
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon
        ch = ConvexHull(points_2d)
        vertices = points_2d[ch.vertices]
        return Polygon(vertices)
    except (ImportError, Exception):
        pass

    return None
