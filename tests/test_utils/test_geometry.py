"""Tests for gss.utils.geometry — shared geometry utilities."""

import numpy as np
import pytest


class TestComputeConcaveHull:
    def test_square_points(self):
        """Square arrangement should return a polygon covering all points."""
        from gss.utils.geometry import compute_concave_hull

        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        hull = compute_concave_hull(pts)
        assert hull is not None
        assert hull.geom_type == "Polygon"
        assert hull.area > 0

    def test_l_shape_points(self):
        """L-shaped point set should produce a valid polygon."""
        from gss.utils.geometry import compute_concave_hull

        # L-shape: 6 corner points
        pts = np.array([
            [0, 0], [5, 0], [5, 3],
            [3, 3], [3, 5], [0, 5],
        ], dtype=float)
        hull = compute_concave_hull(pts)
        assert hull is not None
        assert hull.geom_type == "Polygon"
        assert hull.area > 0

    def test_insufficient_points_returns_none(self):
        """Less than 3 points should return None."""
        from gss.utils.geometry import compute_concave_hull

        assert compute_concave_hull(np.array([[0, 0], [1, 1]])) is None
        assert compute_concave_hull(np.array([[0, 0]])) is None
        assert compute_concave_hull(None) is None

    def test_collinear_points(self):
        """Collinear points (no area) should return None or degenerate polygon."""
        from gss.utils.geometry import compute_concave_hull

        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        hull = compute_concave_hull(pts)
        # May return None or a degenerate polygon
        if hull is not None:
            # A line has no area
            assert hull.area < 0.001

    def test_min_area_ratio_guard(self):
        """Overly concave result should fall back to convex hull."""
        from gss.utils.geometry import compute_concave_hull

        # Dense cluster — convex hull is the expected result
        rng = np.random.RandomState(42)
        pts = rng.uniform(0, 10, (50, 2))
        hull = compute_concave_hull(pts, min_area_ratio=0.3)
        assert hull is not None
        assert hull.area > 0

    def test_returns_shapely_polygon(self):
        """Return type should be a shapely Polygon."""
        from gss.utils.geometry import compute_concave_hull

        pts = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float)
        hull = compute_concave_hull(pts)
        assert hull is not None
        assert hasattr(hull, "exterior")
        assert hasattr(hull, "area")
        coords = list(hull.exterior.coords)
        assert len(coords) >= 4  # at least 3 vertices + closing point
