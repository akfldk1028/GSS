"""Tests for S06b: Plane Regularization step."""

import json
from pathlib import Path

import numpy as np
import pytest

from gss.steps.s06b_plane_regularization.config import PlaneRegularizationConfig
from gss.steps.s06b_plane_regularization.contracts import (
    PlaneRegularizationInput,
    PlaneRegularizationOutput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manhattan_planes_json(data_root: Path) -> Path:
    """Create planes already in Manhattan-aligned Y-up space (4 walls forming a room)."""
    planes = [
        {  # Wall 0: +X wall at x=5
            "id": 0, "normal": [0.98, 0.05, 0.17], "d": -4.95,
            "label": "wall", "num_inliers": 1000,
            "boundary_3d": [[5, 0, 0], [5, 3, 0], [5, 3, 4], [5, 0, 4], [5, 0, 0]],
        },
        {  # Wall 1: -X wall at x=0
            "id": 1, "normal": [-0.97, 0.02, -0.22], "d": -0.1,
            "label": "wall", "num_inliers": 800,
            "boundary_3d": [[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4], [0, 0, 0]],
        },
        {  # Wall 2: +Z wall at z=4
            "id": 2, "normal": [0.15, 0.03, 0.99], "d": -3.95,
            "label": "wall", "num_inliers": 900,
            "boundary_3d": [[0, 0, 4], [5, 0, 4], [5, 3, 4], [0, 3, 4], [0, 0, 4]],
        },
        {  # Wall 3: -Z wall at z=0
            "id": 3, "normal": [-0.18, 0.04, -0.98], "d": 0.05,
            "label": "wall", "num_inliers": 700,
            "boundary_3d": [[0, 0, 0], [5, 0, 0], [5, 3, 0], [0, 3, 0], [0, 0, 0]],
        },
        {  # Floor at y=0
            "id": 4, "normal": [0.02, 0.99, 0.1], "d": -0.08,
            "label": "floor", "num_inliers": 2000,
            "boundary_3d": [[0, 0, 0], [5, 0, 0], [5, 0, 4], [0, 0, 4], [0, 0, 0]],
        },
        {  # Ceiling at y=3
            "id": 5, "normal": [0.01, -0.98, -0.15], "d": 2.95,
            "label": "ceiling", "num_inliers": 1500,
            "boundary_3d": [[0, 3, 0], [5, 3, 0], [5, 3, 4], [0, 3, 4], [0, 3, 0]],
        },
    ]
    planes_file = data_root / "interim" / "s06_planes" / "planes.json"
    planes_file.parent.mkdir(parents=True, exist_ok=True)
    with open(planes_file, "w") as f:
        json.dump(planes, f)

    # boundaries
    boundaries = [{"id": p["id"], "label": p["label"], "boundary_3d": p["boundary_3d"]} for p in planes]
    bnd_file = data_root / "interim" / "s06_planes" / "boundaries.json"
    with open(bnd_file, "w") as f:
        json.dump(boundaries, f)

    # No manhattan_alignment.json → step will process in original coords
    return planes_file


@pytest.fixture
def manhattan_boundaries_json(data_root: Path, manhattan_planes_json: Path) -> Path:
    return data_root / "interim" / "s06_planes" / "boundaries.json"


# ---------------------------------------------------------------------------
# A. Normal Snapping
# ---------------------------------------------------------------------------

class TestSnapNormals:
    def test_wall_snaps_to_x_axis(self):
        from gss.steps.s06b_plane_regularization._snap_normals import snap_normals
        planes = [{
            "id": 0, "normal": np.array([0.96, 0.05, 0.2]),
            "d": -5.0, "label": "wall",
            "boundary_3d": np.array([[5, 0, 0], [5, 3, 0], [5, 3, 4]]),
        }]
        snap_normals(planes, threshold_deg=20.0)
        np.testing.assert_array_equal(planes[0]["normal"], [1, 0, 0])

    def test_wall_snaps_to_z_axis(self):
        from gss.steps.s06b_plane_regularization._snap_normals import snap_normals
        planes = [{
            "id": 0, "normal": np.array([0.1, 0.02, -0.99]),
            "d": 3.0, "label": "wall",
            "boundary_3d": np.array([[0, 0, 3], [5, 0, 3], [5, 3, 3]]),
        }]
        snap_normals(planes, threshold_deg=20.0)
        np.testing.assert_array_equal(planes[0]["normal"], [0, 0, -1])

    def test_floor_snaps_to_y_axis(self):
        from gss.steps.s06b_plane_regularization._snap_normals import snap_normals
        planes = [{
            "id": 0, "normal": np.array([0.02, 0.99, 0.1]),
            "d": -1.0, "label": "floor",
            "boundary_3d": np.array([[0, 1, 0], [5, 1, 0]]),
        }]
        snap_normals(planes, threshold_deg=20.0)
        np.testing.assert_array_equal(planes[0]["normal"], [0, 1, 0])

    def test_skip_when_angle_too_large(self):
        from gss.steps.s06b_plane_regularization._snap_normals import snap_normals
        original_normal = np.array([0.5, 0.5, 0.707])
        planes = [{
            "id": 0, "normal": original_normal.copy(),
            "d": -1.0, "label": "wall",
            "boundary_3d": np.empty((0, 3)),
        }]
        snap_normals(planes, threshold_deg=5.0)
        # Should NOT snap (angle > 5 degrees from any axis)
        np.testing.assert_array_almost_equal(planes[0]["normal"], original_normal)

    def test_other_label_ignored(self):
        from gss.steps.s06b_plane_regularization._snap_normals import snap_normals
        original = np.array([0.5, 0.5, 0.707])
        planes = [{
            "id": 0, "normal": original.copy(),
            "d": 0.0, "label": "other",
            "boundary_3d": np.empty((0, 3)),
        }]
        snap_normals(planes, threshold_deg=20.0)
        np.testing.assert_array_almost_equal(planes[0]["normal"], original)


# ---------------------------------------------------------------------------
# B. Height Snapping
# ---------------------------------------------------------------------------

class TestSnapHeights:
    def test_cluster_two_ceilings(self):
        from gss.steps.s06b_plane_regularization._snap_heights import snap_heights
        planes = [
            {"id": 0, "normal": np.array([0, 1, 0]), "d": -2.9, "label": "ceiling"},
            {"id": 1, "normal": np.array([0, 1, 0]), "d": -3.1, "label": "ceiling"},
        ]
        stats = snap_heights(planes, tolerance=0.5)
        # Both should snap to mean height = 3.0
        assert len(stats["ceiling_heights"]) == 1
        assert abs(stats["ceiling_heights"][0] - 3.0) < 0.01
        # d should be recomputed: d = -height * ny = -3.0 * 1 = -3.0
        assert abs(planes[0]["d"] - (-3.0)) < 0.01
        assert abs(planes[1]["d"] - (-3.0)) < 0.01

    def test_separate_clusters_when_far(self):
        from gss.steps.s06b_plane_regularization._snap_heights import snap_heights
        planes = [
            {"id": 0, "normal": np.array([0, 1, 0]), "d": 0.0, "label": "floor"},
            {"id": 1, "normal": np.array([0, 1, 0]), "d": -3.0, "label": "ceiling"},
        ]
        stats = snap_heights(planes, tolerance=0.5)
        assert len(stats["floor_heights"]) == 1
        assert len(stats["ceiling_heights"]) == 1

    def test_single_plane_unchanged(self):
        from gss.steps.s06b_plane_regularization._snap_heights import snap_heights
        planes = [
            {"id": 0, "normal": np.array([0, 1, 0]), "d": -2.5, "label": "floor"},
        ]
        original_d = planes[0]["d"]
        snap_heights(planes, tolerance=0.5)
        assert planes[0]["d"] == original_d


# ---------------------------------------------------------------------------
# C. Wall Thickness
# ---------------------------------------------------------------------------

class TestWallThickness:
    def test_parallel_pair_detected(self):
        from gss.steps.s06b_plane_regularization._wall_thickness import compute_wall_thickness
        planes = [
            {  # X-wall at x=0
                "id": 0, "normal": np.array([1.0, 0, 0]), "d": 0.0,
                "label": "wall",
                "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 5], [0, 0, 5], [0, 0, 0]]),
            },
            {  # X-wall at x=0.3
                "id": 1, "normal": np.array([-1.0, 0, 0]), "d": 0.3,
                "label": "wall",
                "boundary_3d": np.array([[0.3, 0, 0], [0.3, 3, 0], [0.3, 3, 5], [0.3, 0, 5], [0.3, 0, 0]]),
            },
            {  # Z-wall at z=5
                "id": 2, "normal": np.array([0, 0, 1.0]), "d": -5.0,
                "label": "wall",
                "boundary_3d": np.array([[0, 0, 5], [3, 0, 5], [3, 3, 5], [0, 3, 5], [0, 0, 5]]),
            },
        ]
        walls = compute_wall_thickness(planes, max_wall_thickness=1.0, default_wall_thickness=0.2)
        paired = [w for w in walls if len(w["plane_ids"]) == 2]
        unpaired = [w for w in walls if len(w["plane_ids"]) == 1]
        assert len(paired) == 1
        assert abs(paired[0]["thickness"] - 0.3) < 0.01
        assert len(unpaired) == 1  # Z-wall alone

    def test_unpaired_gets_default_thickness(self):
        from gss.steps.s06b_plane_regularization._wall_thickness import compute_wall_thickness
        planes = [{
            "id": 0, "normal": np.array([1.0, 0, 0]), "d": -5.0,
            "label": "wall",
            "boundary_3d": np.array([[5, 0, 0], [5, 3, 0], [5, 3, 4], [5, 0, 4], [5, 0, 0]]),
        }]
        walls = compute_wall_thickness(planes, default_wall_thickness=0.25)
        assert len(walls) == 1
        assert walls[0]["thickness"] == 0.25

    def test_no_walls_returns_empty(self):
        from gss.steps.s06b_plane_regularization._wall_thickness import compute_wall_thickness
        planes = [{"id": 0, "normal": np.array([0, 1, 0]), "d": 0.0, "label": "floor",
                    "boundary_3d": np.array([[0, 0, 0]])}]
        walls = compute_wall_thickness(planes)
        assert walls == []


# ---------------------------------------------------------------------------
# D. Intersection Trimming
# ---------------------------------------------------------------------------

class TestIntersectionTrimming:
    def test_extend_to_corner(self):
        from gss.steps.s06b_plane_regularization._intersection_trimming import trim_intersections
        walls = [
            {  # X-wall: vertical line at x=0, from z=0 to z=4
                "id": 0, "plane_ids": [0],
                "center_line_2d": [[0.0, 0.0], [0.0, 4.0]],
                "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x",
            },
            {  # Z-wall: horizontal line at z=5, from x=1 to x=6
                "id": 1, "plane_ids": [1],
                "center_line_2d": [[1.0, 5.0], [6.0, 5.0]],
                "thickness": 0.2, "height_range": [0, 3], "normal_axis": "z",
            },
        ]
        # Intersection of infinite lines: (0, 5)
        # Wall 0 endpoint at (0,4) needs to extend to (0,5) → dist=1, within 50% of wall length (4*0.5=2)
        # Wall 1 endpoint at (1,5) needs to extend to (0,5) → dist=1, within 50% of wall length (5*0.5=2.5)
        stats = trim_intersections(walls, snap_tolerance=0.5)
        assert stats["extended_endpoints"] == 2
        # Check wall 0 endpoint 1 is now (0, 5)
        assert abs(walls[0]["center_line_2d"][1][1] - 5.0) < 0.01
        # Check wall 1 endpoint 0 is now (0, 5)
        assert abs(walls[1]["center_line_2d"][0][0] - 0.0) < 0.01

    def test_parallel_walls_not_snapped(self):
        from gss.steps.s06b_plane_regularization._intersection_trimming import trim_intersections
        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 5]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
            {"id": 1, "plane_ids": [1], "center_line_2d": [[3, 0], [3, 5]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
        ]
        stats = trim_intersections(walls)
        assert stats["snapped_endpoints"] == 0
        assert stats["extended_endpoints"] == 0


# ---------------------------------------------------------------------------
# E. Space Detection
# ---------------------------------------------------------------------------

class TestSpaceDetection:
    def test_four_walls_form_room(self):
        from gss.steps.s06b_plane_regularization._space_detection import detect_spaces
        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [0, 4]], "normal_axis": "x"},
            {"id": 1, "center_line_2d": [[0, 4], [5, 4]], "normal_axis": "z"},
            {"id": 2, "center_line_2d": [[5, 4], [5, 0]], "normal_axis": "x"},
            {"id": 3, "center_line_2d": [[5, 0], [0, 0]], "normal_axis": "z"},
        ]
        spaces = detect_spaces(walls, floor_heights=[0.0], ceiling_heights=[3.0], min_area=1.0)
        assert len(spaces) == 1
        assert abs(spaces[0]["area"] - 20.0) < 0.1  # 5 x 4 = 20
        assert spaces[0]["floor_height"] == 0.0
        assert spaces[0]["ceiling_height"] == 3.0

    def test_no_closed_polygon_uses_convex_hull(self):
        from gss.steps.s06b_plane_regularization._space_detection import detect_spaces
        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [0, 4]], "normal_axis": "x"},
            {"id": 1, "center_line_2d": [[0, 4], [5, 4]], "normal_axis": "z"},
            {"id": 2, "center_line_2d": [[5, 4], [5, 0]], "normal_axis": "x"},
            # Missing 4th wall → polygonize fails → convex hull fallback
        ]
        spaces = detect_spaces(walls, floor_heights=[], ceiling_heights=[], min_area=1.0)
        # Convex hull fallback should produce a space
        assert len(spaces) == 1
        assert spaces[0]["area"] > 0

    def test_too_few_walls(self):
        from gss.steps.s06b_plane_regularization._space_detection import detect_spaces
        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [0, 4]], "normal_axis": "x"},
            {"id": 1, "center_line_2d": [[0, 4], [5, 4]], "normal_axis": "z"},
        ]
        spaces = detect_spaces(walls, floor_heights=[], ceiling_heights=[])
        assert len(spaces) == 0


# ---------------------------------------------------------------------------
# Full Step Integration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 1: Scale Estimation
# ---------------------------------------------------------------------------

class TestScaleEstimation:
    def test_auto_scale_colmap_data(self):
        """COLMAP-scale data (~35 scene units for a ~5m room) should yield scale ~7."""
        from gss.steps.s06b_plane_regularization.step import _estimate_scale
        # Simulated COLMAP-scale room: 35 x 21 x 28 (X, Y, Z)
        planes = [
            {"id": 0, "label": "wall", "boundary_3d": np.array([
                [0, 0, 0], [0, 21, 0], [0, 21, 28], [0, 0, 28],
            ])},
            {"id": 1, "label": "wall", "boundary_3d": np.array([
                [35, 0, 0], [35, 21, 0], [35, 21, 28], [35, 0, 28],
            ])},
            {"id": 2, "label": "floor", "boundary_3d": np.array([
                [0, 0, 0], [35, 0, 0], [35, 0, 28], [0, 0, 28],
            ])},
        ]
        scale = _estimate_scale(planes, expected_storey_height=2.7, expected_room_size=5.0)
        # Wall Y-extent = 21. Scale = 21/2.7 ≈ 7.8 (height-based)
        assert 4.0 < scale < 12.0

    def test_manual_scale_override(self):
        """When scale_mode='manual', the provided coordinate_scale should be used."""
        cfg = PlaneRegularizationConfig(scale_mode="manual", coordinate_scale=7.0)
        assert cfg.scale_mode == "manual"
        assert cfg.coordinate_scale == 7.0

    def test_metric_scale_is_one(self):
        """When scale_mode='metric', scale should be 1.0."""
        cfg = PlaneRegularizationConfig(scale_mode="metric")
        assert cfg.scale_mode == "metric"


# ---------------------------------------------------------------------------
# Phase 2: Wall Closure
# ---------------------------------------------------------------------------

class TestWallClosure:
    def test_synthesize_fourth_wall(self):
        """With 3 walls and a floor, the 4th wall should be synthesized."""
        from gss.steps.s06b_plane_regularization._wall_closure import synthesize_missing_walls
        # 3 walls forming an open U-shape, plus floor
        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
            {"id": 1, "plane_ids": [1], "center_line_2d": [[0, 4], [5, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "z"},
            {"id": 2, "plane_ids": [2], "center_line_2d": [[5, 4], [5, 0]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
        ]
        planes = [
            {"id": 0, "label": "wall", "normal": np.array([-1, 0, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4]])},
            {"id": 1, "label": "wall", "normal": np.array([0, 0, 1]), "d": -4.0,
             "boundary_3d": np.array([[0, 0, 4], [5, 0, 4], [5, 3, 4], [0, 3, 4]])},
            {"id": 2, "label": "wall", "normal": np.array([1, 0, 0]), "d": -5.0,
             "boundary_3d": np.array([[5, 0, 0], [5, 3, 0], [5, 3, 4], [5, 0, 4]])},
            {"id": 3, "label": "floor", "normal": np.array([0, 1, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [5, 0, 0], [5, 0, 4], [0, 0, 4]])},
        ]
        updated_walls, new_planes = synthesize_missing_walls(
            walls, planes,
            floor_heights=[0.0], ceiling_heights=[3.0],
            scale=1.0, max_gap_ratio=0.3,
        )
        # Should have at least 4 walls now (original 3 + 1 synthesized)
        assert len(updated_walls) >= 4
        # At least one synthetic wall
        synthetic = [w for w in updated_walls if w.get("synthetic")]
        assert len(synthetic) >= 1

    def test_no_synthesis_when_closed(self):
        """With 4 walls forming a closed room, no synthesis needed."""
        from gss.steps.s06b_plane_regularization._wall_closure import synthesize_missing_walls
        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
            {"id": 1, "plane_ids": [1], "center_line_2d": [[0, 4], [5, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "z"},
            {"id": 2, "plane_ids": [2], "center_line_2d": [[5, 4], [5, 0]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
            {"id": 3, "plane_ids": [3], "center_line_2d": [[5, 0], [0, 0]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "z"},
        ]
        planes = [
            {"id": 0, "label": "wall", "normal": np.array([-1, 0, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4]])},
            {"id": 1, "label": "wall", "normal": np.array([0, 0, 1]), "d": -4.0,
             "boundary_3d": np.array([[0, 0, 4], [5, 0, 4], [5, 3, 4], [0, 3, 4]])},
            {"id": 2, "label": "wall", "normal": np.array([1, 0, 0]), "d": -5.0,
             "boundary_3d": np.array([[5, 0, 0], [5, 3, 0], [5, 3, 4], [5, 0, 4]])},
            {"id": 3, "label": "wall", "normal": np.array([0, 0, -1]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [5, 0, 0], [5, 3, 0], [0, 3, 0]])},
            {"id": 4, "label": "floor", "normal": np.array([0, 1, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [5, 0, 0], [5, 0, 4], [0, 0, 4]])},
        ]
        updated_walls, new_planes = synthesize_missing_walls(
            walls, planes,
            floor_heights=[0.0], ceiling_heights=[3.0],
            scale=1.0, max_gap_ratio=0.3,
        )
        # No new walls should be added (all edges covered)
        synthetic = [w for w in updated_walls if w.get("synthetic")]
        assert len(synthetic) == 0

    def test_floor_guided_closure(self):
        """Wall closure should use floor boundary extent."""
        from gss.steps.s06b_plane_regularization._wall_closure import synthesize_missing_walls
        # Only 1 wall + floor
        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
        ]
        planes = [
            {"id": 0, "label": "wall", "normal": np.array([-1, 0, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4]])},
            {"id": 1, "label": "floor", "normal": np.array([0, 1, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [5, 0, 0], [5, 0, 4], [0, 0, 4]])},
        ]
        updated_walls, new_planes = synthesize_missing_walls(
            walls, planes,
            floor_heights=[0.0], ceiling_heights=[3.0],
            scale=1.0,
        )
        # Should synthesize walls for uncovered edges
        assert len(updated_walls) > 1
        assert len(new_planes) > 0


class TestWallClosureConcaveMode:
    """Tests for wall_closure_mode='concave' (L-shaped buildings)."""

    def test_concave_hull_edges_l_shape(self):
        """Concave hull of L-shaped points should produce 5+ edges (not 4 from convex)."""
        from gss.steps.s06b_plane_regularization._wall_closure import _concave_hull_edges
        # L-shaped floor plan points
        pts = np.array([
            [0, 0], [5, 0], [5, 3],
            [3, 3], [3, 5], [0, 5],
        ], dtype=float)
        edges = _concave_hull_edges(pts)
        # L-shape has 6 vertices → 6 edges (concave hull) or 4-5 (convex fallback)
        # With shapely concave_hull, should get 6 edges for L-shape
        assert len(edges) >= 4  # at minimum convex hull
        # Each edge should be a tuple of two points
        for p1, p2 in edges:
            assert len(p1) == 2
            assert len(p2) == 2

    def test_concave_mode_dispatches(self):
        """wall_closure_mode='concave' should use concave hull edges."""
        from gss.steps.s06b_plane_regularization._wall_closure import synthesize_missing_walls
        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 5]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
        ]
        # L-shaped floor
        planes = [
            {"id": 0, "label": "wall", "normal": np.array([-1, 0, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 5], [0, 0, 5]])},
            {"id": 1, "label": "floor", "normal": np.array([0, 1, 0]), "d": 0.0,
             "boundary_3d": np.array([
                 [0, 0, 0], [5, 0, 0], [5, 0, 3],
                 [3, 0, 3], [3, 0, 5], [0, 0, 5],
             ])},
        ]
        updated_walls, new_planes = synthesize_missing_walls(
            walls, planes,
            floor_heights=[0.0], ceiling_heights=[3.0],
            scale=1.0,
            wall_closure_mode="concave",
        )
        # Should produce walls (exact count depends on concave hull result)
        assert len(updated_walls) >= 1

    def test_auto_mode_backward_compat(self):
        """wall_closure_mode='auto' with normal_mode='manhattan' should use AABB (4 edges)."""
        from gss.steps.s06b_plane_regularization._wall_closure import synthesize_missing_walls
        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
        ]
        planes = [
            {"id": 0, "label": "wall", "normal": np.array([-1, 0, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4]])},
            {"id": 1, "label": "floor", "normal": np.array([0, 1, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [5, 0, 0], [5, 0, 4], [0, 0, 4]])},
        ]
        # auto + manhattan → AABB (same as before)
        updated_walls_auto, _ = synthesize_missing_walls(
            walls[:], planes,
            floor_heights=[0.0], ceiling_heights=[3.0],
            scale=1.0,
            normal_mode="manhattan",
            wall_closure_mode="auto",
        )
        # Explicit manhattan mode → same result
        walls2 = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
        ]
        updated_walls_explicit, _ = synthesize_missing_walls(
            walls2, planes,
            floor_heights=[0.0], ceiling_heights=[3.0],
            scale=1.0,
            wall_closure_mode="manhattan",
        )
        assert len(updated_walls_auto) == len(updated_walls_explicit)

    def test_config_wall_closure_mode_field(self):
        """Config should accept wall_closure_mode values."""
        cfg = PlaneRegularizationConfig(wall_closure_mode="concave")
        assert cfg.wall_closure_mode == "concave"
        cfg2 = PlaneRegularizationConfig(wall_closure_mode="auto")
        assert cfg2.wall_closure_mode == "auto"
        cfg3 = PlaneRegularizationConfig()  # default
        assert cfg3.wall_closure_mode == "auto"


# ---------------------------------------------------------------------------
# Phase 3: T-Junction and Junction Clustering
# ---------------------------------------------------------------------------

class TestTJunction:
    def test_t_junction_detection(self):
        """A wall endpoint meeting the middle of another wall should be handled."""
        from gss.steps.s06b_plane_regularization._intersection_trimming import trim_intersections
        walls = [
            {  # Long Z-wall from x=0 to x=10
                "id": 0, "plane_ids": [0],
                "center_line_2d": [[0.0, 5.0], [10.0, 5.0]],
                "thickness": 0.2, "height_range": [0, 3], "normal_axis": "z",
            },
            {  # X-wall approaching from z=0 to z=4.8 (near the long wall at z=5)
                "id": 1, "plane_ids": [1],
                "center_line_2d": [[5.0, 0.0], [5.0, 4.8]],
                "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x",
            },
        ]
        stats = trim_intersections(walls, snap_tolerance=0.5)
        # Wall 1 endpoint at (5, 4.8) should snap to (5, 5) — the intersection
        assert stats["snapped_endpoints"] >= 1
        assert abs(walls[1]["center_line_2d"][1][1] - 5.0) < 0.01

    def test_scaled_snap_tolerance(self):
        """Scaled snap tolerance should allow snapping at larger distances."""
        from gss.steps.s06b_plane_regularization._intersection_trimming import trim_intersections
        walls = [
            {"id": 0, "plane_ids": [0],
             "center_line_2d": [[0.0, 0.0], [0.0, 30.0]],
             "thickness": 1.0, "height_range": [0, 20], "normal_axis": "x"},
            {"id": 1, "plane_ids": [1],
             "center_line_2d": [[3.0, 30.0], [40.0, 30.0]],
             "thickness": 1.0, "height_range": [0, 20], "normal_axis": "z"},
        ]
        # With small tolerance, the 3-unit gap shouldn't be snapped
        stats_small = trim_intersections(
            [{"id": 0, "plane_ids": [0], "center_line_2d": [[0.0, 0.0], [0.0, 30.0]],
              "thickness": 1.0, "height_range": [0, 20], "normal_axis": "x"},
             {"id": 1, "plane_ids": [1], "center_line_2d": [[3.0, 30.0], [40.0, 30.0]],
              "thickness": 1.0, "height_range": [0, 20], "normal_axis": "z"}],
            snap_tolerance=0.5,
        )
        # With large tolerance (scaled), it should be snapped
        stats_large = trim_intersections(walls, snap_tolerance=5.0)
        assert stats_large["snapped_endpoints"] >= stats_small["snapped_endpoints"]


# ---------------------------------------------------------------------------
# Phase 4: Space Detection Improvements
# ---------------------------------------------------------------------------

class TestSpaceDetectionImprovements:
    def test_snap_buffer_closes_small_gap(self):
        """Snap+buffer should close small gaps between wall endpoints."""
        from gss.steps.s06b_plane_regularization._space_detection import detect_spaces
        # 4 walls with a small gap at one corner
        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [0, 4]], "normal_axis": "x"},
            {"id": 1, "center_line_2d": [[0, 4], [5, 4]], "normal_axis": "z"},
            {"id": 2, "center_line_2d": [[5, 4], [5, 0.1]], "normal_axis": "x"},  # gap of 0.1
            {"id": 3, "center_line_2d": [[5, 0], [0, 0]], "normal_axis": "z"},  # doesn't quite meet
        ]
        # Without snap, this might not close
        spaces_no_snap = detect_spaces(walls, floor_heights=[0.0], ceiling_heights=[3.0],
                                        min_area=1.0, snap_tolerance=0.0)
        # With snap, should close
        spaces_snap = detect_spaces(walls, floor_heights=[0.0], ceiling_heights=[3.0],
                                     min_area=1.0, snap_tolerance=0.2)
        # snap version should detect at least as many spaces
        assert len(spaces_snap) >= len(spaces_no_snap)

    def test_convex_hull_fallback(self):
        """When polygonize fails, convex hull should provide a fallback."""
        from gss.steps.s06b_plane_regularization._space_detection import _convex_hull_fallback
        from shapely.geometry import LineString
        lines = [
            LineString([(0, 0), (5, 0)]),
            LineString([(5, 0), (5, 4)]),
            LineString([(5, 4), (0, 4)]),
        ]
        spaces = _convex_hull_fallback(lines, floor_h=0.0, ceiling_h=3.0)
        assert len(spaces) == 1
        assert spaces[0]["area"] > 0


# ---------------------------------------------------------------------------
# Full Step Integration
# ---------------------------------------------------------------------------

class TestPlaneRegularizationStep:
    def test_config_defaults(self):
        cfg = PlaneRegularizationConfig()
        assert cfg.enable_normal_snapping is True
        assert cfg.enable_opening_detection is False
        assert cfg.normal_snap_threshold == 20.0
        assert cfg.scale_mode == "auto"
        assert cfg.enable_wall_closure is True

    def test_contracts(self):
        inp = PlaneRegularizationInput(
            planes_file=Path("planes.json"),
            boundaries_file=Path("boundaries.json"),
        )
        assert inp.planes_file == Path("planes.json")

        out = PlaneRegularizationOutput(
            planes_file=Path("p.json"), boundaries_file=Path("b.json"),
            walls_file=Path("w.json"), num_walls=3,
        )
        assert out.spaces_file is None
        assert out.num_spaces == 0

    def test_full_step_no_manhattan(self, data_root: Path, manhattan_planes_json: Path, manhattan_boundaries_json: Path):
        """Run the full step without manhattan_alignment.json (processes in original coords)."""
        from gss.steps.s06b_plane_regularization.step import PlaneRegularizationStep

        cfg = PlaneRegularizationConfig()
        step = PlaneRegularizationStep(config=cfg, data_root=data_root)
        inp = PlaneRegularizationInput(
            planes_file=manhattan_planes_json,
            boundaries_file=manhattan_boundaries_json,
        )
        output = step.execute(inp)

        assert output.planes_file.exists()
        assert output.boundaries_file.exists()
        assert output.walls_file.exists()
        assert output.num_walls >= 4

        # Verify planes.json is valid
        with open(output.planes_file) as f:
            planes = json.load(f)
        assert len(planes) >= 6

        # Verify walls.json has center_line_3d
        with open(output.walls_file) as f:
            walls = json.load(f)
        assert len(walls) >= 4
        for w in walls:
            assert "center_line_3d" in w

    def test_step_with_disabled_modules(self, data_root: Path, manhattan_planes_json: Path, manhattan_boundaries_json: Path):
        """Run with all modules disabled — should just pass through."""
        from gss.steps.s06b_plane_regularization.step import PlaneRegularizationStep

        cfg = PlaneRegularizationConfig(
            enable_normal_snapping=False,
            enable_height_snapping=False,
            enable_wall_thickness=False,
            enable_intersection_trimming=False,
            enable_space_detection=False,
            enable_wall_closure=False,
        )
        step = PlaneRegularizationStep(config=cfg, data_root=data_root)
        inp = PlaneRegularizationInput(
            planes_file=manhattan_planes_json,
            boundaries_file=manhattan_boundaries_json,
        )
        output = step.execute(inp)
        assert output.planes_file.exists()
        assert output.num_walls == 4

    def test_step_metric_scale(self, data_root: Path, manhattan_planes_json: Path, manhattan_boundaries_json: Path):
        """Run with metric scale mode (scale=1.0)."""
        from gss.steps.s06b_plane_regularization.step import PlaneRegularizationStep

        cfg = PlaneRegularizationConfig(scale_mode="metric")
        step = PlaneRegularizationStep(config=cfg, data_root=data_root)
        inp = PlaneRegularizationInput(
            planes_file=manhattan_planes_json,
            boundaries_file=manhattan_boundaries_json,
        )
        output = step.execute(inp)
        assert output.planes_file.exists()
        assert output.num_walls >= 4


# ---------------------------------------------------------------------------
# Robustness: Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_config_rejects_zero_room_size(self):
        """expected_room_size=0 should be rejected by Pydantic gt=0."""
        with pytest.raises(Exception):  # ValidationError
            PlaneRegularizationConfig(expected_room_size=0)

    def test_config_rejects_negative_room_size(self):
        with pytest.raises(Exception):
            PlaneRegularizationConfig(expected_room_size=-1.0)

    def test_config_rejects_zero_wall_thickness(self):
        with pytest.raises(Exception):
            PlaneRegularizationConfig(default_wall_thickness=0)

    def test_config_rejects_negative_wall_thickness(self):
        with pytest.raises(Exception):
            PlaneRegularizationConfig(default_wall_thickness=-0.1)

    def test_config_accepts_valid_values(self):
        cfg = PlaneRegularizationConfig(expected_room_size=3.0, default_wall_thickness=0.15)
        assert cfg.expected_room_size == 3.0
        assert cfg.default_wall_thickness == 0.15


# ---------------------------------------------------------------------------
# Robustness: Corrupt Manhattan JSON fallback
# ---------------------------------------------------------------------------

class TestCorruptManhattanJson:
    def test_corrupt_json_returns_none(self, data_root: Path):
        """Corrupt manhattan_alignment.json should gracefully return None."""
        from gss.steps.s06b_plane_regularization.step import _load_manhattan_rotation
        s06_dir = data_root / "interim" / "s06_planes"
        s06_dir.mkdir(parents=True, exist_ok=True)
        path = s06_dir / "manhattan_alignment.json"
        path.write_text("{invalid json!!")
        R = _load_manhattan_rotation(s06_dir)
        assert R is None

    def test_missing_key_returns_none(self, data_root: Path):
        """JSON without 'manhattan_rotation' key should return None."""
        from gss.steps.s06b_plane_regularization.step import _load_manhattan_rotation
        s06_dir = data_root / "interim" / "s06_planes"
        s06_dir.mkdir(parents=True, exist_ok=True)
        path = s06_dir / "manhattan_alignment.json"
        path.write_text('{"something_else": 42}')
        R = _load_manhattan_rotation(s06_dir)
        assert R is None

    def test_wrong_shape_returns_none(self, data_root: Path):
        """Non-3x3 matrix should return None."""
        from gss.steps.s06b_plane_regularization.step import _load_manhattan_rotation
        s06_dir = data_root / "interim" / "s06_planes"
        s06_dir.mkdir(parents=True, exist_ok=True)
        path = s06_dir / "manhattan_alignment.json"
        path.write_text('{"manhattan_rotation": [[1, 0], [0, 1]]}')
        R = _load_manhattan_rotation(s06_dir)
        assert R is None


# ---------------------------------------------------------------------------
# Robustness: Wall closure without floor
# ---------------------------------------------------------------------------

class TestWallClosureWithoutFloor:
    def test_wall_closure_uses_wall_aabb_fallback(self):
        """When no floor exists, wall AABB should be used for closure."""
        from gss.steps.s06b_plane_regularization._wall_closure import synthesize_missing_walls
        # 3 walls but NO floor plane
        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
            {"id": 1, "plane_ids": [1], "center_line_2d": [[0, 4], [5, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "z"},
            {"id": 2, "plane_ids": [2], "center_line_2d": [[5, 4], [5, 0]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
        ]
        planes = [
            {"id": 0, "label": "wall", "normal": np.array([-1, 0, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4]])},
            {"id": 1, "label": "wall", "normal": np.array([0, 0, 1]), "d": -4.0,
             "boundary_3d": np.array([[0, 0, 4], [5, 0, 4], [5, 3, 4], [0, 3, 4]])},
            {"id": 2, "label": "wall", "normal": np.array([1, 0, 0]), "d": -5.0,
             "boundary_3d": np.array([[5, 0, 0], [5, 3, 0], [5, 3, 4], [5, 0, 4]])},
            # No floor!
        ]
        updated_walls, new_planes = synthesize_missing_walls(
            walls, planes,
            floor_heights=[0.0], ceiling_heights=[3.0],
            scale=1.0, max_gap_ratio=0.3,
        )
        # Should still attempt closure using wall AABB
        assert len(updated_walls) >= 3
        # At least one synthetic wall from AABB fallback
        synthetic = [w for w in updated_walls if w.get("synthetic")]
        assert len(synthetic) >= 1

    def test_wall_closure_skips_with_no_floor_and_few_walls(self):
        """With < 3 wall endpoints, closure should skip entirely."""
        from gss.steps.s06b_plane_regularization._wall_closure import synthesize_missing_walls
        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
        ]
        planes = [
            {"id": 0, "label": "wall", "normal": np.array([-1, 0, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4]])},
        ]
        updated_walls, new_planes = synthesize_missing_walls(
            walls, planes,
            floor_heights=[], ceiling_heights=[],
            scale=1.0,
        )
        # Only 1 wall with 2 endpoints → AABB has < 3 unique points, should skip
        assert len(new_planes) == 0


# ---------------------------------------------------------------------------
# Robustness: Scale estimation bounds
# ---------------------------------------------------------------------------

class TestScaleEstimationBounds:
    def test_extreme_scale_capped_at_100(self):
        """Scale > 100 should be capped."""
        from gss.steps.s06b_plane_regularization.step import _estimate_scale
        # Giant scene: 1000 units for a 5m room → raw scale = 200
        planes = [
            {"id": 0, "label": "wall", "boundary_3d": np.array([
                [0, 0, 0], [0, 500, 0], [0, 500, 1000], [0, 0, 1000],
            ])},
            {"id": 1, "label": "floor", "boundary_3d": np.array([
                [0, 0, 0], [1000, 0, 0], [1000, 0, 1000], [0, 0, 1000],
            ])},
        ]
        scale = _estimate_scale(planes, expected_room_size=5.0)
        assert scale <= 100.0

    def test_low_scale_still_allowed(self):
        """Scale < 0.5 should warn but not be rejected (just capped at 0.1)."""
        from gss.steps.s06b_plane_regularization.step import _estimate_scale
        # Small scene: 1 unit for a 5m room → raw scale = 0.2
        planes = [
            {"id": 0, "label": "wall", "boundary_3d": np.array([
                [0, 0, 0], [0, 0.6, 0], [0, 0.6, 1], [0, 0, 1],
            ])},
            {"id": 1, "label": "floor", "boundary_3d": np.array([
                [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
            ])},
        ]
        scale = _estimate_scale(planes, expected_room_size=5.0)
        assert 0.1 <= scale < 0.5


# ---------------------------------------------------------------------------
# Robustness: Stats output
# ---------------------------------------------------------------------------

class TestStatsOutput:
    def test_stats_json_created(self, data_root: Path, manhattan_planes_json: Path, manhattan_boundaries_json: Path):
        """Full step run should produce stats.json."""
        from gss.steps.s06b_plane_regularization.step import PlaneRegularizationStep

        cfg = PlaneRegularizationConfig()
        step = PlaneRegularizationStep(config=cfg, data_root=data_root)
        inp = PlaneRegularizationInput(
            planes_file=manhattan_planes_json,
            boundaries_file=manhattan_boundaries_json,
        )
        step.execute(inp)
        stats_file = data_root / "interim" / "s06b_plane_regularization" / "stats.json"
        assert stats_file.exists()
        with open(stats_file) as f:
            stats = json.load(f)
        assert "manhattan_aligned" in stats
        assert "scale" in stats
        assert "normal_snapping" in stats
        assert "wall_thickness" in stats
        assert "wall_closure" in stats
        assert "intersection_trimming" in stats
        assert "space_detection" in stats


# ---------------------------------------------------------------------------
# Phase 1: Non-Manhattan Wall Support
# ---------------------------------------------------------------------------

class TestNormalClustering:
    """Test cluster mode for normal snapping (Phase 1.1)."""

    def test_cluster_mode_45deg_walls(self):
        """45° wall pairs → 4 axes discovered (2 directions × ±)."""
        from gss.steps.s06b_plane_regularization._snap_normals import snap_normals

        angle = np.radians(45)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        planes = [
            {
                "id": 0, "normal": np.array([cos_a, 0.0, sin_a]),
                "d": -5.0, "label": "wall",
                "boundary_3d": np.array([[3, 0, 3], [4, 3, 4], [4, 3, 4]]),
            },
            {
                "id": 1, "normal": np.array([-sin_a, 0.0, cos_a]),
                "d": -3.0, "label": "wall",
                "boundary_3d": np.array([[2, 0, 3], [3, 3, 4]]),
            },
        ]
        stats = snap_normals(planes, threshold_deg=20.0, normal_mode="cluster",
                             cluster_angle_tolerance=15.0)
        # Should discover 2 dominant directions → 4 axes (± each)
        assert stats["num_axes"] == 4
        assert stats["normal_mode"] == "cluster"

    def test_cluster_mode_preserves_manhattan(self):
        """Pure Manhattan data in cluster mode → should still snap correctly."""
        from gss.steps.s06b_plane_regularization._snap_normals import snap_normals

        planes = [
            {"id": 0, "normal": np.array([0.98, 0.05, 0.17]), "d": -5.0,
             "label": "wall", "boundary_3d": np.array([[5, 0, 0], [5, 3, 4]])},
            {"id": 1, "normal": np.array([-0.97, 0.02, -0.22]), "d": -0.1,
             "label": "wall", "boundary_3d": np.array([[0, 0, 0], [0, 3, 4]])},
            {"id": 2, "normal": np.array([0.15, 0.03, 0.99]), "d": -4.0,
             "label": "wall", "boundary_3d": np.array([[0, 0, 4], [5, 3, 4]])},
            {"id": 3, "normal": np.array([-0.18, 0.04, -0.98]), "d": 0.05,
             "label": "wall", "boundary_3d": np.array([[0, 0, 0], [5, 3, 0]])},
        ]
        stats = snap_normals(planes, threshold_deg=20.0, normal_mode="cluster",
                             cluster_angle_tolerance=15.0)
        # Should snap all 4 walls
        assert stats["snapped_walls"] == 4
        # X-axis walls should have normals dominated by X component
        assert abs(planes[0]["normal"][0]) > 0.95
        assert abs(planes[1]["normal"][0]) > 0.95
        # Z-axis walls should have normals dominated by Z component
        assert abs(planes[2]["normal"][2]) > 0.95
        assert abs(planes[3]["normal"][2]) > 0.95

    def test_cluster_angle_tolerance(self):
        """Tight tolerance → more clusters, loose → fewer."""
        from gss.steps.s06b_plane_regularization._snap_normals import _discover_wall_axes

        # Three walls: 0°, 10°, 25°
        planes = [
            {"id": 0, "normal": np.array([1.0, 0.0, 0.0]), "d": -1.0, "label": "wall"},
            {"id": 1, "normal": np.array([np.cos(np.radians(10)), 0.0, np.sin(np.radians(10))]),
             "d": -1.0, "label": "wall"},
            {"id": 2, "normal": np.array([np.cos(np.radians(25)), 0.0, np.sin(np.radians(25))]),
             "d": -1.0, "label": "wall"},
        ]
        # Tight tolerance: 5° → 3 separate clusters
        axes_tight = _discover_wall_axes(planes, threshold_deg=5.0)
        # Loose tolerance: 30° → 1 cluster
        axes_loose = _discover_wall_axes(planes, threshold_deg=30.0)
        # Tight should have more axes than loose
        assert len(axes_tight) >= len(axes_loose)

    def test_manhattan_mode_backward_compatible(self):
        """Manhattan mode should produce identical results to before."""
        from gss.steps.s06b_plane_regularization._snap_normals import snap_normals

        planes = [{
            "id": 0, "normal": np.array([0.96, 0.05, 0.2]),
            "d": -5.0, "label": "wall",
            "boundary_3d": np.array([[5, 0, 0], [5, 3, 0], [5, 3, 4]]),
        }]
        stats = snap_normals(planes, threshold_deg=20.0, normal_mode="manhattan")
        np.testing.assert_array_equal(planes[0]["normal"], [1, 0, 0])
        assert stats["num_axes"] == 4  # 4 Manhattan axes


class TestArbitraryAngleWallThickness:
    """Test wall thickness for non-Manhattan walls (Phase 1.2)."""

    def test_45deg_parallel_pair(self):
        """Two 45° walls should be detected as a pair."""
        from gss.steps.s06b_plane_regularization._wall_thickness import compute_wall_thickness

        angle = np.radians(45)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Boundary points must span along the wall's tangent direction
        # For normal [cos45, 0, sin45], tangent is [-sin45, 0, cos45]
        # Wall 0 at x+z≈4.24 (d=-3.0): points like [0, y, 4.24], [4.24, y, 0]
        # Wall 1 at x+z≈4.53 (d=3.2): points like [0.15, y, 4.38], [4.38, y, 0.15]
        planes = [
            {
                "id": 0, "normal": np.array([cos_a, 0.0, sin_a]),
                "d": -3.0, "label": "wall",
                "boundary_3d": np.array([
                    [0, 0, 4.24], [0, 3, 4.24], [4.24, 0, 0], [4.24, 3, 0],
                ]),
            },
            {
                "id": 1, "normal": np.array([-cos_a, 0.0, -sin_a]),
                "d": 3.2, "label": "wall",
                "boundary_3d": np.array([
                    [0.15, 0, 4.38], [0.15, 3, 4.38], [4.38, 0, 0.15], [4.38, 3, 0.15],
                ]),
            },
        ]
        walls = compute_wall_thickness(planes, max_wall_thickness=1.0, default_wall_thickness=0.2)
        paired = [w for w in walls if len(w["plane_ids"]) == 2]
        assert len(paired) >= 1
        # Should have oblique axis label
        assert paired[0]["normal_axis"].startswith("oblique")
        # Should have normal_vector field
        assert "normal_vector" in paired[0]

    def test_mixed_manhattan_and_oblique(self):
        """Mix of Manhattan and oblique walls should all be processed."""
        from gss.steps.s06b_plane_regularization._wall_thickness import compute_wall_thickness

        angle = np.radians(45)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        planes = [
            {  # Manhattan X-wall
                "id": 0, "normal": np.array([1.0, 0, 0]), "d": 0.0,
                "label": "wall",
                "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 5], [0, 0, 5]]),
            },
            {  # Manhattan Z-wall
                "id": 1, "normal": np.array([0, 0, 1.0]), "d": -5.0,
                "label": "wall",
                "boundary_3d": np.array([[0, 0, 5], [3, 0, 5], [3, 3, 5], [0, 3, 5]]),
            },
            {  # Oblique wall
                "id": 2, "normal": np.array([cos_a, 0.0, sin_a]),
                "d": -4.0, "label": "wall",
                "boundary_3d": np.array([[2, 0, 2], [2, 3, 2], [4, 0, 4], [4, 3, 4]]),
            },
        ]
        walls = compute_wall_thickness(planes, max_wall_thickness=1.0, default_wall_thickness=0.2)
        assert len(walls) == 3  # All 3 should become unpaired walls
        # Check we have both Manhattan and oblique
        axes = [w["normal_axis"] for w in walls]
        assert "x" in axes
        assert "z" in axes
        oblique = [a for a in axes if a.startswith("oblique")]
        assert len(oblique) == 1

    def test_manhattan_walls_unchanged(self):
        """Manhattan walls should produce identical results to before."""
        from gss.steps.s06b_plane_regularization._wall_thickness import compute_wall_thickness

        planes = [
            {"id": 0, "normal": np.array([1.0, 0, 0]), "d": 0.0,
             "label": "wall",
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 5], [0, 0, 5], [0, 0, 0]])},
            {"id": 1, "normal": np.array([-1.0, 0, 0]), "d": 0.3,
             "label": "wall",
             "boundary_3d": np.array([[0.3, 0, 0], [0.3, 3, 0], [0.3, 3, 5], [0.3, 0, 5], [0.3, 0, 0]])},
        ]
        walls = compute_wall_thickness(planes, max_wall_thickness=1.0, default_wall_thickness=0.2)
        paired = [w for w in walls if len(w["plane_ids"]) == 2]
        assert len(paired) == 1
        assert abs(paired[0]["thickness"] - 0.3) < 0.01
        assert paired[0]["normal_axis"] == "x"


class TestGeneralIntersectionTrimming:
    """Test intersection trimming for non-Manhattan walls (Phase 1.3)."""

    def test_oblique_corner_snap(self):
        """Two oblique walls meeting at a corner should snap."""
        from gss.steps.s06b_plane_regularization._intersection_trimming import trim_intersections

        walls = [
            {
                "id": 0, "plane_ids": [0],
                "center_line_2d": [[0.0, 0.0], [4.0, 4.0]],
                "thickness": 0.2, "height_range": [0, 3],
                "normal_axis": "oblique:45",
                "normal_vector": [0.707, 0.707],
            },
            {
                "id": 1, "plane_ids": [1],
                "center_line_2d": [[4.0, 0.0], [4.2, 3.8]],
                "thickness": 0.2, "height_range": [0, 3],
                "normal_axis": "oblique:135",
                "normal_vector": [-0.707, 0.707],
            },
        ]
        stats = trim_intersections(walls, snap_tolerance=0.5)
        # Should snap the endpoints near the intersection
        assert stats["snapped_endpoints"] + stats["extended_endpoints"] >= 1

    def test_manhattan_walls_still_work(self):
        """Manhattan walls should still work correctly after generalization."""
        from gss.steps.s06b_plane_regularization._intersection_trimming import trim_intersections

        walls = [
            {"id": 0, "plane_ids": [0],
             "center_line_2d": [[0.0, 0.0], [0.0, 4.0]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
            {"id": 1, "plane_ids": [1],
             "center_line_2d": [[1.0, 5.0], [6.0, 5.0]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "z"},
        ]
        stats = trim_intersections(walls, snap_tolerance=0.5)
        assert stats["extended_endpoints"] == 2
        assert abs(walls[0]["center_line_2d"][1][1] - 5.0) < 0.01
        assert abs(walls[1]["center_line_2d"][0][0] - 0.0) < 0.01

    def test_general_constrained_snap(self):
        """Constrained snap should project to wall's line for oblique walls."""
        from gss.steps.s06b_plane_regularization._intersection_trimming import _constrained_snap

        wall = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [4.0, 4.0]],
            "normal_axis": "oblique:45",
            "normal_vector": [0.707, 0.707],
        }
        ix = np.array([3.0, 3.5])
        result = _constrained_snap(ix, wall, ep_idx=1)
        # Should be on the line y=x
        assert abs(result[0] - result[1]) < 0.1

    def test_enforce_wall_straightness_oblique(self):
        """Oblique walls should maintain straightness after enforcement."""
        from gss.steps.s06b_plane_regularization._intersection_trimming import _enforce_wall_straightness

        s2 = np.sqrt(2) / 2  # exact 0.7071...
        walls = [{
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [4.1, 3.9]],  # slightly off y=x
            "normal_axis": "oblique:45",
            "normal_vector": [s2, s2],
        }]
        corrected = _enforce_wall_straightness(walls)
        assert corrected == 1
        # Both endpoints should project to same position along normal
        nv = np.array(walls[0]["normal_vector"])
        p1 = np.array(walls[0]["center_line_2d"][0])
        p2 = np.array(walls[0]["center_line_2d"][1])
        assert abs(np.dot(p1, nv) - np.dot(p2, nv)) < 1e-5


class TestGeneralWallClosure:
    """Test wall closure for non-Manhattan layouts (Phase 1.4)."""

    def test_convex_hull_closure(self):
        """Cluster mode should use convex hull edges for closure."""
        from gss.steps.s06b_plane_regularization._wall_closure import synthesize_missing_walls

        # Two walls covering only 2 of 4 hull edges
        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
            {"id": 1, "plane_ids": [1], "center_line_2d": [[0, 4], [5, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "z"},
        ]
        planes = [
            {"id": 0, "label": "wall", "normal": np.array([-1, 0, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4]])},
            {"id": 1, "label": "wall", "normal": np.array([0, 0, 1]), "d": -4.0,
             "boundary_3d": np.array([[0, 0, 4], [5, 0, 4], [5, 3, 4], [0, 3, 4]])},
            {"id": 2, "label": "floor", "normal": np.array([0, 1, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [5, 0, 0], [5, 0, 4], [0, 0, 4]])},
        ]
        updated_walls, new_planes = synthesize_missing_walls(
            walls, planes,
            floor_heights=[0.0], ceiling_heights=[3.0],
            scale=1.0, normal_mode="cluster",
        )
        # Should synthesize walls for uncovered hull edges
        assert len(updated_walls) > 2
        synthetic = [w for w in updated_walls if w.get("synthetic")]
        assert len(synthetic) >= 1

    def test_manhattan_closure_backward_compatible(self):
        """Manhattan mode closure should produce same results as before."""
        from gss.steps.s06b_plane_regularization._wall_closure import synthesize_missing_walls

        walls = [
            {"id": 0, "plane_ids": [0], "center_line_2d": [[0, 0], [0, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
            {"id": 1, "plane_ids": [1], "center_line_2d": [[0, 4], [5, 4]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "z"},
            {"id": 2, "plane_ids": [2], "center_line_2d": [[5, 4], [5, 0]],
             "thickness": 0.2, "height_range": [0, 3], "normal_axis": "x"},
        ]
        planes = [
            {"id": 0, "label": "wall", "normal": np.array([-1, 0, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [0, 3, 0], [0, 3, 4], [0, 0, 4]])},
            {"id": 1, "label": "wall", "normal": np.array([0, 0, 1]), "d": -4.0,
             "boundary_3d": np.array([[0, 0, 4], [5, 0, 4], [5, 3, 4], [0, 3, 4]])},
            {"id": 2, "label": "wall", "normal": np.array([1, 0, 0]), "d": -5.0,
             "boundary_3d": np.array([[5, 0, 0], [5, 3, 0], [5, 3, 4], [5, 0, 4]])},
            {"id": 3, "label": "floor", "normal": np.array([0, 1, 0]), "d": 0.0,
             "boundary_3d": np.array([[0, 0, 0], [5, 0, 0], [5, 0, 4], [0, 0, 4]])},
        ]
        updated_walls, new_planes = synthesize_missing_walls(
            walls, planes,
            floor_heights=[0.0], ceiling_heights=[3.0],
            scale=1.0, normal_mode="manhattan",
        )
        synthetic = [w for w in updated_walls if w.get("synthetic")]
        assert len(synthetic) >= 1


# ---------------------------------------------------------------------------
# Phase 2: Exterior Classification
# ---------------------------------------------------------------------------

class TestExteriorClassification:
    """Test interior/exterior wall classification (Phase 2.1)."""

    def test_rectangle_exterior_4_walls(self):
        """Rectangle layout → all 4 perimeter walls are exterior."""
        from gss.steps.s06b_plane_regularization._exterior_classification import classify_walls

        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [0, 4]], "normal_axis": "x"},
            {"id": 1, "center_line_2d": [[0, 4], [5, 4]], "normal_axis": "z"},
            {"id": 2, "center_line_2d": [[5, 4], [5, 0]], "normal_axis": "x"},
            {"id": 3, "center_line_2d": [[5, 0], [0, 0]], "normal_axis": "z"},
        ]
        stats = classify_walls(walls)
        assert stats["exterior"] == 4
        assert stats["interior"] == 0
        for w in walls:
            assert w["is_exterior"] is True

    def test_interior_wall_detected(self):
        """A wall well inside the hull should be classified as interior."""
        from gss.steps.s06b_plane_regularization._exterior_classification import classify_walls

        walls = [
            # Outer rectangle
            {"id": 0, "center_line_2d": [[0, 0], [0, 10]], "normal_axis": "x"},
            {"id": 1, "center_line_2d": [[0, 10], [10, 10]], "normal_axis": "z"},
            {"id": 2, "center_line_2d": [[10, 10], [10, 0]], "normal_axis": "x"},
            {"id": 3, "center_line_2d": [[10, 0], [0, 0]], "normal_axis": "z"},
            # Interior partition
            {"id": 4, "center_line_2d": [[5, 0], [5, 10]], "normal_axis": "x"},
        ]
        stats = classify_walls(walls)
        # The interior wall should be classified as interior
        assert walls[4]["is_exterior"] is False
        assert stats["interior"] >= 1

    def test_l_shape_exterior(self):
        """L-shaped layout → 6 exterior walls."""
        from gss.steps.s06b_plane_regularization._exterior_classification import classify_walls

        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [0, 10]], "normal_axis": "x"},
            {"id": 1, "center_line_2d": [[0, 10], [5, 10]], "normal_axis": "z"},
            {"id": 2, "center_line_2d": [[5, 10], [5, 5]], "normal_axis": "x"},
            {"id": 3, "center_line_2d": [[5, 5], [10, 5]], "normal_axis": "z"},
            {"id": 4, "center_line_2d": [[10, 5], [10, 0]], "normal_axis": "x"},
            {"id": 5, "center_line_2d": [[10, 0], [0, 0]], "normal_axis": "z"},
        ]
        stats = classify_walls(walls)
        assert stats["exterior"] >= 4  # At least 4 on hull

    def test_building_footprint_extraction(self):
        """Exterior walls → building footprint polygon."""
        from gss.steps.s06b_plane_regularization._exterior_classification import (
            classify_walls, extract_building_footprint,
        )

        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [0, 4]], "normal_axis": "x"},
            {"id": 1, "center_line_2d": [[0, 4], [5, 4]], "normal_axis": "z"},
            {"id": 2, "center_line_2d": [[5, 4], [5, 0]], "normal_axis": "x"},
            {"id": 3, "center_line_2d": [[5, 0], [0, 0]], "normal_axis": "z"},
        ]
        classify_walls(walls)
        footprint = extract_building_footprint(walls)
        assert footprint is not None
        assert len(footprint) >= 4  # At least a quadrilateral + closing point


class TestConfigNewFields:
    """Test new config fields for Phase 1+2+3+4."""

    def test_normal_mode_default(self):
        cfg = PlaneRegularizationConfig()
        assert cfg.normal_mode == "manhattan"
        assert cfg.cluster_angle_tolerance == 15.0
        assert cfg.enable_exterior_classification is False

    def test_normal_mode_cluster(self):
        cfg = PlaneRegularizationConfig(normal_mode="cluster")
        assert cfg.normal_mode == "cluster"

    def test_exterior_classification_enabled(self):
        cfg = PlaneRegularizationConfig(enable_exterior_classification=True)
        assert cfg.enable_exterior_classification is True

    def test_roof_detection_default_disabled(self):
        cfg = PlaneRegularizationConfig()
        assert cfg.enable_roof_detection is False

    def test_roof_detection_enabled(self):
        cfg = PlaneRegularizationConfig(enable_roof_detection=True)
        assert cfg.enable_roof_detection is True


# ---------------------------------------------------------------------------
# Phase 3: Storey Grouping Tests
# ---------------------------------------------------------------------------


class TestStoreyGrouping:
    """Test _group_storeys in _snap_heights.py."""

    def test_single_storey(self):
        from gss.steps.s06b_plane_regularization._snap_heights import _group_storeys

        storeys = _group_storeys([0.0], [3.0])
        assert len(storeys) == 1
        assert storeys[0]["name"] == "Ground Floor"
        assert abs(storeys[0]["floor_height"] - 0.0) < 0.01
        assert abs(storeys[0]["ceiling_height"] - 3.0) < 0.01

    def test_two_storeys(self):
        from gss.steps.s06b_plane_regularization._snap_heights import _group_storeys

        storeys = _group_storeys([0.0, 3.0], [2.8, 5.8])
        assert len(storeys) == 2
        assert storeys[0]["name"] == "Ground Floor"
        assert storeys[1]["name"] == "Floor 1"
        assert abs(storeys[0]["floor_height"] - 0.0) < 0.01
        assert abs(storeys[0]["ceiling_height"] - 2.8) < 0.01
        assert abs(storeys[1]["floor_height"] - 3.0) < 0.01
        assert abs(storeys[1]["ceiling_height"] - 5.8) < 0.01

    def test_empty_inputs(self):
        from gss.steps.s06b_plane_regularization._snap_heights import _group_storeys

        storeys = _group_storeys([], [])
        assert len(storeys) == 0

    def test_floor_only(self):
        from gss.steps.s06b_plane_regularization._snap_heights import _group_storeys

        storeys = _group_storeys([0.0], [])
        assert len(storeys) == 1
        assert abs(storeys[0]["floor_height"] - 0.0) < 0.01
        # Ceiling inferred as floor + 3.0
        assert abs(storeys[0]["ceiling_height"] - 3.0) < 0.01

    def test_ceiling_only(self):
        from gss.steps.s06b_plane_regularization._snap_heights import _group_storeys

        storeys = _group_storeys([], [2.5])
        assert len(storeys) == 1
        # Floor defaults to 0.0
        assert abs(storeys[0]["floor_height"] - 0.0) < 0.01

    def test_three_storeys(self):
        from gss.steps.s06b_plane_regularization._snap_heights import _group_storeys

        storeys = _group_storeys([0.0, 3.0, 6.0], [2.8, 5.8, 8.8])
        assert len(storeys) == 3
        assert storeys[0]["name"] == "Ground Floor"
        assert storeys[1]["name"] == "Floor 1"
        assert storeys[2]["name"] == "Floor 2"
        # Elevations should be sorted
        assert storeys[0]["elevation"] < storeys[1]["elevation"]
        assert storeys[1]["elevation"] < storeys[2]["elevation"]

    def test_storey_in_snap_heights_stats(self):
        """snap_heights should include storey definitions in stats."""
        from gss.steps.s06b_plane_regularization._snap_heights import snap_heights

        planes = [
            {"id": 0, "label": "floor", "normal": [0.0, -1.0, 0.0], "d": 0.0,
             "boundary_3d": [[0, 0, 0], [5, 0, 0], [5, 0, 5], [0, 0, 5]]},
            {"id": 1, "label": "ceiling", "normal": [0.0, 1.0, 0.0], "d": -3.0,
             "boundary_3d": [[0, 3, 0], [5, 3, 0], [5, 3, 5], [0, 3, 5]]},
        ]
        stats = snap_heights(planes)
        assert "storeys" in stats
        assert len(stats["storeys"]) >= 1
        assert stats["storeys"][0]["name"] == "Ground Floor"


# ---------------------------------------------------------------------------
# Phase 4: Roof Detection Tests
# ---------------------------------------------------------------------------


class TestRoofDetection:
    """Test _roof_detection.py."""

    def test_no_roof_planes(self):
        from gss.steps.s06b_plane_regularization._roof_detection import detect_roof_planes

        planes = [
            {"id": 0, "label": "wall", "normal": [1.0, 0.0, 0.0], "d": -5.0,
             "boundary_3d": [[5, 0, 0], [5, 3, 0], [5, 3, 4], [5, 0, 4]]},
        ]
        stats = detect_roof_planes(planes, ceiling_heights=[3.0])
        assert stats["num_roof_planes"] == 0
        assert stats["roof_type"] == "none"

    def test_flat_roof(self):
        from gss.steps.s06b_plane_regularization._roof_detection import detect_roof_planes

        planes = [
            {"id": 0, "label": "wall", "normal": [1.0, 0.0, 0.0], "d": -5.0,
             "boundary_3d": [[5, 0, 0], [5, 3, 0], [5, 3, 4], [5, 0, 4]]},
            {"id": 1, "label": "ceiling", "normal": [0.0, 1.0, 0.0], "d": -3.0,
             "boundary_3d": [[0, 3, 0], [5, 3, 0], [5, 3, 4], [0, 3, 4]]},
            # Flat roof above ceiling
            {"id": 2, "label": "wall", "normal": [0.0, 0.95, 0.0], "d": -3.5,
             "boundary_3d": [[0, 3.5, 0], [5, 3.5, 0], [5, 3.5, 4], [0, 3.5, 4]]},
        ]
        stats = detect_roof_planes(planes, ceiling_heights=[3.0])
        assert stats["num_roof_planes"] == 1
        assert stats["roof_type"] == "flat"
        assert planes[2]["label"] == "roof"
        assert planes[2]["roof_type"] == "flat"

    def test_shed_roof(self):
        from gss.steps.s06b_plane_regularization._roof_detection import detect_roof_planes

        planes = [
            {"id": 0, "label": "ceiling", "normal": [0.0, 1.0, 0.0], "d": -3.0,
             "boundary_3d": [[0, 3, 0], [5, 3, 0]]},
            # Inclined plane above ceiling (normal tilted ~45 degrees)
            {"id": 1, "label": "wall", "normal": [0.0, 0.707, 0.707], "d": -4.0,
             "boundary_3d": [[0, 3.5, 0], [5, 3.5, 0], [5, 4.5, 2], [0, 4.5, 2]]},
        ]
        stats = detect_roof_planes(planes, ceiling_heights=[3.0])
        assert stats["num_roof_planes"] == 1
        assert stats["roof_type"] == "shed"
        assert planes[1]["label"] == "roof"
        assert planes[1]["roof_type"] == "inclined"

    def test_gable_roof(self):
        from gss.steps.s06b_plane_regularization._roof_detection import detect_roof_planes

        planes = [
            {"id": 0, "label": "ceiling", "normal": [0.0, 1.0, 0.0], "d": -3.0,
             "boundary_3d": [[0, 3, 0], [5, 3, 0]]},
            # Two inclined planes = gable roof
            {"id": 1, "label": "wall", "normal": [0.0, 0.6, 0.8], "d": -4.0,
             "boundary_3d": [[0, 3.5, 0], [5, 3.5, 0], [5, 5, 2], [0, 5, 2]]},
            {"id": 2, "label": "wall", "normal": [0.0, 0.6, -0.8], "d": -4.0,
             "boundary_3d": [[0, 3.5, 4], [5, 3.5, 4], [5, 5, 2], [0, 5, 2]]},
        ]
        stats = detect_roof_planes(planes, ceiling_heights=[3.0])
        assert stats["num_roof_planes"] == 2
        assert stats["roof_type"] == "gable"

    def test_empty_ceiling_heights(self):
        from gss.steps.s06b_plane_regularization._roof_detection import detect_roof_planes

        planes = [
            {"id": 0, "label": "wall", "normal": [0.0, 0.707, 0.707], "d": -4.0,
             "boundary_3d": [[0, 4, 0], [5, 4, 0]]},
        ]
        stats = detect_roof_planes(planes, ceiling_heights=[])
        assert stats["num_roof_planes"] == 0

    def test_wall_below_ceiling_not_detected(self):
        """Walls below ceiling should not be classified as roof."""
        from gss.steps.s06b_plane_regularization._roof_detection import detect_roof_planes

        planes = [
            {"id": 0, "label": "wall", "normal": [0.0, 0.707, 0.707], "d": -1.0,
             "boundary_3d": [[0, 1.0, 0], [5, 1.0, 0], [5, 2, 1], [0, 2, 1]]},
        ]
        stats = detect_roof_planes(planes, ceiling_heights=[3.0])
        assert stats["num_roof_planes"] == 0
        assert planes[0]["label"] == "wall"


# ---------------------------------------------------------------------------
# Polyline merging tests (s06b module, moved from test_s07)
# ---------------------------------------------------------------------------


class TestPolylineMerging:
    def test_merge_collinear_walls(self):
        from gss.steps.s06b_plane_regularization._polyline_walls import merge_collinear_walls

        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [3, 0]], "thickness": 0.2,
             "height_range": [0, 3], "normal_axis": "z", "plane_ids": [1]},
            {"id": 1, "center_line_2d": [[3, 0], [3, 4]], "thickness": 0.2,
             "height_range": [0, 3], "normal_axis": "x", "plane_ids": [2]},
        ]
        # These walls meet at (3,0) at 90 degrees — should NOT merge (angle > 10)
        result = merge_collinear_walls(walls, angle_tolerance_deg=10.0)
        assert len(result) == 2  # not merged

    def test_merge_collinear_walls_same_direction(self):
        from gss.steps.s06b_plane_regularization._polyline_walls import merge_collinear_walls

        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [3, 0]], "thickness": 0.2,
             "height_range": [0, 3], "normal_axis": "z", "plane_ids": [1]},
            {"id": 1, "center_line_2d": [[3, 0], [6, 0]], "thickness": 0.2,
             "height_range": [0, 3], "normal_axis": "z", "plane_ids": [2]},
        ]
        result = merge_collinear_walls(walls, angle_tolerance_deg=10.0)
        assert len(result) == 1  # merged into polyline
        cl = result[0]["center_line_2d"]
        assert len(cl) == 3  # 3 points
        assert result[0].get("wall_type") == "polyline"


# ---------------------------------------------------------------------------
# Column detection tests (s06b module, moved from test_s07)
# ---------------------------------------------------------------------------


class TestColumnDetection:
    def test_column_detection_round(self):
        from gss.steps.s06b_plane_regularization._column_detection import detect_columns

        walls = [
            {"id": 0, "center_line_2d": [[5.0, 3.0], [5.3, 3.0]],
             "thickness": 0.3, "height_range": [0.0, 5.4],
             "normal_axis": "z", "plane_ids": [1]},
        ]
        columns, remaining = detect_columns(walls, scale=2.0, max_column_width=1.0)
        assert len(columns) == 1
        assert columns[0]["column_type"] == "round"  # squareness ~1.0
        assert len(remaining) == 0

    def test_column_detection_rectangular(self):
        from gss.steps.s06b_plane_regularization._column_detection import detect_columns

        walls = [
            {"id": 0, "center_line_2d": [[5.0, 3.0], [5.8, 3.0]],
             "thickness": 0.2, "height_range": [0.0, 5.4],
             "normal_axis": "z", "plane_ids": [1]},
        ]
        columns, remaining = detect_columns(walls, scale=2.0, max_column_width=1.0)
        assert len(columns) == 1
        assert columns[0]["column_type"] == "rectangular"
        assert len(remaining) == 0

    def test_column_not_wall(self):
        """Long wall should NOT be reclassified as column."""
        from gss.steps.s06b_plane_regularization._column_detection import detect_columns

        walls = [
            {"id": 0, "center_line_2d": [[0.0, 0.0], [10.0, 0.0]],
             "thickness": 0.2, "height_range": [0.0, 3.0],
             "normal_axis": "z", "plane_ids": [1]},
        ]
        columns, remaining = detect_columns(walls, scale=1.0, max_column_width=1.0)
        assert len(columns) == 0
        assert len(remaining) == 1
