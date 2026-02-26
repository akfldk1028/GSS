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

    def test_no_closed_polygon(self):
        from gss.steps.s06b_plane_regularization._space_detection import detect_spaces
        walls = [
            {"id": 0, "center_line_2d": [[0, 0], [0, 4]], "normal_axis": "x"},
            {"id": 1, "center_line_2d": [[0, 4], [5, 4]], "normal_axis": "z"},
            {"id": 2, "center_line_2d": [[5, 4], [5, 0]], "normal_axis": "x"},
            # Missing 4th wall → no closed polygon
        ]
        spaces = detect_spaces(walls, floor_heights=[], ceiling_heights=[], min_area=1.0)
        assert len(spaces) == 0

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

class TestPlaneRegularizationStep:
    def test_config_defaults(self):
        cfg = PlaneRegularizationConfig()
        assert cfg.enable_normal_snapping is True
        assert cfg.enable_opening_detection is False
        assert cfg.normal_snap_threshold == 20.0

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
        assert output.num_walls == 4

        # Verify planes.json is valid
        with open(output.planes_file) as f:
            planes = json.load(f)
        assert len(planes) == 6

        # Verify walls.json
        with open(output.walls_file) as f:
            walls = json.load(f)
        assert len(walls) >= 4

    def test_step_with_disabled_modules(self, data_root: Path, manhattan_planes_json: Path, manhattan_boundaries_json: Path):
        """Run with all modules disabled — should just pass through."""
        from gss.steps.s06b_plane_regularization.step import PlaneRegularizationStep

        cfg = PlaneRegularizationConfig(
            enable_normal_snapping=False,
            enable_height_snapping=False,
            enable_wall_thickness=False,
            enable_intersection_trimming=False,
            enable_space_detection=False,
        )
        step = PlaneRegularizationStep(config=cfg, data_root=data_root)
        inp = PlaneRegularizationInput(
            planes_file=manhattan_planes_json,
            boundaries_file=manhattan_boundaries_json,
        )
        output = step.execute(inp)
        assert output.planes_file.exists()
        assert output.num_walls == 4
