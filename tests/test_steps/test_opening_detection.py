"""Tests for opening detection (s06b Module F) and IFC opening builder (s07)."""

from pathlib import Path

import numpy as np
import pytest


# ── Histogram gap detection tests ──


class TestFindGaps1D:
    def test_clear_gap(self):
        """A distribution with an obvious gap should be detected."""
        from gss.steps.s06b_plane_regularization._opening_detection import _find_gaps_1d

        # Dense points at [0,1] and [2,3] with gap at [1,2]
        rng = np.random.default_rng(42)
        left = rng.uniform(0.0, 0.9, 500)
        right = rng.uniform(2.1, 3.0, 500)
        values = np.concatenate([left, right])

        gaps = _find_gaps_1d(values, bin_size=0.05, threshold_ratio=0.5, min_gap_size=0.3)
        assert len(gaps) >= 1

        # The gap should roughly cover the [1, 2] region
        gap = gaps[0]
        assert gap[1] - gap[0] > 0.3  # at least 0.3 units wide

    def test_no_gap(self):
        """Uniform distribution should have no gaps."""
        from gss.steps.s06b_plane_regularization._opening_detection import _find_gaps_1d

        values = np.random.default_rng(42).uniform(0, 3, 1000)
        gaps = _find_gaps_1d(values, bin_size=0.05, threshold_ratio=0.7, min_gap_size=0.3)
        assert len(gaps) == 0

    def test_empty_input(self):
        from gss.steps.s06b_plane_regularization._opening_detection import _find_gaps_1d

        gaps = _find_gaps_1d(np.array([]), bin_size=0.05, threshold_ratio=0.7, min_gap_size=0.3)
        assert gaps == []

    def test_min_gap_size_filter(self):
        """Tiny gaps below min_gap_size should be filtered."""
        from gss.steps.s06b_plane_regularization._opening_detection import _find_gaps_1d

        rng = np.random.default_rng(42)
        left = rng.uniform(0.0, 1.0, 500)
        right = rng.uniform(1.1, 2.0, 500)
        values = np.concatenate([left, right])

        # The gap is ~0.1 units, below min_gap_size=0.3
        gaps = _find_gaps_1d(values, bin_size=0.02, threshold_ratio=0.5, min_gap_size=0.3)
        assert len(gaps) == 0


# ── Wall projection tests ──


class TestProjectToWallFrame:
    def test_z_normal_wall(self):
        """Points along a Z-normal wall should project correctly."""
        from gss.steps.s06b_plane_regularization._opening_detection import _project_to_wall_frame

        # Wall: center_line [[0, 0], [4, 0]] → runs along X, normal=Z
        # Points: along X axis at Z=0
        pts = np.array([
            [0.5, 1.0, 0.0],
            [1.5, 1.5, 0.0],
            [3.0, 2.0, 0.0],
        ])
        result = _project_to_wall_frame(
            pts, normal_axis="z",
            center_line_2d=[[0.0, 0.0], [4.0, 0.0]],
            height_range=[0.0, 3.0],
        )
        assert result is not None
        u, v = result
        assert len(u) == 3
        assert len(v) == 3
        # u should be X coordinate - x_min (= 0)
        np.testing.assert_allclose(u, [0.5, 1.5, 3.0], atol=0.01)

    def test_x_normal_wall(self):
        from gss.steps.s06b_plane_regularization._opening_detection import _project_to_wall_frame

        # Wall: center_line [[2.0, 0.0], [2.0, 3.0]] → runs along Z, normal=X
        pts = np.array([
            [2.0, 0.5, 0.5],
            [2.0, 1.0, 1.5],
            [2.0, 2.0, 2.5],
        ])
        result = _project_to_wall_frame(
            pts, normal_axis="x",
            center_line_2d=[[2.0, 0.0], [2.0, 3.0]],
            height_range=[0.0, 3.0],
        )
        assert result is not None
        u, v = result
        assert len(u) == 3

    def test_u_direction_consistency_p1_greater_than_p2(self):
        """When p1 > p2, u should still be measured from p1 in the p1→p2 direction."""
        from gss.steps.s06b_plane_regularization._opening_detection import _project_to_wall_frame

        # Wall: p1=[4,0] → p2=[0,0], runs along X, normal=Z, p1 > p2
        pts = np.array([
            [3.5, 1.0, 0.0],  # near p1
            [0.5, 1.0, 0.0],  # near p2
        ])
        result = _project_to_wall_frame(
            pts, normal_axis="z",
            center_line_2d=[[4.0, 0.0], [0.0, 0.0]],
            height_range=[0.0, 3.0],
        )
        assert result is not None
        u, v = result
        # p1=4, p2=0, direction is negative, so u = (x - p1_x) * (-1)
        # pt(3.5): u = (3.5 - 4.0) * (-1) = 0.5
        # pt(0.5): u = (0.5 - 4.0) * (-1) = 3.5
        np.testing.assert_allclose(u, [0.5, 3.5], atol=0.01)
        # u values should be positive and cover [0, wall_length]
        assert np.all(u >= 0)

    def test_empty_points(self):
        from gss.steps.s06b_plane_regularization._opening_detection import _project_to_wall_frame

        result = _project_to_wall_frame(
            np.empty((0, 3)), normal_axis="z",
            center_line_2d=[[0.0, 0.0], [3.0, 0.0]],
            height_range=[0.0, 3.0],
        )
        assert result is None


# ── Opening classification tests ──


class TestDetectOpeningsInWall:
    def test_door_detection(self):
        """A gap from floor with height >= 1.8m should be classified as door."""
        from gss.steps.s06b_plane_regularization._opening_detection import (
            _detect_openings_in_wall, OpeningConfig,
        )

        rng = np.random.default_rng(42)
        # Wall 5m wide, 3m tall
        # Door gap at u=[1.5, 2.5], v=[0, 2.1]
        # Generate dense points everywhere except the door
        u_left = rng.uniform(0.0, 1.4, 300)
        u_right = rng.uniform(2.6, 5.0, 400)
        u_all = np.concatenate([u_left, u_right])
        v_all = rng.uniform(0.0, 3.0, len(u_all))

        cfg = OpeningConfig(
            histogram_resolution=0.1,
            histogram_threshold=0.5,
            min_opening_width=0.3,
            min_opening_height=0.3,
            door_sill_max=0.1,
            door_min_height=1.8,
        )
        openings = _detect_openings_in_wall(u_all, v_all, wall_height=3.0, cfg=cfg)

        # Should find at least one opening
        assert len(openings) >= 1
        # First opening should be roughly at the door location
        door = openings[0]
        assert door.width > 0.3

    def test_no_openings_in_solid_wall(self):
        from gss.steps.s06b_plane_regularization._opening_detection import (
            _detect_openings_in_wall, OpeningConfig,
        )

        rng = np.random.default_rng(42)
        u = rng.uniform(0.0, 5.0, 1000)
        v = rng.uniform(0.0, 3.0, 1000)

        cfg = OpeningConfig()
        openings = _detect_openings_in_wall(u, v, wall_height=3.0, cfg=cfg)
        assert len(openings) == 0


# ── Opening config tests ──


class TestOpeningConfig:
    def test_config_fields_exist(self):
        from gss.steps.s06b_plane_regularization.config import PlaneRegularizationConfig

        cfg = PlaneRegularizationConfig()
        assert cfg.enable_opening_detection is False
        assert cfg.opening_histogram_resolution == 0.05
        assert cfg.opening_histogram_threshold == 0.7
        assert cfg.opening_min_width == 0.3
        assert cfg.opening_min_height == 0.3
        assert cfg.opening_door_sill_max == 0.1
        assert cfg.opening_door_min_height == 1.8
        assert cfg.opening_min_points == 100


# ── IFC Opening builder tests ──


def _has_ifcopenshell() -> bool:
    try:
        import ifcopenshell
        return True
    except ImportError:
        return False


needs_ifc = pytest.mark.skipif(not _has_ifcopenshell(), reason="ifcopenshell not installed")


@needs_ifc
class TestOpeningBuilder:
    def _make_ctx_and_wall(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, assign_to_storey
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = create_ifc_file()
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [5.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
            "openings": [
                {
                    "type": "door",
                    "position_along_wall": [1.0, 2.0],
                    "height_range": [0.0, 2.1],
                    "width": 1.0,
                    "height": 2.1,
                },
                {
                    "type": "window",
                    "position_along_wall": [3.0, 4.0],
                    "height_range": [0.8, 2.0],
                    "width": 1.0,
                    "height": 1.2,
                },
            ],
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0,
            create_axis=False, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        assign_to_storey(ctx, [wall])
        return ctx, wall, wall_data

    def test_create_openings(self):
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall

        ctx, wall, wall_data = self._make_ctx_and_wall()
        count = create_openings_for_wall(ctx, wall, wall_data, scale=1.0)

        assert count == 2

        # Verify IFC elements
        openings = ctx.ifc.by_type("IfcOpeningElement")
        assert len(openings) == 2

        doors = ctx.ifc.by_type("IfcDoor")
        assert len(doors) == 1
        assert "Door" in doors[0].Name

        windows = ctx.ifc.by_type("IfcWindow")
        assert len(windows) == 1
        assert "Window" in windows[0].Name

        # Verify void relationships
        voids = ctx.ifc.by_type("IfcRelVoidsElement")
        assert len(voids) == 2

        fills = ctx.ifc.by_type("IfcRelFillsElement")
        assert len(fills) == 2

    def test_no_openings(self):
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file

        ctx = create_ifc_file()
        wall_data = {"id": 0}
        count = create_openings_for_wall(ctx, None, wall_data, scale=1.0)
        assert count == 0

    def test_opening_oz_is_relative_to_wall_base(self):
        """Opening oz must be v_start (relative to wall), not wall_base + v_start."""
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall

        ctx, wall, wall_data = self._make_ctx_and_wall()
        # Wall base is at z=0 (height_range[0]=0), window at v_start=0.8
        create_openings_for_wall(ctx, wall, wall_data, scale=1.0)

        # Get the window's opening element placement
        openings = ctx.ifc.by_type("IfcOpeningElement")
        for op in openings:
            rel_place = op.ObjectPlacement.RelativePlacement
            coords = rel_place.Location.Coordinates
            oz = coords[2]
            # All openings should have z relative to wall base, not absolute
            # Window: v_start=0.8 → oz=0.8, NOT 0+0.8=0.8 (same in this case)
            # But if wall_base were non-zero (e.g., 1.0), oz should still be
            # v_start, not 1.0+v_start
            assert oz >= 0.0
            # oz must not exceed wall height (3.0)
            assert oz < 3.0

    def test_opening_oz_nonzero_base(self):
        """When wall base > 0, oz should still be v_start (not base + v_start)."""
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, assign_to_storey
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = create_ifc_file()
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [5.0, 0.0]],
            "thickness": 0.2,
            "height_range": [2.0, 5.0],  # wall base at z=2.0
            "normal_axis": "z",
            "openings": [
                {
                    "type": "window",
                    "position_along_wall": [1.0, 2.0],
                    "height_range": [0.8, 2.0],  # relative to wall base in scene units
                    "width": 1.0,
                    "height": 1.2,
                },
            ],
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0,
            create_axis=False, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        assign_to_storey(ctx, [wall])

        # No floor_z → wall placed at height_range[0]/scale = 2.0
        create_openings_for_wall(ctx, wall, wall_data, scale=1.0)

        openings = ctx.ifc.by_type("IfcOpeningElement")
        assert len(openings) == 1
        coords = openings[0].ObjectPlacement.RelativePlacement.Location.Coordinates
        oz = coords[2]
        # oz should be v_start=0.8, not wall_base(2.0)+v_start(0.8)=2.8
        assert abs(oz - 0.8) < 0.01, f"oz={oz}, expected 0.8 (not {2.0 + 0.8})"

    def test_opening_oz_with_floor_z_offset(self):
        """When floor_z differs from per-wall base, oz should compensate."""
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, assign_to_storey
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = create_ifc_file()
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [5.0, 0.0]],
            "thickness": 0.2,
            "height_range": [2.0, 5.0],  # per-wall base = 2.0
            "normal_axis": "z",
            "openings": [
                {
                    "type": "window",
                    "position_along_wall": [1.0, 2.0],
                    "height_range": [0.8, 2.0],
                    "width": 1.0,
                    "height": 1.2,
                },
            ],
        }
        # floor_z=1.8: storey floor is 0.2m below per-wall base
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0, floor_z=1.8,
            create_axis=False, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        assign_to_storey(ctx, [wall])

        create_openings_for_wall(ctx, wall, wall_data, scale=1.0, floor_z=1.8)

        openings = ctx.ifc.by_type("IfcOpeningElement")
        assert len(openings) == 1
        coords = openings[0].ObjectPlacement.RelativePlacement.Location.Coordinates
        oz = coords[2]
        # Wall placed at z=1.8 (floor_z), opening v_start=0.8 relative to
        # height_range[0]=2.0. base_offset = 2.0 - 1.8 = 0.2
        # oz = v_start + base_offset = 0.8 + 0.2 = 1.0
        # World z = 1.8 + 1.0 = 2.8 = height_range[0] + v_start ✓
        assert abs(oz - 1.0) < 0.01, f"oz={oz}, expected 1.0 (v_start=0.8 + offset=0.2)"

    def test_fill_element_has_distinct_solid(self):
        """Fill element (door/window) must not share geometry with opening."""
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall

        ctx, wall, wall_data = self._make_ctx_and_wall()
        create_openings_for_wall(ctx, wall, wall_data, scale=1.0)

        openings = ctx.ifc.by_type("IfcOpeningElement")
        doors = ctx.ifc.by_type("IfcDoor")
        windows = ctx.ifc.by_type("IfcWindow")

        # Collect all IfcExtrudedAreaSolid ids used by openings
        opening_solid_ids = set()
        for op in openings:
            for rep in op.Representation.Representations:
                for item in rep.Items:
                    opening_solid_ids.add(item.id())

        # Fill elements must use different solid objects
        for fill in list(doors) + list(windows):
            for rep in fill.Representation.Representations:
                for item in rep.Items:
                    assert item.id() not in opening_solid_ids, (
                        f"Fill element {fill.Name} shares solid #{item.id()} with opening"
                    )

    def test_scaled_openings(self):
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, assign_to_storey
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = create_ifc_file()
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [10.0, 0.0]],
            "thickness": 0.4,
            "height_range": [0.0, 6.0],
            "normal_axis": "z",
            "openings": [
                {
                    "type": "door",
                    "position_along_wall": [2.0, 4.0],
                    "height_range": [0.0, 4.2],
                    "width": 2.0,
                    "height": 4.2,
                }
            ],
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=2.0,
            create_axis=False, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        assign_to_storey(ctx, [wall])

        count = create_openings_for_wall(ctx, wall, wall_data, scale=2.0)
        assert count == 1

        doors = ctx.ifc.by_type("IfcDoor")
        assert len(doors) == 1


# ── Manhattan rotation test ──


class TestManhattanRotation:
    def test_rotated_points_match_manhattan_planes(self):
        """After rotating COLMAP points with manhattan_rotation,
        inlier extraction should correctly match Manhattan planes."""
        from gss.steps.s06b_plane_regularization._opening_detection import (
            _extract_wall_inliers,
        )

        # 90° rotation about Y axis: R maps x→z, z→-x
        R = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.0,  0.0],
            [1.0, 0.0,  0.0],
        ])

        # Original COLMAP points: a plane at x=2 (normal along x)
        rng = np.random.default_rng(42)
        n_pts = 200
        colmap_pts = np.column_stack([
            np.full(n_pts, 2.0) + rng.normal(0, 0.01, n_pts),
            rng.uniform(0, 3, n_pts),
            rng.uniform(0, 5, n_pts),
        ])

        # After rotation: plane at z=2 (normal along z in Manhattan space)
        manhattan_pts = colmap_pts @ R.T
        manhattan_normal = np.array([0.0, 0.0, 1.0])
        manhattan_d = -2.0  # n·p + d = 0 → z + d = 0 → d = -z = -2

        # Without rotation: COLMAP points vs Manhattan plane → poor match
        inliers_wrong = _extract_wall_inliers(
            colmap_pts, manhattan_normal, manhattan_d, distance_threshold=0.1,
        )

        # With rotation: Manhattan points vs Manhattan plane → good match
        inliers_correct = _extract_wall_inliers(
            manhattan_pts, manhattan_normal, manhattan_d, distance_threshold=0.1,
        )

        # Correct extraction should find most points
        assert len(inliers_correct) > 0.8 * n_pts
        # Without rotation, far fewer points match (or zero)
        assert len(inliers_correct) > len(inliers_wrong)
