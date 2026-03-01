"""Tests for S07: IFC Export — wall builder, polyline walls, opening shapes."""

import math

from tests.test_steps.conftest import needs_ifc


# ── Wall builder unit tests ──


@needs_ifc
class TestWallBuilder:
    def test_basic_wall(self, ifc_context):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [3.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
        }
        wall = create_wall_from_centerline(ctx, wall_data, scale=1.0)
        assert wall is not None
        assert wall.is_a("IfcWall")
        assert wall.Name == "Wall_0"

    def test_wall_with_axis_representation(self, ifc_context):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 1,
            "center_line_2d": [[0.0, 0.0], [5.0, 0.0]],
            "thickness": 0.3,
            "height_range": [0.0, 2.8],
            "normal_axis": "z",
        }
        wall = create_wall_from_centerline(ctx, wall_data, scale=1.0, create_axis=True)
        assert wall is not None

        # Check dual representation (Axis + Body)
        reps = wall.Representation.Representations
        rep_ids = [r.RepresentationIdentifier for r in reps]
        assert "Axis" in rep_ids
        assert "Body" in rep_ids

    def test_wall_without_axis(self, ifc_context):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 2,
            "center_line_2d": [[0.0, 0.0], [4.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
        }
        wall = create_wall_from_centerline(ctx, wall_data, scale=1.0, create_axis=False)
        reps = wall.Representation.Representations
        rep_ids = [r.RepresentationIdentifier for r in reps]
        assert "Axis" not in rep_ids
        assert "Body" in rep_ids

    def test_polyline_profile(self, ifc_context):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [4.0, 0.0]],
            "thickness": 0.25,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0,
            create_axis=False, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        # IfcArbitraryClosedProfileDef with polyline (5 points: 4 corners + close)
        profiles = ctx.ifc.by_type("IfcArbitraryClosedProfileDef")
        assert len(profiles) >= 1
        p = profiles[0]
        assert p.ProfileName == "Wall Profile"
        polyline = p.OuterCurve
        pts = polyline.Points
        assert len(pts) == 5  # 4 corners + closing point
        # First and last point should be the same (closed polyline)
        assert abs(pts[0].Coordinates[0] - pts[4].Coordinates[0]) < 0.001
        assert abs(pts[0].Coordinates[1] - pts[4].Coordinates[1]) < 0.001
        # Compute bounding box of polyline to check dimensions
        xs = [pt.Coordinates[0] for pt in pts]
        ys = [pt.Coordinates[1] for pt in pts]
        bbox_width = max(xs) - min(xs)
        bbox_height = max(ys) - min(ys)
        assert abs(bbox_width - 4.0) < 0.001  # wall length
        assert abs(bbox_height - 0.25) < 0.001  # wall thickness

    def test_material_layer_set(self, ifc_context):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [3.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0, create_material_layers=True
        )
        # Verify IfcMaterialLayerSetUsage exists
        usages = ctx.ifc.by_type("IfcMaterialLayerSetUsage")
        assert len(usages) >= 1
        assert usages[0].LayerSetDirection == "AXIS2"

    def test_wall_type(self, ifc_context):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [3.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
        }
        create_wall_from_centerline(ctx, wall_data, scale=1.0, create_wall_type=True)

        wall_types = ctx.ifc.by_type("IfcWallType")
        assert len(wall_types) >= 1
        assert wall_types[0].PredefinedType == "SOLIDWALL"

        # IfcRelDefinesByType should exist
        rel_types = ctx.ifc.by_type("IfcRelDefinesByType")
        assert len(rel_types) >= 1

    def test_property_set_synthetic(self, ifc_context):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 5,
            "center_line_2d": [[0.0, 0.0], [2.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
            "synthetic": True,
        }
        wall = create_wall_from_centerline(ctx, wall_data, scale=1.0, create_property_set=True)
        assert wall.Name == "Wall_5_Synthetic"

        psets = ctx.ifc.by_type("IfcPropertySet")
        assert len(psets) >= 1
        pset = psets[0]
        prop_names = [p.Name for p in pset.HasProperties]
        assert "IsExternal" in prop_names
        assert "Synthetic" in prop_names
        assert "Source" in prop_names

    def test_wall_scale(self, ifc_context):
        """Coordinate scale should divide all coordinates."""
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [2.0, 0.0]],
            "thickness": 0.4,
            "height_range": [0.0, 6.0],
            "normal_axis": "z",
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=2.0,
            create_axis=False, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        # With scale=2.0: length=1.0, thickness=0.2, height=3.0
        profiles = ctx.ifc.by_type("IfcArbitraryClosedProfileDef")
        p = profiles[0]
        polyline = p.OuterCurve
        pts = polyline.Points
        xs = [pt.Coordinates[0] for pt in pts]
        ys = [pt.Coordinates[1] for pt in pts]
        bbox_width = max(xs) - min(xs)
        bbox_height = max(ys) - min(ys)
        assert abs(bbox_width - 1.0) < 0.001  # scaled length
        assert abs(bbox_height - 0.2) < 0.001  # scaled thickness

    def test_oblique_wall_polyline(self, ifc_context):
        """Oblique (45°) wall should produce correctly oriented polyline."""
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 10,
            "center_line_2d": [[0.0, 0.0], [3.0, 3.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "oblique:45",
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0,
            create_axis=False, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        assert wall is not None

        profiles = ctx.ifc.by_type("IfcArbitraryClosedProfileDef")
        assert len(profiles) >= 1
        p = profiles[0]
        pts = p.OuterCurve.Points
        assert len(pts) == 5  # closed polyline

        # Verify edge lengths: 2 long edges (wall length) + 2 short edges (thickness)
        coords = [(pt.Coordinates[0], pt.Coordinates[1]) for pt in pts[:4]]
        edge_lens = []
        for i in range(4):
            j = (i + 1) % 4
            dx = coords[j][0] - coords[i][0]
            dy = coords[j][1] - coords[i][1]
            edge_lens.append(math.sqrt(dx*dx + dy*dy))
        edge_lens.sort()
        wall_len = math.sqrt(3**2 + 3**2)  # ~4.243
        # Two short edges ≈ 0.2, two long edges ≈ 4.243
        assert abs(edge_lens[0] - 0.2) < 0.01
        assert abs(edge_lens[1] - 0.2) < 0.01
        assert abs(edge_lens[2] - wall_len) < 0.01
        assert abs(edge_lens[3] - wall_len) < 0.01

    def test_degenerate_wall_skipped(self, ifc_context):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        # Zero-length wall
        wall_data = {
            "id": 0,
            "center_line_2d": [[1.0, 1.0], [1.0, 1.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
        }
        wall = create_wall_from_centerline(ctx, wall_data, scale=1.0)
        assert wall is None


# ── Polyline wall tests ──


@needs_ifc
class TestPolylineWall:
    def test_polyline_wall_3_points(self, ifc_context):
        """L-shaped wall with 3-point center-line."""
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]],
            "wall_type": "polyline",
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "",
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0,
            create_axis=True, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        assert wall is not None
        assert wall.is_a("IfcWall")

        # Check that the profile is a polygon (not just 4 corners)
        profiles = ctx.ifc.by_type("IfcArbitraryClosedProfileDef")
        assert len(profiles) >= 1
        pts = profiles[0].OuterCurve.Points
        # Buffer of 3-point polyline creates polygon with more than 5 points
        assert len(pts) > 5

    def test_polyline_wall_backward_compat(self, ifc_context):
        """2-point wall should produce same 5-point polyline as before."""
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [4.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0,
            create_axis=False, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        assert wall is not None
        profiles = ctx.ifc.by_type("IfcArbitraryClosedProfileDef")
        pts = profiles[0].OuterCurve.Points
        assert len(pts) == 5  # 4 corners + close

    def test_polyline_wall_axis_representation(self, ifc_context):
        """Polyline axis should have N points."""
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [5.0, 3.0]],
            "wall_type": "polyline",
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0,
            create_axis=True, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        assert wall is not None
        reps = wall.Representation.Representations
        axis_reps = [r for r in reps if r.RepresentationIdentifier == "Axis"]
        assert len(axis_reps) == 1
        polyline = axis_reps[0].Items[0]
        assert len(polyline.Points) == 4


# ── Opening shape tests ──


@needs_ifc
class TestOpeningShapes:
    def test_rectangular_backward_compat(self, ifc_context):
        """Opening without 'shape' field should default to rectangular."""
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [5.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
            "openings": [
                {"type": "window", "position_along_wall": [1.0, 2.0],
                 "height_range": [1.0, 2.0]},
            ],
        }
        wall = create_wall_from_centerline(ctx, wall_data, scale=1.0)
        n = create_openings_for_wall(ctx, wall, wall_data, scale=1.0)
        assert n == 1
        # Should use IfcRectangleProfileDef
        rect_profiles = ctx.ifc.by_type("IfcRectangleProfileDef")
        assert len(rect_profiles) >= 1

    def test_circular_opening_profile(self, ifc_context):
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [5.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
            "openings": [
                {"type": "window", "shape": "circular", "radius": 0.4,
                 "position_along_wall": [2.0, 2.8],
                 "height_range": [1.5, 2.3]},
            ],
        }
        wall = create_wall_from_centerline(ctx, wall_data, scale=1.0)
        n = create_openings_for_wall(ctx, wall, wall_data, scale=1.0)
        assert n == 1
        circle_profiles = ctx.ifc.by_type("IfcCircleProfileDef")
        assert len(circle_profiles) >= 1

    def test_arched_opening_profile(self, ifc_context):
        from gss.steps.s07_ifc_export._opening_builder import create_openings_for_wall
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = ifc_context
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [5.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
            "openings": [
                {"type": "window", "shape": "arched", "arch_radius": 0.5,
                 "position_along_wall": [1.0, 2.0],
                 "height_range": [0.5, 2.5]},
            ],
        }
        wall = create_wall_from_centerline(ctx, wall_data, scale=1.0)
        n = create_openings_for_wall(ctx, wall, wall_data, scale=1.0)
        assert n == 1
