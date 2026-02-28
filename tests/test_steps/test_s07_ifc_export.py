"""Tests for S07: IFC Export step (center-line based, Cloud2BIM pattern)."""

from pathlib import Path

import pytest

from gss.steps.s07_ifc_export.config import IfcExportConfig
from gss.steps.s07_ifc_export.contracts import IfcExportInput, IfcExportOutput


def _has_ifcopenshell() -> bool:
    try:
        import ifcopenshell
        return True
    except ImportError:
        return False


needs_ifc = pytest.mark.skipif(not _has_ifcopenshell(), reason="ifcopenshell not installed")


# ── Contract tests ──


class TestIfcContracts:
    def test_config_defaults(self):
        cfg = IfcExportConfig()
        assert cfg.ifc_version == "IFC4"
        assert cfg.default_wall_thickness == 0.2
        assert cfg.storey_name == "Ground Floor"
        assert cfg.create_wall_types is True
        assert cfg.create_material_layers is True
        assert cfg.create_axis_representation is True
        assert cfg.include_synthetic_walls is True

    def test_config_new_fields(self):
        cfg = IfcExportConfig(
            author_name="Test",
            organization_name="TestOrg",
            wall_material_name="Steel",
            create_spaces=False,
        )
        assert cfg.author_name == "Test"
        assert cfg.organization_name == "TestOrg"
        assert cfg.wall_material_name == "Steel"
        assert cfg.create_spaces is False

    def test_input_requires_walls_file(self):
        inp = IfcExportInput(walls_file=Path("/tmp/walls.json"))
        assert inp.walls_file == Path("/tmp/walls.json")
        assert inp.spaces_file is None
        assert inp.planes_file is None

    def test_output_has_coordinate_scale(self):
        schema = IfcExportOutput.model_json_schema()
        assert "coordinate_scale" in schema["properties"]
        assert "ifc_path" in schema["properties"]
        assert "num_walls" in schema["properties"]


# ── Builder unit tests ──


@needs_ifc
class TestIfcBuilder:
    def test_create_ifc_file(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file

        ctx = create_ifc_file(project_name="TestProject")
        assert ctx.ifc is not None
        assert ctx.project is not None
        assert ctx.storey is not None
        assert ctx.body_context is not None
        assert ctx.axis_context is not None
        assert ctx.owner_history is not None

        # Verify spatial hierarchy
        projects = ctx.ifc.by_type("IfcProject")
        assert len(projects) == 1
        assert projects[0].Name == "TestProject"

        sites = ctx.ifc.by_type("IfcSite")
        assert len(sites) == 1

        buildings = ctx.ifc.by_type("IfcBuilding")
        assert len(buildings) == 1

        storeys = ctx.ifc.by_type("IfcBuildingStorey")
        assert len(storeys) == 1

    def test_owner_history(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file

        ctx = create_ifc_file(author_name="Alice", organization_name="ACME")
        oh = ctx.owner_history
        assert oh is not None
        person = oh.OwningUser.ThePerson
        assert person.GivenName == "Alice"
        org = oh.OwningUser.TheOrganization
        assert org.Name == "ACME"


@needs_ifc
class TestWallBuilder:
    def _make_ctx(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file
        return create_ifc_file()

    def test_basic_wall(self):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = self._make_ctx()
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

    def test_wall_with_axis_representation(self):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = self._make_ctx()
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

    def test_wall_without_axis(self):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = self._make_ctx()
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

    def test_rectangle_profile(self):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = self._make_ctx()
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
        # Find the IfcRectangleProfileDef
        profiles = ctx.ifc.by_type("IfcRectangleProfileDef")
        assert len(profiles) >= 1
        p = profiles[0]
        assert abs(p.XDim - 4.0) < 0.001  # wall length
        assert abs(p.YDim - 0.25) < 0.001  # wall thickness

    def test_material_layer_set(self):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = self._make_ctx()
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

    def test_wall_type(self):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = self._make_ctx()
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

    def test_property_set_synthetic(self):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = self._make_ctx()
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

    def test_wall_scale(self):
        """Coordinate scale should divide all coordinates."""
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = self._make_ctx()
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
        profiles = ctx.ifc.by_type("IfcRectangleProfileDef")
        p = profiles[0]
        assert abs(p.XDim - 1.0) < 0.001
        assert abs(p.YDim - 0.2) < 0.001

    def test_degenerate_wall_skipped(self):
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = self._make_ctx()
        # Zero-length wall
        wall_data = {
            "id": 0,
            "center_line_2d": [[1.0, 1.0], [1.0, 1.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
        }
        wall = create_wall_from_centerline(ctx, wall_data, scale=1.0)
        assert wall is None


@needs_ifc
class TestSlabBuilder:
    def _make_ctx(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file
        return create_ifc_file()

    def test_floor_slab(self):
        from gss.steps.s07_ifc_export._slab_builder import create_floor_slab

        ctx = self._make_ctx()
        space = {
            "id": 0,
            "boundary_2d": [[-1, 3], [1, 3], [1, -1], [-1, -1], [-1, 3]],
            "floor_height": -0.5,
            "ceiling_height": 2.5,
        }
        slab = create_floor_slab(ctx, space, scale=1.0, thickness=0.3)
        assert slab is not None
        assert slab.is_a("IfcSlab")
        assert slab.Name == "Floor_Room0"

    def test_ceiling_slab(self):
        from gss.steps.s07_ifc_export._slab_builder import create_ceiling_slab

        ctx = self._make_ctx()
        space = {
            "id": 0,
            "boundary_2d": [[-1, 3], [1, 3], [1, -1], [-1, -1], [-1, 3]],
            "floor_height": -0.5,
            "ceiling_height": 2.5,
        }
        slab = create_ceiling_slab(ctx, space, scale=1.0, thickness=0.3)
        assert slab is not None
        assert slab.Name == "Ceiling_Room0"


@needs_ifc
class TestSpaceBuilder:
    def _make_ctx(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file
        return create_ifc_file()

    def test_create_space(self):
        from gss.steps.s07_ifc_export._space_builder import create_space

        ctx = self._make_ctx()
        space_data = {
            "id": 0,
            "boundary_2d": [[-1, 3], [1, 3], [1, -1], [-1, -1], [-1, 3]],
            "area": 8.0,
            "floor_height": -0.5,
            "ceiling_height": 2.5,
        }
        space = create_space(ctx, space_data, scale=1.0)
        assert space is not None
        assert space.is_a("IfcSpace")
        assert space.Name == "Room_0"

        # Check property set
        psets = ctx.ifc.by_type("IfcPropertySet")
        pset_names = [ps.Name for ps in psets]
        assert "Pset_SpaceCommon" in pset_names

    def test_space_area_scaled(self):
        from gss.steps.s07_ifc_export._space_builder import create_space

        ctx = self._make_ctx()
        space_data = {
            "id": 0,
            "boundary_2d": [[-2, 6], [2, 6], [2, -2], [-2, -2], [-2, 6]],
            "area": 32.0,
            "floor_height": 0.0,
            "ceiling_height": 4.0,
        }
        create_space(ctx, space_data, scale=2.0)
        # area_m2 = 32.0 / (2.0*2.0) = 8.0
        psets = ctx.ifc.by_type("IfcPropertySet")
        for ps in psets:
            if ps.Name == "Pset_SpaceCommon":
                for prop in ps.HasProperties:
                    if prop.Name == "GrossFloorArea":
                        assert abs(prop.NominalValue.wrappedValue - 8.0) < 0.01


# ── Integration test ──


@needs_ifc
class TestIfcExportStep:
    def test_export_with_center_line_data(
        self, data_root: Path, sample_walls_json: Path, sample_spaces_json: Path
    ):
        from gss.steps.s07_ifc_export.step import IfcExportStep

        cfg = IfcExportConfig(project_name="Test_BIM")
        step = IfcExportStep(config=cfg, data_root=data_root)
        inp = IfcExportInput(
            walls_file=sample_walls_json,
            spaces_file=sample_spaces_json,
        )

        output = step.execute(inp)
        assert output.ifc_path.exists()
        assert output.ifc_path.suffix == ".ifc"
        assert output.num_walls == 4  # 4 walls (including 1 synthetic)
        assert output.num_slabs == 2  # 1 floor + 1 ceiling
        assert output.num_spaces == 1
        assert output.coordinate_scale == 1.0
        assert output.ifc_version == "IFC4"

        # Verify IFC file is readable
        import ifcopenshell
        ifc = ifcopenshell.open(str(output.ifc_path))
        walls = ifc.by_type("IfcWall")
        slabs = ifc.by_type("IfcSlab")
        spaces = ifc.by_type("IfcSpace")
        assert len(walls) == 4
        assert len(slabs) == 2
        assert len(spaces) == 1

        # Verify wall types exist
        wall_types = ifc.by_type("IfcWallType")
        assert len(wall_types) >= 1

        # Verify material layer sets exist
        layer_sets = ifc.by_type("IfcMaterialLayerSet")
        assert len(layer_sets) >= 1

    def test_exclude_synthetic_walls(
        self, data_root: Path, sample_walls_json: Path, sample_spaces_json: Path
    ):
        from gss.steps.s07_ifc_export.step import IfcExportStep

        cfg = IfcExportConfig(
            project_name="Test_NoSynth",
            include_synthetic_walls=False,
            create_slabs=False,
            create_spaces=False,
        )
        step = IfcExportStep(config=cfg, data_root=data_root)
        inp = IfcExportInput(
            walls_file=sample_walls_json,
            spaces_file=sample_spaces_json,
        )

        output = step.execute(inp)
        assert output.num_walls == 3  # synthetic excluded
        assert output.num_slabs == 0
        assert output.num_spaces == 0

    def test_minimal_export_walls_only(
        self, data_root: Path, sample_walls_json: Path
    ):
        from gss.steps.s07_ifc_export.step import IfcExportStep

        cfg = IfcExportConfig(
            project_name="Test_WallsOnly",
            create_slabs=False,
            create_spaces=False,
            create_wall_types=False,
            create_material_layers=False,
            create_property_sets=False,
            create_axis_representation=False,
        )
        step = IfcExportStep(config=cfg, data_root=data_root)
        inp = IfcExportInput(walls_file=sample_walls_json)

        output = step.execute(inp)
        assert output.num_walls == 4
        assert output.num_slabs == 0
        assert output.num_spaces == 0

        import ifcopenshell
        ifc = ifcopenshell.open(str(output.ifc_path))
        # No wall types in minimal mode
        assert len(ifc.by_type("IfcWallType")) == 0
        assert len(ifc.by_type("IfcMaterialLayerSetUsage")) == 0

    def test_coordinate_mapping(
        self, data_root: Path, sample_walls_json: Path, sample_spaces_json: Path
    ):
        """Verify Manhattan → IFC coordinate mapping is correct."""
        from gss.steps.s07_ifc_export.step import IfcExportStep

        cfg = IfcExportConfig(project_name="Test_Coords")
        step = IfcExportStep(config=cfg, data_root=data_root)
        inp = IfcExportInput(
            walls_file=sample_walls_json,
            spaces_file=sample_spaces_json,
        )

        output = step.execute(inp)

        import ifcopenshell
        ifc = ifcopenshell.open(str(output.ifc_path))

        # Check that rectangle profiles have correct dimensions
        profiles = ifc.by_type("IfcRectangleProfileDef")
        assert len(profiles) >= 1

        # Wall 0: center_line [[-1, 3], [1, 3]] → length 2.0, thickness 0.2
        # Wall 1: center_line [[-1, -1], [-1, 3]] → length 4.0, thickness 0.2
        lengths = sorted([p.XDim for p in profiles])
        assert abs(lengths[0] - 2.0) < 0.01  # short walls
        assert abs(lengths[-1] - 4.0) < 0.01  # long walls


# ── Phase 3: Multi-storey tests ──


@needs_ifc
class TestMultiStoreyBuilder:
    def test_single_storey_backward_compat(self):
        """No storeys param → single storey (original behavior)."""
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file

        ctx = create_ifc_file(project_name="Test_SingleStorey")
        storeys = ctx.ifc.by_type("IfcBuildingStorey")
        assert len(storeys) == 1
        assert ctx.storey is not None
        assert len(ctx.storeys) == 1

    def test_multi_storey_creation(self):
        """Multiple storey defs → multiple IfcBuildingStorey."""
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file

        storey_defs = [
            {"name": "Ground Floor", "floor_height": 0.0, "ceiling_height": 3.0, "elevation": 0.0},
            {"name": "Floor 1", "floor_height": 3.0, "ceiling_height": 6.0, "elevation": 3.0},
        ]
        ctx = create_ifc_file(project_name="Test_MultiStorey", storeys=storey_defs)
        storeys = ctx.ifc.by_type("IfcBuildingStorey")
        assert len(storeys) == 2
        assert len(ctx.storeys) == 2
        assert "Ground Floor" in ctx.storeys
        assert "Floor 1" in ctx.storeys
        # Default storey should be the first (ground floor)
        assert ctx.storey == ctx.storeys["Ground Floor"]

    def test_storey_elevation(self):
        """Storey elevation should be set correctly."""
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file

        storey_defs = [
            {"name": "Ground Floor", "floor_height": 0.0, "ceiling_height": 3.0, "elevation": 0.0},
            {"name": "Floor 1", "floor_height": 3.0, "ceiling_height": 6.0, "elevation": 3.0},
        ]
        ctx = create_ifc_file(storeys=storey_defs)
        s0 = ctx.storeys["Ground Floor"]
        s1 = ctx.storeys["Floor 1"]
        assert abs(s0.Elevation - 0.0) < 0.01
        assert abs(s1.Elevation - 3.0) < 0.01

    def test_get_storey_for_height(self):
        """Height-based storey lookup."""
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, get_storey_for_height

        storey_defs = [
            {"name": "Ground Floor", "floor_height": 0.0, "ceiling_height": 3.0, "elevation": 0.0},
            {"name": "Floor 1", "floor_height": 3.0, "ceiling_height": 6.0, "elevation": 3.0},
        ]
        ctx = create_ifc_file(storeys=storey_defs)

        # Height 1.5 → Ground Floor
        s = get_storey_for_height(ctx, 1.5, storey_defs, scale=1.0)
        assert s == ctx.storeys["Ground Floor"]

        # Height 4.5 → Floor 1
        s = get_storey_for_height(ctx, 4.5, storey_defs, scale=1.0)
        assert s == ctx.storeys["Floor 1"]

    def test_get_storey_fallback(self):
        """Fallback to default storey when no storeys defined."""
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, get_storey_for_height

        ctx = create_ifc_file()
        s = get_storey_for_height(ctx, 1.5, [], scale=1.0)
        assert s == ctx.storey


# ── Phase 4: Roof builder tests ──


@needs_ifc
class TestRoofBuilder:
    def _make_ctx(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file
        return create_ifc_file()

    def test_flat_roof(self):
        from gss.steps.s07_ifc_export._roof_builder import create_roof

        ctx = self._make_ctx()
        roof_planes = [{
            "id": 0, "label": "roof", "roof_type": "flat",
            "normal": [0.0, 1.0, 0.0], "d": -3.5,
            "boundary_3d": [[0, 3.5, 0], [5, 3.5, 0], [5, 3.5, 4], [0, 3.5, 4]],
        }]
        ifc_roof, count = create_roof(ctx, roof_planes, scale=1.0)
        assert ifc_roof is not None
        assert ifc_roof.is_a("IfcRoof")
        assert count == 1
        # Should have IfcSlab with ROOF type
        slabs = ctx.ifc.by_type("IfcSlab")
        roof_slabs = [s for s in slabs if s.PredefinedType == "ROOF"]
        assert len(roof_slabs) == 1

    def test_no_roof_planes(self):
        from gss.steps.s07_ifc_export._roof_builder import create_roof

        ctx = self._make_ctx()
        ifc_roof, count = create_roof(ctx, [], scale=1.0)
        assert ifc_roof is None
        assert count == 0

    def test_multiple_roof_slabs(self):
        from gss.steps.s07_ifc_export._roof_builder import create_roof

        ctx = self._make_ctx()
        roof_planes = [
            {"id": 0, "label": "roof", "roof_type": "flat",
             "normal": [0.0, 1.0, 0.0], "d": -3.5,
             "boundary_3d": [[0, 3.5, 0], [5, 3.5, 0], [5, 3.5, 4], [0, 3.5, 4]]},
            {"id": 1, "label": "roof", "roof_type": "flat",
             "normal": [0.0, 1.0, 0.0], "d": -4.0,
             "boundary_3d": [[0, 4.0, 0], [3, 4.0, 0], [3, 4.0, 2], [0, 4.0, 2]]},
        ]
        _, count = create_roof(ctx, roof_planes, scale=1.0)
        assert count == 2


# ── Phase 5: Site footprint tests ──


@needs_ifc
class TestSiteFootprint:
    def test_set_site_footprint(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, set_site_footprint

        ctx = create_ifc_file()
        footprint = [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]]
        set_site_footprint(ctx, footprint, scale=1.0)

        assert ctx.site.Representation is not None
        assert ctx.site.ObjectPlacement is not None

    def test_site_footprint_empty(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, set_site_footprint

        ctx = create_ifc_file()
        set_site_footprint(ctx, [], scale=1.0)
        # Site should have no representation
        assert ctx.site.Representation is None

    def test_site_footprint_scaled(self):
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, set_site_footprint

        ctx = create_ifc_file()
        # Points at scale=2.0 → IFC coords /2
        footprint = [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]]
        set_site_footprint(ctx, footprint, scale=2.0)
        assert ctx.site.Representation is not None
