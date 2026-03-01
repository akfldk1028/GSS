"""Tests for S07: IFC Export step — core builder & integration."""

from pathlib import Path

from gss.steps.s07_ifc_export.config import IfcExportConfig
from gss.steps.s07_ifc_export.contracts import IfcExportInput, IfcExportOutput
from tests.test_steps.conftest import needs_ifc


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

        # Check that polyline profiles have correct bounding box dimensions
        profiles = ifc.by_type("IfcArbitraryClosedProfileDef")
        assert len(profiles) >= 1

        # Compute wall length = longer bbox dimension for each profile
        lengths = []
        for p in profiles:
            pts = p.OuterCurve.Points
            xs = [pt.Coordinates[0] for pt in pts]
            ys = [pt.Coordinates[1] for pt in pts]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            lengths.append(max(w, h))  # longer dim = wall length
        lengths.sort()
        # Wall 0: center_line [[-1, 3], [1, 3]] → length 2.0
        # Wall 1: center_line [[-1, -1], [-1, 3]] → length 4.0
        assert abs(lengths[0] - 2.0) < 0.01  # short walls
        assert abs(lengths[-1] - 4.0) < 0.01  # long walls


# ── Multi-storey tests ──


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


# ── Site footprint tests ──


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


# ── Contract tests for new output fields ──


class TestNewOutputFields:
    def test_output_has_roof_fields(self):
        schema = IfcExportOutput.model_json_schema()
        assert "num_roof_slabs" in schema["properties"]
        assert "roof_type" in schema["properties"]
        assert "num_columns" in schema["properties"]
        assert "num_tessellated" in schema["properties"]

    def test_output_defaults(self):
        out = IfcExportOutput(ifc_path=Path("/tmp/test.ifc"))
        assert out.num_roof_slabs == 0
        assert out.roof_type == "none"
        assert out.num_columns == 0
        assert out.num_tessellated == 0

    def test_config_new_toggles(self):
        cfg = IfcExportConfig()
        assert cfg.create_roof_annotations is True
        assert cfg.create_columns is True
        assert cfg.create_tessellated is True
        assert cfg.tessellation_max_faces == 50000

    def test_input_new_fields(self):
        inp = IfcExportInput(walls_file=Path("/tmp/walls.json"))
        assert inp.building_context_file is None
        assert inp.columns_file is None
        assert inp.mesh_elements_file is None
