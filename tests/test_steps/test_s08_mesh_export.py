"""Tests for S08: Mesh Export step (IFC → GLB/USD)."""

from pathlib import Path

import numpy as np
import pytest

from gss.steps.s08_mesh_export.config import MeshExportConfig
from gss.steps.s08_mesh_export.contracts import MeshExportInput, MeshExportOutput


def _has_ifcopenshell() -> bool:
    try:
        import ifcopenshell
        return True
    except ImportError:
        return False


def _has_trimesh() -> bool:
    try:
        import trimesh
        return True
    except ImportError:
        return False


def _has_pxr() -> bool:
    try:
        from pxr import Usd
        return True
    except ImportError:
        return False


needs_ifc = pytest.mark.skipif(not _has_ifcopenshell(), reason="ifcopenshell not installed")
needs_trimesh = pytest.mark.skipif(not _has_trimesh(), reason="trimesh not installed")
needs_pxr = pytest.mark.skipif(not _has_pxr(), reason="usd-core not installed")


# ── Contract tests ──


class TestMeshExportContracts:
    def test_config_defaults(self):
        cfg = MeshExportConfig()
        assert cfg.export_glb is True
        assert cfg.export_usd is True
        assert cfg.export_usdz is False
        assert cfg.color_scheme == "by_class"
        assert cfg.include_spaces is False
        assert cfg.usd_up_axis == "Z"
        assert cfg.usd_meters_per_unit == 1.0

    def test_config_color_overrides(self):
        cfg = MeshExportConfig(
            color_wall=[1.0, 0.0, 0.0, 1.0],
            include_spaces=True,
        )
        assert cfg.color_wall == [1.0, 0.0, 0.0, 1.0]
        assert cfg.include_spaces is True

    def test_input_requires_ifc_path(self):
        inp = MeshExportInput(ifc_path=Path("/tmp/test.ifc"))
        assert inp.ifc_path == Path("/tmp/test.ifc")

    def test_output_schema(self):
        schema = MeshExportOutput.model_json_schema()
        assert "glb_path" in schema["properties"]
        assert "usd_path" in schema["properties"]
        assert "num_meshes" in schema["properties"]
        assert "num_vertices" in schema["properties"]

    def test_output_defaults(self):
        out = MeshExportOutput()
        assert out.glb_path is None
        assert out.usd_path is None
        assert out.num_meshes == 0


# ── IFC mesh extraction tests ──


@needs_ifc
class TestIfcToMesh:
    def _create_test_ifc(self, tmp_path: Path) -> Path:
        """Create a minimal IFC file for testing."""
        from gss.steps.s07_ifc_export._ifc_builder import create_ifc_file, assign_to_storey
        from gss.steps.s07_ifc_export._wall_builder import create_wall_from_centerline

        ctx = create_ifc_file(project_name="TestMesh")
        wall_data = {
            "id": 0,
            "center_line_2d": [[0.0, 0.0], [3.0, 0.0]],
            "thickness": 0.2,
            "height_range": [0.0, 3.0],
            "normal_axis": "z",
        }
        wall = create_wall_from_centerline(
            ctx, wall_data, scale=1.0,
            create_axis=False, create_material_layers=False,
            create_wall_type=False, create_property_set=False,
        )
        assign_to_storey(ctx, [wall])

        ifc_path = tmp_path / "test.ifc"
        ctx.ifc.write(str(ifc_path))
        return ifc_path

    def test_extract_meshes(self, tmp_path: Path):
        from gss.steps.s08_mesh_export._ifc_to_mesh import extract_meshes_from_ifc

        ifc_path = self._create_test_ifc(tmp_path)
        meshes = extract_meshes_from_ifc(ifc_path)

        assert len(meshes) >= 1
        for m in meshes:
            assert m.vertices.shape[1] == 3
            assert m.faces.shape[1] == 3
            assert len(m.color) >= 3
            assert m.ifc_class == "IfcWall"

    def test_color_mapping(self, tmp_path: Path):
        from gss.steps.s08_mesh_export._ifc_to_mesh import extract_meshes_from_ifc

        ifc_path = self._create_test_ifc(tmp_path)
        custom_colors = {
            "wall": [1.0, 0.0, 0.0, 1.0],
            "default": [0.5, 0.5, 0.5, 1.0],
        }
        meshes = extract_meshes_from_ifc(ifc_path, color_map=custom_colors)
        assert len(meshes) >= 1
        assert meshes[0].color == [1.0, 0.0, 0.0, 1.0]


# ── GLB writer tests ──


@needs_ifc
@needs_trimesh
class TestGlbWriter:
    def test_write_glb(self, tmp_path: Path):
        from gss.steps.s08_mesh_export._ifc_to_mesh import MeshData
        from gss.steps.s08_mesh_export._glb_writer import write_glb

        meshes = [
            MeshData(
                name="TestBox",
                ifc_class="IfcWall",
                vertices=np.array([
                    [0, 0, 0], [1, 0, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1],
                ], dtype=np.float64),
                faces=np.array([
                    [0, 1, 2], [3, 4, 5],
                ], dtype=np.int32),
                color=[0.8, 0.8, 0.8, 1.0],
            )
        ]
        output = tmp_path / "test.glb"
        result = write_glb(meshes, output)

        assert result.exists()
        assert result.stat().st_size > 0

        # Verify it's valid GLB
        import trimesh
        scene = trimesh.load(str(result))
        assert scene is not None


# ── Coordinate transform tests ──


@needs_trimesh
class TestGlbYupConversion:
    def test_glb_yup_conversion(self):
        """GLB writer should convert IFC Z-up to glTF Y-up: (x,y,z)→(x,z,-y)."""
        from gss.steps.s08_mesh_export._ifc_to_mesh import MeshData
        from gss.steps.s08_mesh_export._glb_writer import write_glb

        verts_zup = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=np.float64)
        meshes = [MeshData(
            name="Test", ifc_class="IfcWall",
            vertices=verts_zup,
            faces=np.array([[0, 1, 0]], dtype=np.int32),
            color=[0.8, 0.8, 0.8, 1.0],
        )]
        output = Path(__file__).parent.parent / "tmp_glb_test.glb"
        try:
            write_glb(meshes, output)
            import trimesh
            scene = trimesh.load(str(output))
            # Get the single geometry
            geom = list(scene.geometry.values())[0]
            loaded_verts = np.array(geom.vertices)
            # Expected: (x, z, -y)
            expected = np.array([[1.0, 3.0, -2.0], [4.0, 6.0, -5.0]])
            np.testing.assert_allclose(loaded_verts, expected, atol=1e-5)
        finally:
            output.unlink(missing_ok=True)


# ── USD writer tests ──


@needs_pxr
class TestUsdWriter:
    def test_has_pxr(self):
        from gss.steps.s08_mesh_export._usd_writer import _has_pxr
        assert _has_pxr() is True

    def test_sanitize_name(self):
        from gss.steps.s08_mesh_export._usd_writer import _sanitize_name
        assert _sanitize_name("Wall_0") == "Wall_0"
        assert _sanitize_name("Wall 0/sub") == "Wall_0_sub"
        assert _sanitize_name("0_invalid") == "_0_invalid"
        assert _sanitize_name("") == "_unnamed"

    def test_write_usd(self, tmp_path: Path):
        from gss.steps.s08_mesh_export._ifc_to_mesh import MeshData
        from gss.steps.s08_mesh_export._usd_writer import write_usd

        meshes = [
            MeshData(
                name="TestBox",
                ifc_class="IfcWall",
                vertices=np.array([
                    [0, 0, 0], [1, 0, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1],
                ], dtype=np.float64),
                faces=np.array([
                    [0, 1, 2], [3, 4, 5],
                ], dtype=np.int32),
                color=[0.8, 0.6, 0.4, 1.0],
            )
        ]
        output = tmp_path / "test.usdc"
        result = write_usd(meshes, output)

        assert result.exists()
        assert result.stat().st_size > 0

        # Verify it's a valid USD file
        from pxr import Usd, UsdGeom, UsdShade
        stage = Usd.Stage.Open(str(result))
        assert stage is not None
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z

        # Verify MaterialBindingAPI is properly applied (no warnings)
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                assert prim.HasAPI(UsdShade.MaterialBindingAPI)
                mat_api = UsdShade.MaterialBindingAPI(prim)
                mat, _ = mat_api.ComputeBoundMaterial()
                assert mat.GetPath().pathString != ""

    def test_usd_zup_preserves_vertices(self, tmp_path: Path):
        """With up_axis=Z (default), vertices should be preserved as-is."""
        from gss.steps.s08_mesh_export._ifc_to_mesh import MeshData
        from gss.steps.s08_mesh_export._usd_writer import write_usd

        verts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        meshes = [MeshData(
            name="ZUp", ifc_class="IfcWall",
            vertices=verts.copy(),
            faces=np.array([[0, 1, 0]], dtype=np.int32),
            color=[0.8, 0.8, 0.8, 1.0],
        )]
        output = tmp_path / "zup.usdc"
        write_usd(meshes, output, up_axis="Z")

        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(output))
        prim = stage.GetPrimAtPath("/Building/ZUp")
        mesh_prim = UsdGeom.Mesh(prim)
        loaded = np.array(mesh_prim.GetPointsAttr().Get())
        np.testing.assert_allclose(loaded, verts, atol=1e-5)

    def test_usd_yup_transforms_vertices(self, tmp_path: Path):
        """With up_axis=Y, vertices should transform (x,y,z)→(x,z,-y)."""
        from gss.steps.s08_mesh_export._ifc_to_mesh import MeshData
        from gss.steps.s08_mesh_export._usd_writer import write_usd

        verts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        meshes = [MeshData(
            name="YUp", ifc_class="IfcWall",
            vertices=verts.copy(),
            faces=np.array([[0, 1, 0]], dtype=np.int32),
            color=[0.8, 0.8, 0.8, 1.0],
        )]
        output = tmp_path / "yup.usdc"
        write_usd(meshes, output, up_axis="Y")

        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(output))
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
        prim = stage.GetPrimAtPath("/Building/YUp")
        mesh_prim = UsdGeom.Mesh(prim)
        loaded = np.array(mesh_prim.GetPointsAttr().Get())
        expected = np.array([[1.0, 3.0, -2.0], [4.0, 6.0, -5.0]])
        np.testing.assert_allclose(loaded, expected, atol=1e-5)


# ── Integration test ──


@needs_ifc
@needs_trimesh
class TestMeshExportStep:
    def test_full_export(
        self, data_root: Path, sample_walls_json: Path, sample_spaces_json: Path
    ):
        """Integration test: s07 → s08 pipeline."""
        from gss.steps.s07_ifc_export.step import IfcExportStep
        from gss.steps.s07_ifc_export.config import IfcExportConfig
        from gss.steps.s07_ifc_export.contracts import IfcExportInput
        from gss.steps.s08_mesh_export.step import MeshExportStep

        # First create IFC file via s07
        ifc_cfg = IfcExportConfig(project_name="Test_Mesh")
        ifc_step = IfcExportStep(config=ifc_cfg, data_root=data_root)
        ifc_output = ifc_step.execute(IfcExportInput(
            walls_file=sample_walls_json,
            spaces_file=sample_spaces_json,
        ))
        assert ifc_output.ifc_path.exists()

        # Then export meshes via s08
        mesh_cfg = MeshExportConfig(export_glb=True, export_usd=False)
        mesh_step = MeshExportStep(config=mesh_cfg, data_root=data_root)
        mesh_output = mesh_step.execute(MeshExportInput(ifc_path=ifc_output.ifc_path))

        assert mesh_output.glb_path is not None
        assert mesh_output.glb_path.exists()
        assert mesh_output.num_meshes > 0
        assert mesh_output.num_vertices > 0
        assert mesh_output.num_faces > 0
        assert mesh_output.usd_path is None  # export_usd=False
