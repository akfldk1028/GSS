"""Tests for S07: IFC Export step."""

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


class TestIfcContracts:
    def test_config_defaults(self):
        cfg = IfcExportConfig()
        assert cfg.ifc_version == "IFC4"
        assert cfg.default_wall_thickness == 0.2

    def test_output_schema(self):
        schema = IfcExportOutput.model_json_schema()
        assert "ifc_path" in schema["properties"]
        assert "num_walls" in schema["properties"]


class TestIfcExportStep:
    @pytest.mark.skipif(not _has_ifcopenshell(), reason="ifcopenshell not installed")
    def test_export_with_sample_data(
        self, data_root: Path, sample_planes_json: Path, sample_boundaries_json: Path
    ):
        from gss.steps.s07_ifc_export.step import IfcExportStep

        cfg = IfcExportConfig(project_name="Test_BIM")
        step = IfcExportStep(config=cfg, data_root=data_root)
        inp = IfcExportInput(planes_file=sample_planes_json, boundaries_file=sample_boundaries_json)

        output = step.execute(inp)
        assert output.ifc_path.exists()
        assert output.ifc_path.suffix == ".ifc"
        assert output.num_walls >= 1
        assert output.num_slabs >= 1
        assert output.ifc_version == "IFC4"

        # Verify IFC file is readable
        import ifcopenshell
        ifc = ifcopenshell.open(str(output.ifc_path))
        walls = ifc.by_type("IfcWall")
        slabs = ifc.by_type("IfcSlab")
        assert len(walls) == output.num_walls
        assert len(slabs) == output.num_slabs
