"""Tests for S05: TSDF Fusion step."""

from pathlib import Path

import pytest

from gss.steps.s05_tsdf_fusion.config import TsdfFusionConfig
from gss.steps.s05_tsdf_fusion.contracts import TsdfFusionInput, TsdfFusionOutput


def _has_open3d() -> bool:
    try:
        import open3d
        return True
    except ImportError:
        return False


class TestTsdfContracts:
    def test_input_model(self):
        inp = TsdfFusionInput(depth_dir=Path("/tmp/depth"), poses_file=Path("/tmp/poses.json"))
        assert inp.depth_dir == Path("/tmp/depth")

    def test_config_defaults(self):
        cfg = TsdfFusionConfig()
        assert cfg.voxel_size == 0.006
        assert cfg.sdf_trunc == 0.04
        assert cfg.depth_trunc == 5.0

    def test_output_schema(self):
        schema = TsdfFusionOutput.model_json_schema()
        assert "surface_points_path" in schema["properties"]


class TestTsdfFusionStep:
    @pytest.mark.skipif(not _has_open3d(), reason="open3d not installed")
    def test_fusion_with_synthetic_data(self, data_root: Path, sample_depth_data):
        from gss.steps.s05_tsdf_fusion.step import TsdfFusionStep

        depth_dir, poses_file = sample_depth_data
        cfg = TsdfFusionConfig(voxel_size=0.05, sdf_trunc=0.2)
        step = TsdfFusionStep(config=cfg, data_root=data_root)
        inp = TsdfFusionInput(depth_dir=depth_dir, poses_file=poses_file)

        output = step.execute(inp)
        assert output.surface_points_path.exists()
        assert output.num_surface_points >= 0
        assert output.metadata_path.exists()
