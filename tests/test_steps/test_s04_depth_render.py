"""Tests for S04: Depth Render step."""

from pathlib import Path

import pytest

from gss.steps.s04_depth_render.config import DepthRenderConfig
from gss.steps.s04_depth_render.contracts import DepthRenderInput, DepthRenderOutput


class TestDepthRenderContracts:
    def test_input_model(self):
        inp = DepthRenderInput(
            model_path=Path("/tmp/model.ply"),
            sparse_dir=Path("/tmp/sparse"),
        )
        assert inp.model_path == Path("/tmp/model.ply")

    def test_output_schema(self):
        schema = DepthRenderOutput.model_json_schema()
        assert "depth_dir" in schema["properties"]
        assert "normal_dir" in schema["properties"]
        assert "num_views" in schema["properties"]
        assert "poses_file" in schema["properties"]

    def test_config_defaults(self):
        cfg = DepthRenderConfig()
        assert cfg.num_views == 400
        assert cfg.render_normals is True
        assert cfg.render_resolution_scale == 1.0
        assert cfg.view_selection == "uniform"


class TestDepthRenderStep:
    def test_validate_missing_model(self, data_root: Path):
        from gss.steps.s04_depth_render.step import DepthRenderStep

        cfg = DepthRenderConfig()
        step = DepthRenderStep(config=cfg, data_root=data_root)
        inp = DepthRenderInput(
            model_path=Path("/nonexistent/model.ply"),
            sparse_dir=Path("/nonexistent/sparse"),
        )
        assert step.validate_inputs(inp) is False
