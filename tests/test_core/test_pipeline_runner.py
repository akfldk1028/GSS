"""Tests for core pipeline runner and contracts."""

from pathlib import Path

import pytest
import yaml

from gss.core.contracts import PipelineConfig, StepEntry, CameraIntrinsics, CameraPose, StepMeta
from gss.core.pipeline_runner import load_pipeline_config, import_step_class, load_step_config


class TestContracts:
    def test_step_meta(self):
        meta = StepMeta(step_name="test", elapsed_seconds=1.5, params={"a": 1})
        assert meta.step_name == "test"
        assert meta.elapsed_seconds == 1.5

    def test_camera_intrinsics(self):
        cam = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        assert cam.width == 640

    def test_camera_pose(self):
        pose = CameraPose(image_name="frame_00000.png", matrix_4x4=[float(i) for i in range(16)])
        assert len(pose.matrix_4x4) == 16

    def test_pipeline_config(self):
        cfg = PipelineConfig(
            project_name="test",
            data_root=Path("./data"),
            steps=[StepEntry(name="s1", module="gss.steps.s01_extract_frames", config_file="c.yaml")],
        )
        assert len(cfg.steps) == 1
        assert cfg.steps[0].enabled is True


class TestPipelineRunner:
    def test_load_pipeline_config(self, tmp_path: Path):
        config = {
            "project_name": "test_project",
            "data_root": str(tmp_path / "data"),
            "steps": [
                {"name": "extract_frames", "module": "gss.steps.s01_extract_frames",
                 "config_file": "configs/steps/s01_extract_frames.yaml", "depends_on": [], "enabled": True},
            ],
        }
        config_file = tmp_path / "pipeline.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        cfg = load_pipeline_config(config_file)
        assert cfg.project_name == "test_project"
        assert len(cfg.steps) == 1

    def test_import_step_class(self):
        cls = import_step_class("gss.steps.s01_extract_frames")
        assert cls.__name__ == "ExtractFramesStep"
        assert hasattr(cls, "input_type")
        assert hasattr(cls, "output_type")

    def test_import_all_steps(self):
        modules = [
            "gss.steps.s00_import_ply",
            "gss.steps.s01_extract_frames",
            "gss.steps.s02_colmap",
            "gss.steps.s03_gaussian_splatting",
            "gss.steps.s03_planargs",
            "gss.steps.s04_depth_render",
            "gss.steps.s05_tsdf_fusion",
            "gss.steps.s06_plane_extraction",
            "gss.steps.s06b_plane_regularization",
            "gss.steps.s07_ifc_export",
        ]
        for module in modules:
            cls = import_step_class(module)
            assert cls.name, f"{module} has empty name"
            schema = cls.get_input_schema()
            assert "properties" in schema

    def test_load_step_config(self, tmp_path: Path):
        from gss.steps.s01_extract_frames.config import ExtractFramesConfig

        config_data = {"target_fps": 5.0, "max_frames": 100}
        config_file = tmp_path / "s01.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_step_config(config_file, ExtractFramesConfig)
        assert cfg.target_fps == 5.0
        assert cfg.max_frames == 100
