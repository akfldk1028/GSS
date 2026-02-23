"""Tests for S02: COLMAP SfM step."""

from pathlib import Path

import pytest

from gss.steps.s02_colmap.config import ColmapConfig
from gss.steps.s02_colmap.contracts import ColmapInput, ColmapOutput


class TestColmapContracts:
    def test_input_model(self):
        inp = ColmapInput(frames_dir=Path("/tmp/frames"))
        assert inp.frames_dir == Path("/tmp/frames")

    def test_output_schema(self):
        schema = ColmapOutput.model_json_schema()
        assert "sparse_dir" in schema["properties"]
        assert "num_registered" in schema["properties"]
        assert "num_points3d" in schema["properties"]

    def test_config_defaults(self):
        cfg = ColmapConfig()
        assert cfg.matcher == "sequential"
        assert cfg.match_window == 10
        assert cfg.single_camera is True
        assert cfg.camera_model == "OPENCV"
        assert cfg.max_num_features == 8192


class TestColmapStep:
    def test_validate_missing_dir(self, data_root: Path):
        from gss.steps.s02_colmap.step import ColmapStep

        cfg = ColmapConfig()
        step = ColmapStep(config=cfg, data_root=data_root)
        inp = ColmapInput(frames_dir=Path("/nonexistent/frames"))
        assert step.validate_inputs(inp) is False

    def test_validate_too_few_frames(self, data_root: Path, tmp_path: Path):
        from gss.steps.s02_colmap.step import ColmapStep

        frames_dir = tmp_path / "sparse_frames"
        frames_dir.mkdir()
        # Only 2 frames (need at least 3)
        (frames_dir / "a.png").write_bytes(b"fake")
        (frames_dir / "b.png").write_bytes(b"fake")

        cfg = ColmapConfig()
        step = ColmapStep(config=cfg, data_root=data_root)
        inp = ColmapInput(frames_dir=frames_dir)
        assert step.validate_inputs(inp) is False

    def test_validate_sufficient_frames(self, data_root: Path, tmp_path: Path):
        from gss.steps.s02_colmap.step import ColmapStep

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(5):
            (frames_dir / f"img_{i:03d}.png").write_bytes(b"fake")

        cfg = ColmapConfig()
        step = ColmapStep(config=cfg, data_root=data_root)
        inp = ColmapInput(frames_dir=frames_dir)
        assert step.validate_inputs(inp) is True
