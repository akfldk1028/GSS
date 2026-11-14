"""Unit tests for Step 03: 3DGRUT (mocked trainer)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gss.steps.s03_3dgrut.config import S03_3dgrutConfig
from gss.steps.s03_3dgrut.contracts import S03_3dgrutInput, S03_3dgrutOutput
from gss.steps.s03_3dgrut.step import ThreeDGRUTStep, _find_checkpoint, _count_ply_points


@pytest.fixture
def tmp_data(tmp_path: Path):
    """Create minimal directory structure for testing."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_001.jpg").write_bytes(b"fake")

    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    (sparse_dir / "cameras.bin").write_bytes(b"fake")

    grut_dir = tmp_path / "3dgrut"
    grut_dir.mkdir()
    # Create a fake training script
    scripts_dir = grut_dir / "threedgrut" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "train.py").write_text("# fake train script")
    (scripts_dir / "export.py").write_text("# fake export script")

    data_root = tmp_path / "data"
    data_root.mkdir()

    return {
        "frames_dir": frames_dir,
        "sparse_dir": sparse_dir,
        "grut_dir": grut_dir,
        "data_root": data_root,
    }


class TestContracts:
    def test_input_schema(self):
        schema = S03_3dgrutInput.model_json_schema()
        assert "frames_dir" in schema["properties"]
        assert "sparse_dir" in schema["properties"]

    def test_output_schema(self):
        schema = S03_3dgrutOutput.model_json_schema()
        assert "model_path" in schema["properties"]
        assert "checkpoint_path" in schema["properties"]
        assert "usdz_path" in schema["properties"]
        assert "num_gaussians" in schema["properties"]

    def test_output_usdz_optional(self):
        out = S03_3dgrutOutput(
            model_path=Path("/a/b.ply"),
            checkpoint_path=Path("/a/ckpt.pt"),
            num_gaussians=1000,
            training_iterations=30000,
        )
        assert out.usdz_path is None


class TestConfig:
    def test_defaults(self):
        cfg = S03_3dgrutConfig()
        assert cfg.method == "3dgut"
        assert cfg.num_iterations == 30000
        assert cfg.export_ply is True
        assert cfg.export_usdz is True

    def test_3dgrt_method(self):
        cfg = S03_3dgrutConfig(method="3dgrt", config_name="apps/colmap_3dgrt")
        assert cfg.method == "3dgrt"


class TestValidation:
    def test_valid_inputs(self, tmp_data):
        step = ThreeDGRUTStep(
            config=S03_3dgrutConfig(threedgrut_path=tmp_data["grut_dir"]),
            data_root=tmp_data["data_root"],
        )
        inputs = S03_3dgrutInput(
            frames_dir=tmp_data["frames_dir"],
            sparse_dir=tmp_data["sparse_dir"],
        )
        assert step.validate_inputs(inputs) is True

    def test_missing_frames_dir(self, tmp_data):
        step = ThreeDGRUTStep(
            config=S03_3dgrutConfig(threedgrut_path=tmp_data["grut_dir"]),
            data_root=tmp_data["data_root"],
        )
        inputs = S03_3dgrutInput(
            frames_dir=tmp_data["data_root"] / "nonexistent",
            sparse_dir=tmp_data["sparse_dir"],
        )
        assert step.validate_inputs(inputs) is False

    def test_missing_grut(self, tmp_data):
        step = ThreeDGRUTStep(
            config=S03_3dgrutConfig(threedgrut_path=tmp_data["data_root"] / "no_grut"),
            data_root=tmp_data["data_root"],
        )
        inputs = S03_3dgrutInput(
            frames_dir=tmp_data["frames_dir"],
            sparse_dir=tmp_data["sparse_dir"],
        )
        assert step.validate_inputs(inputs) is False

    def test_invalid_method(self, tmp_data):
        step = ThreeDGRUTStep(
            config=S03_3dgrutConfig(
                method="invalid",
                threedgrut_path=tmp_data["grut_dir"],
            ),
            data_root=tmp_data["data_root"],
        )
        inputs = S03_3dgrutInput(
            frames_dir=tmp_data["frames_dir"],
            sparse_dir=tmp_data["sparse_dir"],
        )
        assert step.validate_inputs(inputs) is False


class TestRun:
    @patch("subprocess.run")
    def test_training_success(self, mock_run, tmp_data):
        """Test that run() invokes subprocess and finds checkpoint."""
        # Create a fake checkpoint
        output_dir = tmp_data["data_root"] / "interim" / "s03_3dgrut"
        output_dir.mkdir(parents=True)
        fake_ckpt = output_dir / "checkpoint.pt"
        fake_ckpt.write_bytes(b"fake checkpoint")

        # Mock subprocess to succeed
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        step = ThreeDGRUTStep(
            config=S03_3dgrutConfig(
                threedgrut_path=tmp_data["grut_dir"],
                export_ply=False,  # Skip PLY export (no real model)
                export_usdz=False,
            ),
            data_root=tmp_data["data_root"],
        )
        inputs = S03_3dgrutInput(
            frames_dir=tmp_data["frames_dir"],
            sparse_dir=tmp_data["sparse_dir"],
        )

        result = step.run(inputs)

        assert result.checkpoint_path == fake_ckpt
        assert result.training_iterations == 30000
        assert mock_run.called

    @patch("subprocess.run")
    def test_training_failure(self, mock_run, tmp_data):
        """Test that run() raises on training failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="CUDA OOM", stdout="")

        step = ThreeDGRUTStep(
            config=S03_3dgrutConfig(threedgrut_path=tmp_data["grut_dir"]),
            data_root=tmp_data["data_root"],
        )
        inputs = S03_3dgrutInput(
            frames_dir=tmp_data["frames_dir"],
            sparse_dir=tmp_data["sparse_dir"],
        )

        with pytest.raises(RuntimeError, match="3DGRUT training failed"):
            step.run(inputs)


class TestHelpers:
    def test_find_checkpoint(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"data")
        assert _find_checkpoint(tmp_path) == ckpt

    def test_find_checkpoint_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _find_checkpoint(tmp_path)
