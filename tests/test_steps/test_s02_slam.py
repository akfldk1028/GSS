"""Unit tests for Step 02-alt: GS-SLAM (mocked subprocess)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gss.steps.s02_slam.config import SlamConfig
from gss.steps.s02_slam.contracts import SlamInput, SlamOutput
from gss.steps.s02_slam.step import SlamStep


@pytest.fixture
def tmp_data(tmp_path: Path):
    """Create minimal directory structure for testing."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_001.png").write_bytes(b"fake")
    (frames_dir / "frame_002.png").write_bytes(b"fake")

    # Create fake SLAM repos
    for repo_name in ("MASt3R-SLAM", "MonoGS", "SLAM3R"):
        repo_dir = tmp_path / "clone" / repo_name
        repo_dir.mkdir(parents=True)

    # MASt3R-SLAM needs config/base.yaml and main.py
    mast3r = tmp_path / "clone" / "MASt3R-SLAM"
    (mast3r / "config").mkdir()
    (mast3r / "config" / "base.yaml").write_text("use_calib: False\n")
    (mast3r / "main.py").write_text("# fake")

    # MonoGS needs slam.py and configs dir
    monogs = tmp_path / "clone" / "MonoGS"
    (monogs / "slam.py").write_text("# fake")
    (monogs / "configs" / "mono" / "tum").mkdir(parents=True)
    (monogs / "configs" / "mono" / "tum" / "base_config.yaml").write_text("{}")

    # SLAM3R needs recon.py
    slam3r = tmp_path / "clone" / "SLAM3R"
    (slam3r / "recon.py").write_text("# fake")

    data_root = tmp_path / "data"
    data_root.mkdir()

    return {
        "tmp_path": tmp_path,
        "frames_dir": frames_dir,
        "data_root": data_root,
    }


class TestContracts:
    def test_input_schema(self):
        schema = SlamInput.model_json_schema()
        assert "frames_dir" in schema["properties"]

    def test_output_schema(self):
        schema = SlamOutput.model_json_schema()
        assert "surface_points_path" in schema["properties"]
        assert "metadata_path" in schema["properties"]
        assert "camera_poses_path" in schema["properties"]
        assert "num_surface_points" in schema["properties"]

    def test_output_camera_poses_optional(self):
        out = SlamOutput(
            surface_points_path=Path("/a/surface.ply"),
            metadata_path=Path("/a/meta.json"),
            num_surface_points=1000,
        )
        assert out.camera_poses_path is None

    def test_s06_compatibility(self):
        """SlamOutput must provide fields required by PlaneExtractionInput."""
        from gss.steps.s06_plane_extraction.contracts import PlaneExtractionInput

        s06_fields = set(PlaneExtractionInput.model_fields.keys())
        slam_fields = set(SlamOutput.model_fields.keys())
        assert s06_fields.issubset(slam_fields), (
            f"Missing fields for s06: {s06_fields - slam_fields}"
        )


class TestConfig:
    def test_defaults(self):
        cfg = SlamConfig()
        assert cfg.backend == "mast3r_slam"
        assert cfg.voxel_downsample == 0.02
        assert cfg.estimate_normals is True
        assert cfg.timeout == 7200

    def test_slam3r_backend(self):
        cfg = SlamConfig(backend="slam3r", conda_env="slam3r")
        assert cfg.backend == "slam3r"


class TestValidation:
    def test_valid_inputs_mast3r(self, tmp_data):
        step = SlamStep(
            config=SlamConfig(
                repo_path=tmp_data["tmp_path"] / "clone" / "MASt3R-SLAM",
            ),
            data_root=tmp_data["data_root"],
        )
        inputs = SlamInput(frames_dir=tmp_data["frames_dir"])
        assert step.validate_inputs(inputs) is True

    def test_missing_frames_dir(self, tmp_data):
        step = SlamStep(
            config=SlamConfig(
                repo_path=tmp_data["tmp_path"] / "clone" / "MASt3R-SLAM",
            ),
            data_root=tmp_data["data_root"],
        )
        inputs = SlamInput(frames_dir=tmp_data["data_root"] / "nonexistent")
        assert step.validate_inputs(inputs) is False

    def test_invalid_backend(self, tmp_data):
        step = SlamStep(
            config=SlamConfig(backend="invalid_slam"),
            data_root=tmp_data["data_root"],
        )
        inputs = SlamInput(frames_dir=tmp_data["frames_dir"])
        assert step.validate_inputs(inputs) is False

    def test_missing_repo(self, tmp_data):
        step = SlamStep(
            config=SlamConfig(
                repo_path=tmp_data["tmp_path"] / "clone" / "nonexistent",
            ),
            data_root=tmp_data["data_root"],
        )
        inputs = SlamInput(frames_dir=tmp_data["frames_dir"])
        assert step.validate_inputs(inputs) is False


class TestRun:
    def _create_fake_ply(self, ply_path: Path) -> None:
        """Create a minimal valid PLY file with points."""
        import numpy as np

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                np.random.rand(100, 3).astype(np.float64)
            )
            pcd.colors = o3d.utility.Vector3dVector(
                np.random.rand(100, 3).astype(np.float64)
            )
            ply_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(ply_path), pcd)
        except ImportError:
            pytest.skip("open3d not installed")

    @patch("subprocess.run")
    def test_mast3r_slam_run(self, mock_run, tmp_data):
        """Test MASt3R-SLAM backend with mocked subprocess."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="done")

        repo = tmp_data["tmp_path"] / "clone" / "MASt3R-SLAM"
        # Create fake output PLY where MASt3R-SLAM would write it
        seq_name = tmp_data["frames_dir"].resolve().stem
        logs_dir = repo / "logs" / "gss_slam"
        self._create_fake_ply(logs_dir / f"{seq_name}.ply")
        # Create fake trajectory
        (logs_dir / f"{seq_name}.txt").write_text("0.0 0 0 0 0 0 0 1\n")

        step = SlamStep(
            config=SlamConfig(repo_path=repo),
            data_root=tmp_data["data_root"],
        )
        inputs = SlamInput(frames_dir=tmp_data["frames_dir"])
        result = step.run(inputs)

        assert result.surface_points_path.exists()
        assert result.metadata_path.exists()
        assert result.num_surface_points > 0
        assert result.camera_poses_path is not None

        # Verify metadata
        meta = json.loads(result.metadata_path.read_text())
        assert meta["backend"] == "mast3r_slam"
        assert meta["method"] == "slam_mast3r_slam"

        # Verify subprocess was called with correct args
        call_args = mock_run.call_args[0][0]
        assert "main.py" in " ".join(str(a) for a in call_args)
        assert "--no-viz" in call_args

    @patch("subprocess.run")
    def test_slam3r_run(self, mock_run, tmp_data):
        """Test SLAM3R backend with mocked subprocess."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="done")

        repo = tmp_data["tmp_path"] / "clone" / "SLAM3R"
        # Create fake output PLY where SLAM3R would write it
        slam3r_out = (
            tmp_data["data_root"]
            / "interim"
            / "s02_slam"
            / "output"
            / "gss_slam"
        )
        self._create_fake_ply(slam3r_out / "gss_slam_recon.ply")

        step = SlamStep(
            config=SlamConfig(
                backend="slam3r",
                repo_path=repo,
                conda_env="slam3r",
            ),
            data_root=tmp_data["data_root"],
        )
        inputs = SlamInput(frames_dir=tmp_data["frames_dir"])
        result = step.run(inputs)

        assert result.surface_points_path.exists()
        assert result.num_surface_points > 0

        # Verify subprocess called recon.py
        call_args = mock_run.call_args[0][0]
        assert "recon.py" in " ".join(str(a) for a in call_args)
        assert "--test_name" in call_args

    @patch("subprocess.run")
    def test_subprocess_failure(self, mock_run, tmp_data):
        """Test that run() raises on subprocess failure."""
        mock_run.return_value = MagicMock(
            returncode=1, stderr="CUDA OOM", stdout=""
        )

        step = SlamStep(
            config=SlamConfig(
                repo_path=tmp_data["tmp_path"] / "clone" / "MASt3R-SLAM",
            ),
            data_root=tmp_data["data_root"],
        )
        inputs = SlamInput(frames_dir=tmp_data["frames_dir"])

        with pytest.raises(RuntimeError, match="failed with exit code"):
            step.run(inputs)


class TestHelpers:
    def test_find_output_ply_expected(self, tmp_path):
        ply = tmp_path / "test.ply"
        ply.write_bytes(b"fake ply")
        assert SlamStep._find_output_ply(tmp_path, ply) == ply

    def test_find_output_ply_fallback(self, tmp_path):
        (tmp_path / "sub").mkdir()
        ply = tmp_path / "sub" / "found.ply"
        ply.write_bytes(b"fake ply")
        result = SlamStep._find_output_ply(tmp_path, tmp_path / "missing.ply")
        assert result == ply

    def test_find_output_ply_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SlamStep._find_output_ply(tmp_path, tmp_path / "missing.ply")

    def test_find_camera_poses_txt(self, tmp_path):
        (tmp_path / "poses.txt").write_text("0 0 0 0 0 0 0 1")
        result = SlamStep._find_camera_poses(tmp_path, tmp_path)
        assert result == tmp_path / "poses.txt"

    def test_find_camera_poses_none(self, tmp_path):
        result = SlamStep._find_camera_poses(tmp_path, tmp_path)
        assert result is None


class TestPipelineConfig:
    def test_pipeline_slam_loads(self):
        from gss.core.pipeline_runner import load_pipeline_config

        cfg = load_pipeline_config(Path("configs/pipeline_slam.yaml"))
        assert cfg.project_name == "gss_slam"
        step_names = [s.name for s in cfg.steps]
        assert "slam" in step_names
        assert "plane_extraction" in step_names

        # Verify slam step depends on extract_frames
        slam_step = next(s for s in cfg.steps if s.name == "slam")
        assert "extract_frames" in slam_step.depends_on

        # Verify plane_extraction depends on slam
        pe_step = next(s for s in cfg.steps if s.name == "plane_extraction")
        assert "slam" in pe_step.depends_on
