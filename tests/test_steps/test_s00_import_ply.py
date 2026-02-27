"""Tests for S00: Import external PLY step."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from plyfile import PlyData, PlyElement

from gss.steps.s00_import_ply.config import ImportPlyConfig
from gss.steps.s00_import_ply.contracts import ImportPlyInput, ImportPlyOutput
from gss.steps.s00_import_ply.step import (
    ImportPlyStep,
    _detect_ply_format,
    _load_gaussian_ply,
    _load_pointcloud_ply,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gaussian_ply(path: Path, n: int = 200, with_opacity: bool = True) -> Path:
    """Create a synthetic 3DGS Gaussian PLY file with standard properties."""
    rng = np.random.default_rng(42)
    xyz = rng.standard_normal((n, 3)).astype(np.float32)

    # Minimal Gaussian properties: f_dc (3 SH), scale (3), rot (4), opacity (1)
    f_dc = rng.standard_normal((n, 3)).astype(np.float32)
    scale = rng.uniform(-5, 1, (n, 3)).astype(np.float32)
    rot = rng.standard_normal((n, 4)).astype(np.float32)

    dtype_fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    if with_opacity:
        dtype_fields.append(("opacity", "f4"))

    vertex_data = np.empty(n, dtype=dtype_fields)
    vertex_data["x"] = xyz[:, 0]
    vertex_data["y"] = xyz[:, 1]
    vertex_data["z"] = xyz[:, 2]
    vertex_data["f_dc_0"] = f_dc[:, 0]
    vertex_data["f_dc_1"] = f_dc[:, 1]
    vertex_data["f_dc_2"] = f_dc[:, 2]
    vertex_data["scale_0"] = scale[:, 0]
    vertex_data["scale_1"] = scale[:, 1]
    vertex_data["scale_2"] = scale[:, 2]
    vertex_data["rot_0"] = rot[:, 0]
    vertex_data["rot_1"] = rot[:, 1]
    vertex_data["rot_2"] = rot[:, 2]
    vertex_data["rot_3"] = rot[:, 3]
    if with_opacity:
        # Pre-sigmoid values: large positive = high opacity, large negative = low
        opacity = np.concatenate([
            rng.uniform(2.0, 5.0, n // 2),      # high opacity (sigmoid > 0.88)
            rng.uniform(-5.0, -2.0, n - n // 2), # low opacity (sigmoid < 0.12)
        ]).astype(np.float32)
        vertex_data["opacity"] = opacity

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], text=False).write(str(path))
    return path


def _make_pointcloud_ply(path: Path, n: int = 300) -> Path:
    """Create a standard point cloud PLY file (just XYZ)."""
    rng = np.random.default_rng(123)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rng.standard_normal((n, 3)))
    o3d.io.write_point_cloud(str(path), pcd)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_ply(tmp_path: Path) -> Path:
    return _make_gaussian_ply(tmp_path / "gaussians.ply")


@pytest.fixture
def gaussian_ply_no_opacity(tmp_path: Path) -> Path:
    return _make_gaussian_ply(tmp_path / "gaussians_noopac.ply", with_opacity=False)


@pytest.fixture
def pointcloud_ply(tmp_path: Path) -> Path:
    return _make_pointcloud_ply(tmp_path / "cloud.ply")


# ---------------------------------------------------------------------------
# A. Format Detection
# ---------------------------------------------------------------------------

class TestDetectPlyFormat:
    def test_gaussian_ply_detected(self, gaussian_ply: Path):
        assert _detect_ply_format(gaussian_ply) == "gaussian_splat"

    def test_pointcloud_ply_detected(self, pointcloud_ply: Path):
        assert _detect_ply_format(pointcloud_ply) == "pointcloud"

    def test_gaussian_without_opacity_still_detected(self, gaussian_ply_no_opacity: Path):
        assert _detect_ply_format(gaussian_ply_no_opacity) == "gaussian_splat"


# ---------------------------------------------------------------------------
# B. Gaussian PLY Loading
# ---------------------------------------------------------------------------

class TestLoadGaussianPly:
    def test_loads_xyz(self, gaussian_ply: Path):
        xyz = _load_gaussian_ply(gaussian_ply, min_opacity=0.0)
        assert xyz.shape == (200, 3)
        assert xyz.dtype == np.float64

    def test_opacity_filter_removes_low(self, gaussian_ply: Path):
        xyz = _load_gaussian_ply(gaussian_ply, min_opacity=0.5)
        # First 100 have high opacity (sigmoid > 0.88), rest low (sigmoid < 0.12)
        assert 80 <= len(xyz) <= 120  # approximately half kept

    def test_no_opacity_no_filter(self, gaussian_ply_no_opacity: Path):
        xyz = _load_gaussian_ply(gaussian_ply_no_opacity, min_opacity=0.5)
        assert len(xyz) == 200  # all kept, no opacity to filter on


# ---------------------------------------------------------------------------
# C. Point Cloud PLY Loading
# ---------------------------------------------------------------------------

class TestLoadPointcloudPly:
    def test_loads_points(self, pointcloud_ply: Path):
        pcd = _load_pointcloud_ply(pointcloud_ply)
        assert len(pcd.points) == 300

    def test_empty_ply_raises(self, tmp_path: Path):
        empty_pcd = o3d.geometry.PointCloud()
        empty_path = tmp_path / "empty.ply"
        o3d.io.write_point_cloud(str(empty_path), empty_pcd)
        with pytest.raises(RuntimeError, match="no points"):
            _load_pointcloud_ply(empty_path)


# ---------------------------------------------------------------------------
# D. Full Step Integration
# ---------------------------------------------------------------------------

class TestImportPlyStep:
    def test_gaussian_ply_full_step(self, gaussian_ply: Path, data_root: Path):
        config = ImportPlyConfig(min_opacity=0.5, remove_outliers=False, estimate_normals=False)
        step = ImportPlyStep(config=config, data_root=data_root)
        inp = ImportPlyInput(ply_path=gaussian_ply)

        result = step.execute(inp)

        assert isinstance(result, ImportPlyOutput)
        assert result.surface_points_path.exists()
        assert result.metadata_path.exists()
        assert result.num_surface_points > 0

        # Verify s06 can read the output
        pcd = o3d.io.read_point_cloud(str(result.surface_points_path))
        assert len(pcd.points) == result.num_surface_points

        # Verify metadata schema
        with open(result.metadata_path) as f:
            meta = json.load(f)
        assert meta["source_format"] == "gaussian_splat"
        assert meta["num_surface_points"] == result.num_surface_points

    def test_pointcloud_ply_full_step(self, pointcloud_ply: Path, data_root: Path):
        config = ImportPlyConfig(remove_outliers=False, estimate_normals=False)
        step = ImportPlyStep(config=config, data_root=data_root)
        inp = ImportPlyInput(ply_path=pointcloud_ply)

        result = step.execute(inp)

        assert result.surface_points_path.exists()
        assert result.num_surface_points == 300

        with open(result.metadata_path) as f:
            meta = json.load(f)
        assert meta["source_format"] == "pointcloud"

    def test_outlier_removal_reduces_points(self, gaussian_ply: Path, data_root: Path):
        config = ImportPlyConfig(
            min_opacity=0.0, remove_outliers=True,
            outlier_nb_neighbors=10, outlier_std_ratio=1.0,
            estimate_normals=False,
        )
        step = ImportPlyStep(config=config, data_root=data_root)
        result = step.execute(ImportPlyInput(ply_path=gaussian_ply))

        # Outlier removal should remove some points
        assert result.num_surface_points < 200

    def test_voxel_downsample(self, pointcloud_ply: Path, data_root: Path):
        config = ImportPlyConfig(
            remove_outliers=False, estimate_normals=False, voxel_downsample=1.0
        )
        step = ImportPlyStep(config=config, data_root=data_root)
        result = step.execute(ImportPlyInput(ply_path=pointcloud_ply))

        # Aggressive downsampling should reduce point count
        assert result.num_surface_points < 300

    def test_normal_estimation(self, pointcloud_ply: Path, data_root: Path):
        config = ImportPlyConfig(remove_outliers=False, estimate_normals=True)
        step = ImportPlyStep(config=config, data_root=data_root)
        result = step.execute(ImportPlyInput(ply_path=pointcloud_ply))

        pcd = o3d.io.read_point_cloud(str(result.surface_points_path))
        assert pcd.has_normals()

    def test_validate_missing_file(self, data_root: Path):
        config = ImportPlyConfig()
        step = ImportPlyStep(config=config, data_root=data_root)
        inp = ImportPlyInput(ply_path=Path("/nonexistent/file.ply"))
        assert step.validate_inputs(inp) is False

    def test_validate_wrong_extension(self, tmp_path: Path, data_root: Path):
        bad_file = tmp_path / "data.obj"
        bad_file.write_text("not a ply")
        config = ImportPlyConfig()
        step = ImportPlyStep(config=config, data_root=data_root)
        assert step.validate_inputs(ImportPlyInput(ply_path=bad_file)) is False

    def test_output_dir_created(self, pointcloud_ply: Path, data_root: Path):
        config = ImportPlyConfig(remove_outliers=False, estimate_normals=False)
        step = ImportPlyStep(config=config, data_root=data_root)
        result = step.execute(ImportPlyInput(ply_path=pointcloud_ply))

        output_dir = data_root / "interim" / "s00_import_ply"
        assert output_dir.exists()
        assert (output_dir / "surface_points.ply").exists()
        assert (output_dir / "metadata.json").exists()
