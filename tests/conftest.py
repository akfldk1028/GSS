"""Shared pytest fixtures for GSS pipeline tests."""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Create a temporary data root with standard directory structure."""
    for subdir in ["raw", "interim/s00_import_ply", "interim/s01_frames",
                    "interim/s02_colmap", "interim/s03_gaussians",
                    "interim/s04_depth_maps", "interim/s05_tsdf",
                    "interim/s06_planes", "processed"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def sample_frames_dir(data_root: Path) -> Path:
    """Create sample frames for testing."""
    frames_dir = data_root / "interim" / "s01_frames"
    for i in range(5):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        try:
            import cv2
            cv2.imwrite(str(frames_dir / f"frame_{i:05d}.png"), img)
        except ImportError:
            (frames_dir / f"frame_{i:05d}.png").write_bytes(b"\x00" * 100)
    return frames_dir


@pytest.fixture
def sample_planes_json(data_root: Path) -> Path:
    """Create sample planes.json for testing."""
    planes = [
        {
            "id": 0, "normal": [1.0, 0.0, 0.0], "d": -2.0,
            "label": "wall", "num_inliers": 1000,
            "boundary_3d": [[2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [2.0, 3.0, 2.5], [2.0, 0.0, 2.5]],
        },
        {
            "id": 1, "normal": [0.0, 0.0, 1.0], "d": 0.0,
            "label": "floor", "num_inliers": 2000,
            "boundary_3d": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 3.0, 0.0], [0.0, 3.0, 0.0]],
        },
    ]
    planes_file = data_root / "interim" / "s06_planes" / "planes.json"
    planes_file.parent.mkdir(parents=True, exist_ok=True)
    with open(planes_file, "w") as f:
        json.dump(planes, f)
    return planes_file


@pytest.fixture
def sample_boundaries_json(data_root: Path, sample_planes_json: Path) -> Path:
    """Create sample boundaries.json for testing."""
    with open(sample_planes_json) as f:
        planes = json.load(f)
    boundaries = [{"id": p["id"], "label": p["label"], "boundary_3d": p["boundary_3d"]} for p in planes]
    boundaries_file = data_root / "interim" / "s06_planes" / "boundaries.json"
    with open(boundaries_file, "w") as f:
        json.dump(boundaries, f)
    return boundaries_file


@pytest.fixture
def sample_depth_data(data_root: Path) -> tuple[Path, Path]:
    """Create sample depth maps and poses.json."""
    depth_dir = data_root / "interim" / "s04_depth_maps" / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    views = []
    for i in range(3):
        depth = np.random.uniform(0.5, 3.0, (100, 100)).astype(np.float32)
        fname = f"depth_{i:04d}.npy"
        np.save(str(depth_dir / fname), depth)
        c2w = np.eye(4); c2w[0, 3] = i * 0.5
        views.append({"index": i, "image_name": f"frame_{i:05d}.png", "depth_file": fname,
                       "normal_file": None, "matrix_4x4": c2w.flatten().tolist()})
    poses_data = {
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 50.0, "cy": 50.0, "width": 100, "height": 100},
        "views": views,
    }
    poses_file = data_root / "interim" / "s04_depth_maps" / "poses.json"
    with open(poses_file, "w") as f:
        json.dump(poses_data, f)
    return depth_dir, poses_file


@pytest.fixture
def sample_walls_json(data_root: Path) -> Path:
    """Create sample walls.json for testing (center-line format from s06b)."""
    walls = [
        {
            "id": 0, "plane_ids": [1],
            "center_line_2d": [[-1.0, 3.0], [1.0, 3.0]],
            "thickness": 0.2, "height_range": [-0.5, 2.5],
            "normal_axis": "z",
        },
        {
            "id": 1, "plane_ids": [4],
            "center_line_2d": [[-1.0, -1.0], [-1.0, 3.0]],
            "thickness": 0.2, "height_range": [-0.5, 2.5],
            "normal_axis": "x",
        },
        {
            "id": 2, "plane_ids": [5],
            "center_line_2d": [[1.0, -1.0], [1.0, 3.0]],
            "thickness": 0.2, "height_range": [-0.5, 2.5],
            "normal_axis": "x",
        },
        {
            "id": 3, "plane_ids": [7],
            "center_line_2d": [[-1.0, -1.0], [1.0, -1.0]],
            "thickness": 0.2, "height_range": [-0.5, 2.5],
            "normal_axis": "z",
            "synthetic": True,
        },
    ]
    s06b_dir = data_root / "interim" / "s06b_plane_regularization"
    s06b_dir.mkdir(parents=True, exist_ok=True)
    walls_file = s06b_dir / "walls.json"
    with open(walls_file, "w") as f:
        json.dump(walls, f)
    return walls_file


@pytest.fixture
def sample_spaces_json(data_root: Path) -> Path:
    """Create sample spaces.json for testing (from s06b)."""
    spaces_data = {
        "manhattan_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "coordinate_scale": 1.0,
        "spaces": [
            {
                "id": 0,
                "boundary_2d": [[-1.0, 3.0], [1.0, 3.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 3.0]],
                "area": 8.0,
                "floor_height": -0.5,
                "ceiling_height": 2.5,
            }
        ],
    }
    s06b_dir = data_root / "interim" / "s06b_plane_regularization"
    s06b_dir.mkdir(parents=True, exist_ok=True)
    spaces_file = s06b_dir / "spaces.json"
    with open(spaces_file, "w") as f:
        json.dump(spaces_data, f)
    return spaces_file
