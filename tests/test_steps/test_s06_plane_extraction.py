"""Tests for S06: Plane Extraction step."""

import json
from pathlib import Path

import numpy as np
import pytest

from gss.steps.s06_plane_extraction.config import PlaneExtractionConfig
from gss.steps.s06_plane_extraction.contracts import (
    DetectedPlane, PlaneExtractionInput, PlaneExtractionOutput,
)


def _has_deps() -> bool:
    try:
        import open3d, alphashape, shapely
        return True
    except ImportError:
        return False


class TestPlaneContracts:
    def test_detected_plane(self):
        plane = DetectedPlane(
            id=0, normal=[0.0, 0.0, 1.0], d=0.0,
            label="floor", num_inliers=500,
            boundary_3d=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        )
        assert plane.label == "floor"
        assert len(plane.boundary_3d) == 4

    def test_config_defaults(self):
        cfg = PlaneExtractionConfig()
        assert cfg.max_planes == 30
        assert cfg.distance_threshold == 0.02


class TestPlaneClassification:
    def test_horizontal_classification(self):
        from gss.steps.s06_plane_extraction.step import _classify_plane
        # Horizontal normals return "horizontal" (refined to floor/ceiling later)
        assert _classify_plane(np.array([0, 0, 1.0]), 15.0) == "horizontal"
        assert _classify_plane(np.array([0, 0, -1.0]), 15.0) == "horizontal"

    def test_wall_classification(self):
        from gss.steps.s06_plane_extraction.step import _classify_plane
        assert _classify_plane(np.array([1.0, 0, 0]), 15.0) == "wall"
        assert _classify_plane(np.array([0, 1.0, 0]), 15.0) == "wall"

    def test_angled_wall(self):
        from gss.steps.s06_plane_extraction.step import _classify_plane
        normal = np.array([0.7071, 0.7071, 0.0])  # 45 deg horizontal
        assert _classify_plane(normal, 15.0) == "wall"

    def test_refine_horizontal_labels(self):
        from gss.steps.s06_plane_extraction.step import _refine_horizontal_labels
        planes = [
            {"label": "horizontal", "inlier_points": np.array([[0, 0, 0.0]] * 10)},   # z=0 (floor)
            {"label": "horizontal", "inlier_points": np.array([[0, 0, 10.0]] * 10)},  # z=10 (ceiling)
            {"label": "horizontal", "inlier_points": np.array([[0, 0, 5.0]] * 10)},   # z=5 (furniture)
        ]
        _refine_horizontal_labels(planes, up_axis="z")
        assert planes[0]["label"] == "floor"
        assert planes[1]["label"] == "ceiling"
        assert planes[2]["label"] == "other"


class TestPlaneExtractionStep:
    @pytest.mark.skipif(not _has_deps(), reason="open3d/alphashape/shapely not installed")
    def test_extraction_synthetic(self, data_root: Path):
        import open3d as o3d
        from gss.steps.s06_plane_extraction.step import PlaneExtractionStep

        # Create a synthetic point cloud with a clear floor plane
        n_floor = 2000
        floor_pts = np.column_stack([
            np.random.uniform(0, 5, n_floor),
            np.random.uniform(0, 5, n_floor),
            np.random.normal(0, 0.005, n_floor),  # near z=0
        ])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(floor_pts)

        ply_path = data_root / "interim" / "s05_tsdf" / "surface_points.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(ply_path), pcd)

        meta_path = data_root / "interim" / "s05_tsdf" / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump({"num_surface_points": n_floor}, f)

        cfg = PlaneExtractionConfig(max_planes=5, min_inliers=100, normalize_coords=False)
        step = PlaneExtractionStep(config=cfg, data_root=data_root)
        inp = PlaneExtractionInput(surface_points_path=ply_path, metadata_path=meta_path)

        output = step.execute(inp)
        assert output.num_planes >= 1
        assert output.planes_file.exists()
        assert output.boundaries_file.exists()

        with open(output.planes_file) as f:
            planes = json.load(f)
        assert any(p["label"] == "floor" for p in planes)
