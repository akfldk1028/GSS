"""Step 00: Import external PLY (3DGS Gaussian or plain point cloud) into pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from gss.core.step_base import BaseStep
from .config import ImportPlyConfig
from .contracts import ImportPlyInput, ImportPlyOutput

logger = logging.getLogger(__name__)


def _detect_ply_format(ply_path: Path) -> Literal["gaussian_splat", "pointcloud"]:
    """Detect whether a PLY file is a 3DGS Gaussian splat or a plain point cloud.

    Gaussian PLY files contain properties like f_dc_0, scale_0, rot_0 that are
    not present in standard point cloud PLY files.
    """
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    prop_names = {p.name for p in vertex.properties}

    # Gaussian splat markers: SH coefficients, scale, rotation
    gaussian_markers = {"f_dc_0", "scale_0", "rot_0"}
    if gaussian_markers.issubset(prop_names):
        return "gaussian_splat"
    return "pointcloud"


def _load_gaussian_ply(ply_path: Path, min_opacity: float) -> np.ndarray:
    """Load a 3DGS Gaussian PLY and extract XYZ centers, optionally filtering by opacity.

    Returns:
        (N, 3) array of XYZ positions after opacity filtering.
    """
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    prop_names = {p.name for p in vertex.properties}

    xyz = np.column_stack([
        vertex["x"].astype(np.float64),
        vertex["y"].astype(np.float64),
        vertex["z"].astype(np.float64),
    ])
    logger.info(f"Loaded {len(xyz)} Gaussians from {ply_path.name}")

    # Filter by opacity if available
    if "opacity" in prop_names:
        raw_opacity = vertex["opacity"].astype(np.float64)
        # 3DGS stores pre-sigmoid opacity; apply sigmoid
        opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
        mask = opacity >= min_opacity
        xyz = xyz[mask]
        logger.info(
            f"Opacity filter (>= {min_opacity}): {mask.sum()}/{len(mask)} Gaussians kept"
        )

    return xyz


def _load_pointcloud_ply(ply_path: Path):
    """Load a standard point cloud PLY via Open3D."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.has_points():
        raise RuntimeError(f"PLY file has no points: {ply_path}")
    logger.info(f"Loaded {len(pcd.points)} points from {ply_path.name}")
    return pcd


class ImportPlyStep(BaseStep[ImportPlyInput, ImportPlyOutput, ImportPlyConfig]):
    """Import an external PLY file (3DGS Gaussian or plain point cloud) into the pipeline.

    Produces surface_points.ply + metadata.json compatible with s06 PlaneExtraction.
    """

    name: ClassVar[str] = "import_ply"
    input_type: ClassVar = ImportPlyInput
    output_type: ClassVar = ImportPlyOutput
    config_type: ClassVar = ImportPlyConfig

    def validate_inputs(self, inputs: ImportPlyInput) -> bool:
        if not inputs.ply_path.exists():
            logger.error(f"PLY file not found: {inputs.ply_path}")
            return False
        if inputs.ply_path.suffix.lower() != ".ply":
            logger.error(f"Expected .ply file, got: {inputs.ply_path.suffix}")
            return False
        return True

    def run(self, inputs: ImportPlyInput) -> ImportPlyOutput:
        import open3d as o3d

        output_dir = self.data_root / "interim" / "s00_import_ply"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Detect format and load ---
        ply_format = _detect_ply_format(inputs.ply_path)
        logger.info(f"Detected PLY format: {ply_format}")

        if ply_format == "gaussian_splat":
            xyz = _load_gaussian_ply(inputs.ply_path, self.config.min_opacity)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
        else:
            pcd = _load_pointcloud_ply(inputs.ply_path)

        # --- 2. Voxel downsampling (optional) ---
        if self.config.voxel_downsample > 0:
            before = len(pcd.points)
            pcd = pcd.voxel_down_sample(self.config.voxel_downsample)
            logger.info(
                f"Voxel downsample ({self.config.voxel_downsample}): "
                f"{before} -> {len(pcd.points)} points"
            )

        # --- 3. Statistical outlier removal ---
        if self.config.remove_outliers and len(pcd.points) > self.config.outlier_nb_neighbors:
            before = len(pcd.points)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.config.outlier_nb_neighbors,
                std_ratio=self.config.outlier_std_ratio,
            )
            logger.info(f"Outlier removal: {before} -> {len(pcd.points)} points")

        # --- 4. Normal estimation ---
        if self.config.estimate_normals and not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            logger.info("Estimated normals via KDTree")

        # --- 5. Save outputs ---
        num_points = len(pcd.points)
        if num_points == 0:
            raise RuntimeError("No points remaining after processing. Check input PLY or config.")

        surface_path = output_dir / "surface_points.ply"
        o3d.io.write_point_cloud(str(surface_path), pcd)
        logger.info(f"Saved {num_points} surface points -> {surface_path}")

        metadata = {
            "source": str(inputs.ply_path),
            "source_format": ply_format,
            "num_surface_points": num_points,
            "min_opacity": self.config.min_opacity if ply_format == "gaussian_splat" else None,
            "voxel_downsample": self.config.voxel_downsample,
            "outlier_removal": self.config.remove_outliers,
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return ImportPlyOutput(
            surface_points_path=surface_path,
            metadata_path=metadata_path,
            num_surface_points=num_points,
        )
