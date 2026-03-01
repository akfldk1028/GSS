"""Step 05: TSDF fusion from depth maps using Open3D."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

from gss.core.step_base import BaseStep
from .config import TsdfFusionConfig
from .contracts import TsdfFusionInput, TsdfFusionOutput

logger = logging.getLogger(__name__)


class TsdfFusionStep(BaseStep[TsdfFusionInput, TsdfFusionOutput, TsdfFusionConfig]):
    name: ClassVar[str] = "tsdf_fusion"
    input_type: ClassVar = TsdfFusionInput
    output_type: ClassVar = TsdfFusionOutput
    config_type: ClassVar = TsdfFusionConfig

    def validate_inputs(self, inputs: TsdfFusionInput) -> bool:
        return inputs.depth_dir.exists() and inputs.poses_file.exists()

    def run(self, inputs: TsdfFusionInput) -> TsdfFusionOutput:
        import open3d as o3d

        output_dir = self.data_root / "interim" / "s05_tsdf"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load camera poses and intrinsics
        with open(inputs.poses_file) as f:
            poses_data = json.load(f)

        intrinsics_data = poses_data["intrinsics"]
        views = poses_data["views"]

        width = intrinsics_data["width"]
        height = intrinsics_data["height"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsics_data["fx"],
            intrinsics_data["fy"],
            intrinsics_data["cx"],
            intrinsics_data["cy"],
        )

        # Create TSDF volume
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config.voxel_size,
            sdf_trunc=self.config.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

        # Integrate each depth map
        integrated = 0
        for view in views:
            depth_path = inputs.depth_dir / view["depth_file"]
            if not depth_path.exists():
                logger.warning(f"Depth file missing: {depth_path}")
                continue

            depth_array = np.load(str(depth_path))
            # Clip to max depth (in raw units before scaling)
            depth_array = np.clip(
                depth_array, 0, self.config.depth_trunc * self.config.depth_scale
            ).astype(np.float32)

            depth_image = o3d.geometry.Image(depth_array)
            # Create dummy color image (grayscale zeros) since we don't need color
            color_array = np.zeros((height, width, 3), dtype=np.uint8)
            color_image = o3d.geometry.Image(color_array)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=self.config.depth_scale,  # Open3D divides by this
                depth_trunc=self.config.depth_trunc,
                convert_rgb_to_intensity=False,
            )

            # 4x4 camera-to-world -> world-to-camera (Open3D expects extrinsic)
            c2w = np.array(view["matrix_4x4"]).reshape(4, 4)
            w2c = np.linalg.inv(c2w)

            volume.integrate(rgbd, intrinsic, w2c)
            integrated += 1

        logger.info(f"Integrated {integrated}/{len(views)} depth maps")

        # Extract point cloud
        pcd = volume.extract_point_cloud()
        surface_path = output_dir / "surface_points.ply"
        o3d.io.write_point_cloud(str(surface_path), pcd)

        num_points = len(pcd.points)
        logger.info(f"Extracted {num_points} surface points")

        # Extract triangle mesh (preserved for s06d mesh segmentation)
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh_path = output_dir / "surface_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)

        num_mesh_verts = len(mesh.vertices)
        num_mesh_faces = len(mesh.triangles)
        logger.info(f"Extracted mesh: {num_mesh_verts} vertices, {num_mesh_faces} faces")

        # Save metadata
        metadata = {
            "voxel_size": self.config.voxel_size,
            "sdf_trunc": self.config.sdf_trunc,
            "depth_trunc": self.config.depth_trunc,
            "num_integrated_views": integrated,
            "num_surface_points": num_points,
            "num_mesh_vertices": num_mesh_verts,
            "num_mesh_faces": num_mesh_faces,
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return TsdfFusionOutput(
            tsdf_dir=output_dir,
            surface_points_path=surface_path,
            num_surface_points=num_points,
            metadata_path=metadata_path,
            surface_mesh_path=mesh_path,
            num_mesh_vertices=num_mesh_verts,
            num_mesh_faces=num_mesh_faces,
        )
