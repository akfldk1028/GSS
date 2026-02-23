"""Step 04: Render depth/normal maps from trained Gaussians."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

from gss.core.step_base import BaseStep
from gss.utils.geometry import make_w2c
from .config import DepthRenderConfig
from .contracts import DepthRenderInput, DepthRenderOutput

logger = logging.getLogger(__name__)


class DepthRenderStep(BaseStep[DepthRenderInput, DepthRenderOutput, DepthRenderConfig]):
    name: ClassVar[str] = "depth_render"
    input_type: ClassVar = DepthRenderInput
    output_type: ClassVar = DepthRenderOutput
    config_type: ClassVar = DepthRenderConfig

    def validate_inputs(self, inputs: DepthRenderInput) -> bool:
        return inputs.model_path.exists() and inputs.sparse_dir.exists()

    def run(self, inputs: DepthRenderInput) -> DepthRenderOutput:
        import torch

        output_dir = self.data_root / "interim" / "s04_depth_maps"
        depth_dir = output_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)

        normal_dir = None
        if self.config.render_normals:
            normal_dir = output_dir / "normals"
            normal_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained model
        checkpoint_path = inputs.model_path.parent / "model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
            means = checkpoint["means"].to(device)
            colors = checkpoint["colors"].to(device)
            log_scales = checkpoint["log_scales"].to(device)
            quats = checkpoint["quats"].to(device)
            raw_opacities = checkpoint["raw_opacities"].to(device)
            method = checkpoint.get("method", "3dgs")
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}. "
                "Depth rendering requires the .pt checkpoint from training."
            )

        # Load camera data from COLMAP
        cameras, images = self._load_colmap_data(inputs.sparse_dir)

        # Select views
        views = self._select_views(images, self.config.num_views, self.config.view_selection)
        logger.info(f"Rendering depth for {len(views)} views")

        # Use standard rasterization with render_mode='D' for depth rendering.
        # Works for both 2DGS and 3DGS trained params (avoids 2DGS return value mismatch).
        from gsplat import rasterization

        scales = torch.exp(log_scales)
        opacities = torch.sigmoid(raw_opacities)
        normalized_quats = quats / quats.norm(dim=-1, keepdim=True)

        poses_data = {"intrinsics": None, "views": []}
        rendered_count = 0

        for idx, img_data in enumerate(views):
            cam_id = img_data["camera_id"]
            cam = cameras.get(cam_id)
            if cam is None:
                continue

            params = cam["params"]
            w = int(cam["width"] * self.config.render_resolution_scale)
            h = int(cam["height"] * self.config.render_resolution_scale)
            scale_factor = self.config.render_resolution_scale

            if len(params) >= 4:
                fx, fy, cx, cy = (
                    params[0] * scale_factor,
                    params[1] * scale_factor,
                    params[2] * scale_factor,
                    params[3] * scale_factor,
                )
            else:
                fx = fy = params[0] * scale_factor
                cx, cy = w / 2, h / 2

            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=torch.float32, device=device)

            # World-to-camera
            w2c = make_w2c(img_data["qvec"], img_data["tvec"])
            viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)

            # Camera-to-world for poses output
            c2w = np.linalg.inv(w2c)

            # Store intrinsics from first view
            if poses_data["intrinsics"] is None:
                poses_data["intrinsics"] = {
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "width": w,
                    "height": h,
                }

            with torch.no_grad():
                # Render depth directly via gsplat's built-in depth mode
                renders, alphas, _meta = rasterization(
                    means=means,
                    quats=normalized_quats,
                    scales=scales,
                    opacities=opacities,
                    colors=means,  # unused in D mode but required by API
                    viewmats=viewmat.unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=w,
                    height=h,
                    render_mode="ED",  # expected depth
                )

                depth_map = renders[0, :, :, 0].cpu().numpy()

            # Save depth map
            depth_filename = f"depth_{rendered_count:04d}.npy"
            np.save(str(depth_dir / depth_filename), depth_map.astype(np.float32))

            # Save normal map if requested
            normal_filename = None
            if self.config.render_normals and normal_dir is not None:
                # Approximate normals from depth gradient
                gy, gx = np.gradient(depth_map)
                normal_map = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
                norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
                norm = np.maximum(norm, 1e-8)
                normal_map = normal_map / norm
                normal_filename = f"normal_{rendered_count:04d}.npy"
                np.save(str(normal_dir / normal_filename), normal_map.astype(np.float32))

            poses_data["views"].append({
                "index": rendered_count,
                "image_name": img_data["name"],
                "depth_file": depth_filename,
                "normal_file": normal_filename,
                "matrix_4x4": c2w.flatten().tolist(),
            })
            rendered_count += 1

        # Save poses file
        poses_file = output_dir / "poses.json"
        with open(poses_file, "w") as f:
            json.dump(poses_data, f, indent=2)

        logger.info(f"Rendered {rendered_count} depth maps")

        return DepthRenderOutput(
            depth_dir=depth_dir,
            normal_dir=normal_dir,
            num_views=rendered_count,
            poses_file=poses_file,
        )

    def _load_colmap_data(self, sparse_dir: Path) -> tuple[dict, list[dict]]:
        """Load cameras and images from COLMAP reconstruction."""
        from gss.utils.io import read_colmap_cameras, read_colmap_images
        return read_colmap_cameras(sparse_dir), read_colmap_images(sparse_dir)

    def _select_views(
        self, images: list[dict], num_views: int, strategy: str
    ) -> list[dict]:
        """Select a subset of views for rendering."""
        if strategy == "all" or num_views >= len(images):
            return images

        if strategy == "uniform":
            indices = np.linspace(0, len(images) - 1, num_views, dtype=int)
            return [images[i] for i in indices]

        # Default: random
        rng = np.random.default_rng(42)
        indices = rng.choice(len(images), size=min(num_views, len(images)), replace=False)
        return [images[i] for i in sorted(indices)]
