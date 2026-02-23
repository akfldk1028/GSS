"""Step 03-alt: PlanarGS (NeurIPS 2025) subprocess wrapper.

Replaces s03+s04+s05 with a single PlanarGS step that runs:
  1. run_geomprior.py  (DUSt3R depth/normal prior)
  2. run_lp3.py        (GroundedSAM planar mask)
  3. train.py          (30K iters GS training with co-planarity loss)
  4. render.py         (depth render + TSDF mesh extraction)

PlanarGS uses its own CUDA rasterizer (diff-plane-rasterization) incompatible
with gsplat, so it runs in a separate conda env via subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import PlanarGSConfig
from .contracts import PlanarGSInput, PlanarGSOutput

logger = logging.getLogger(__name__)


class PlanarGSStep(BaseStep[PlanarGSInput, PlanarGSOutput, PlanarGSConfig]):
    name: ClassVar[str] = "planargs"
    input_type: ClassVar = PlanarGSInput
    output_type: ClassVar = PlanarGSOutput
    config_type: ClassVar = PlanarGSConfig

    def validate_inputs(self, inputs: PlanarGSInput) -> bool:
        if not inputs.frames_dir.exists():
            logger.error(f"Frames directory not found: {inputs.frames_dir}")
            return False
        if not inputs.sparse_dir.exists():
            logger.error(f"COLMAP sparse dir not found: {inputs.sparse_dir}")
            return False
        # Check for COLMAP binary files
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            if not (inputs.sparse_dir / name).exists():
                logger.error(f"Missing COLMAP file: {inputs.sparse_dir / name}")
                return False
        repo = self.config.planargs_repo
        if not repo.exists():
            logger.error(f"PlanarGS repo not found: {repo}")
            return False
        return True

    def run(self, inputs: PlanarGSInput) -> PlanarGSOutput:
        cfg = self.config
        repo = Path(cfg.planargs_repo).resolve()
        interim_dir = self.data_root / "interim" / "s03_planargs"
        scene_dir = interim_dir / "scene"
        output_dir = interim_dir / "output"

        scene_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Prepare scene directory (PlanarGS expected layout)
        self._prepare_scene_dir(scene_dir, inputs)

        # 2. Run PlanarGS substeps
        if not cfg.skip_geomprior:
            self._run_geomprior(repo, scene_dir)
        else:
            logger.info("Skipping geomprior (skip_geomprior=True)")

        if not cfg.skip_lp3:
            self._run_lp3(repo, scene_dir)
        else:
            logger.info("Skipping lp3 (skip_lp3=True)")

        if not cfg.skip_train:
            self._run_train(repo, scene_dir, output_dir)
        else:
            logger.info("Skipping train (skip_train=True)")

        self._run_render(repo, output_dir)

        # 3. Extract mesh vertices as point cloud for s06
        mesh_path = output_dir / "mesh" / "tsdf_fusion_post.ply"
        if not mesh_path.exists():
            # Fallback to non-cleaned mesh
            mesh_path = output_dir / "mesh" / "tsdf_fusion.ply"
        if not mesh_path.exists():
            raise RuntimeError(f"TSDF mesh not found at {mesh_path}")

        surface_points_path = interim_dir / "surface_points.ply"
        num_points = self._mesh_to_pointcloud(mesh_path, surface_points_path)

        # 4. Write metadata for s06 compatibility
        metadata_path = interim_dir / "metadata.json"
        metadata = {
            "method": "planargs",
            "iterations": cfg.iterations,
            "voxel_size": cfg.voxel_size,
            "num_surface_points": num_points,
            "mesh_path": str(mesh_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        logger.info(f"PlanarGS complete: {num_points} surface points, mesh at {mesh_path}")

        return PlanarGSOutput(
            surface_points_path=surface_points_path,
            mesh_path=mesh_path,
            metadata_path=metadata_path,
            num_surface_points=num_points,
        )

    def _prepare_scene_dir(self, scene_dir: Path, inputs: PlanarGSInput) -> None:
        """Create scene directory with junctions/symlinks to GSS data.

        PlanarGS expects:
          <scene>/images/       -> frames
          <scene>/sparse/       -> COLMAP sparse (cameras.bin, images.bin, points3D.bin)

        Note: sparse_dir is typically sparse/0/, but PlanarGS reads from <scene>/sparse/
        directly (no 0/ subfolder).
        """
        images_link = scene_dir / "images"
        sparse_link = scene_dir / "sparse"

        frames_target = inputs.frames_dir.resolve()
        sparse_target = inputs.sparse_dir.resolve()

        self._create_link(images_link, frames_target)
        self._create_link(sparse_link, sparse_target)

        logger.info(f"Scene dir prepared: {scene_dir}")
        logger.info(f"  images -> {frames_target}")
        logger.info(f"  sparse -> {sparse_target}")

    @staticmethod
    def _create_link(link_path: Path, target: Path) -> None:
        """Create a directory junction (Windows) or symlink (Unix).

        If the link already exists and points to the same target, skip.
        If it points elsewhere, remove and recreate.
        """
        if link_path.exists() or link_path.is_symlink():
            # Check if already pointing to correct target
            try:
                if link_path.resolve() == target:
                    return
            except OSError:
                pass
            # Remove stale link
            if sys.platform == "win32":
                # Junction removal: rmdir doesn't delete target contents
                os.system(f'rmdir "{link_path}"')
            else:
                link_path.unlink()

        if sys.platform == "win32":
            # NTFS junction: no admin privileges required
            ret = os.system(f'mklink /J "{link_path}" "{target}"')
            if ret != 0:
                raise RuntimeError(f"Failed to create junction: {link_path} -> {target}")
        else:
            link_path.symlink_to(target)

    def _run_subprocess(self, args: list[str], cwd: Path, description: str) -> None:
        """Run a command in the PlanarGS conda environment."""
        conda_env = self.config.conda_env
        # Build conda run command
        cmd = ["conda", "run", "-n", conda_env] + args

        logger.info(f"[{description}] Running: {' '.join(str(a) for a in cmd)}")
        logger.info(f"  cwd: {cwd}")

        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n")[-20:]:
                logger.info(f"[{description}] {line}")

        if result.returncode != 0:
            logger.error(f"[{description}] FAILED (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-30:]:
                    logger.error(f"[{description}] {line}")
            raise RuntimeError(f"{description} failed with exit code {result.returncode}")

        logger.info(f"[{description}] Completed successfully")

    def _run_geomprior(self, repo: Path, scene_dir: Path) -> None:
        """Run DUSt3R geometric prior generation."""
        self._run_subprocess(
            args=[
                "python", "run_geomprior.py",
                "-s", str(scene_dir.resolve()),
                "--group_size", str(self.config.group_size),
            ],
            cwd=repo,
            description="geomprior",
        )

    def _run_lp3(self, repo: Path, scene_dir: Path) -> None:
        """Run GroundedSAM planar mask generation."""
        self._run_subprocess(
            args=[
                "python", "run_lp3.py",
                "-s", str(scene_dir.resolve()),
                "-t", self.config.text_prompts,
            ],
            cwd=repo,
            description="lp3",
        )

    def _run_train(self, repo: Path, scene_dir: Path, output_dir: Path) -> None:
        """Run PlanarGS training."""
        self._run_subprocess(
            args=[
                "python", "train.py",
                "-s", str(scene_dir.resolve()),
                "-m", str(output_dir.resolve()),
                "--iterations", str(self.config.iterations),
            ],
            cwd=repo,
            description="train",
        )

    def _run_render(self, repo: Path, output_dir: Path) -> None:
        """Run depth rendering + TSDF mesh extraction."""
        self._run_subprocess(
            args=[
                "python", "render.py",
                "-m", str(output_dir.resolve()),
                "--skip_test",
                "--voxel_size", str(self.config.voxel_size),
                "--max_depth", str(self.config.max_depth),
            ],
            cwd=repo,
            description="render",
        )

    @staticmethod
    def _mesh_to_pointcloud(mesh_path: Path, output_path: Path) -> int:
        """Extract mesh vertices as a point cloud PLY file."""
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.has_vertices():
            raise RuntimeError(f"Mesh has no vertices: {mesh_path}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
        if mesh.has_vertex_normals():
            pcd.normals = mesh.vertex_normals

        o3d.io.write_point_cloud(str(output_path), pcd)
        num_points = len(pcd.points)
        logger.info(f"Extracted {num_points} vertices from mesh -> {output_path}")
        return num_points
