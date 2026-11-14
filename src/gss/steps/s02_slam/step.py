"""Step 02-alt: GS-SLAM — replaces COLMAP(s02) + 3DGS(s03) with a single SLAM step.

Supports three backends via conda subprocess (Pattern C):
  1. MASt3R-SLAM (CVPR 2025) — dense monocular SLAM, 15 FPS
  2. MonoGS (CVPR 2024 Highlight) — first monocular GS SLAM, 3-10 FPS
  3. SLAM3R (CVPR 2025 Highlight) — feed-forward, 20+ FPS, no training

Each backend runs in its own conda env. Output is a dense point cloud PLY
with normals, directly compatible with s06 plane extraction.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import SlamConfig
from .contracts import SlamInput, SlamOutput

logger = logging.getLogger(__name__)

# Default repo paths per backend (relative to project root)
_DEFAULT_REPOS: dict[str, str] = {
    "mast3r_slam": "clone/MASt3R-SLAM",
    "monogs": "clone/MonoGS",
    "slam3r": "clone/SLAM3R",
}


class SlamStep(BaseStep[SlamInput, SlamOutput, SlamConfig]):
    name: ClassVar[str] = "slam"
    input_type: ClassVar = SlamInput
    output_type: ClassVar = SlamOutput
    config_type: ClassVar = SlamConfig

    def _resolve_repo(self) -> Path:
        """Resolve repo path, using backend-specific default if not overridden."""
        cfg_repo = str(self.config.repo_path)
        # If user kept the default, auto-resolve per backend
        if cfg_repo in _DEFAULT_REPOS.values():
            return Path(_DEFAULT_REPOS.get(self.config.backend, cfg_repo))
        return Path(cfg_repo)

    def validate_inputs(self, inputs: SlamInput) -> bool:
        if not inputs.frames_dir.exists():
            logger.error("Frames directory not found: %s", inputs.frames_dir)
            return False

        if self.config.backend not in _DEFAULT_REPOS:
            logger.error(
                "Invalid backend '%s'. Choose from: %s",
                self.config.backend,
                list(_DEFAULT_REPOS.keys()),
            )
            return False

        repo = self._resolve_repo()
        if not repo.exists():
            logger.error(
                "SLAM repo not found at %s — clone it first:\n"
                "  git clone --recursive https://github.com/%s %s",
                repo,
                _GITHUB_URLS.get(self.config.backend, "???"),
                repo,
            )
            return False

        return True

    def run(self, inputs: SlamInput) -> SlamOutput:
        cfg = self.config
        repo = self._resolve_repo().resolve()
        interim_dir = self.data_root / "interim" / "s02_slam"
        interim_dir.mkdir(parents=True, exist_ok=True)

        output_dir = interim_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Run SLAM backend
        raw_ply = self._run_backend(repo, inputs.frames_dir, output_dir)

        # 2. Post-process: voxel downsample + normal estimation
        surface_points_path = interim_dir / "surface_points.ply"
        num_points = self._postprocess(raw_ply, surface_points_path)

        # 3. Save camera poses if available (search PLY parent + output_dir)
        camera_poses_path = self._find_camera_poses(
            output_dir, raw_ply.parent
        )

        # 4. Write metadata for s06 compatibility
        metadata_path = interim_dir / "metadata.json"
        metadata = {
            "method": f"slam_{cfg.backend}",
            "backend": cfg.backend,
            "voxel_downsample": cfg.voxel_downsample,
            "num_surface_points": num_points,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        logger.info(
            "SLAM (%s) complete: %d surface points", cfg.backend, num_points
        )

        return SlamOutput(
            surface_points_path=surface_points_path,
            metadata_path=metadata_path,
            camera_poses_path=camera_poses_path,
            num_surface_points=num_points,
        )

    # ── Backend dispatchers ─────────────────────────────────────────

    def _run_backend(
        self, repo: Path, frames_dir: Path, output_dir: Path
    ) -> Path:
        """Dispatch to the appropriate SLAM backend. Returns raw output PLY."""
        backend = self.config.backend
        if backend == "mast3r_slam":
            return self._run_mast3r_slam(repo, frames_dir, output_dir)
        elif backend == "monogs":
            return self._run_monogs(repo, frames_dir, output_dir)
        elif backend == "slam3r":
            return self._run_slam3r(repo, frames_dir, output_dir)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _run_mast3r_slam(
        self, repo: Path, frames_dir: Path, output_dir: Path
    ) -> Path:
        """Run MASt3R-SLAM (CVPR 2025).

        CLI: python main.py --dataset <folder> --config config/base.yaml
                            --no-viz --save-as <name>
        Output: logs/<save-as>/<dataset_stem>.ply  (+ .txt trajectory)

        MonocularDataset.save_results=True by default (except realsense/webcam).
        RGBFiles loader (fallback) only globs *.png files.
        We use --no-viz for headless mode.
        """
        save_as = "gss_slam"
        args = [
            "python", "main.py",
            "--dataset", str(frames_dir.resolve()),
            "--config", "config/base.yaml",
            "--no-viz",
            "--save-as", save_as,
        ]
        args.extend(self.config.extra_args)
        self._run_subprocess(args, repo, "MASt3R-SLAM")

        # Output: logs/<save-as>/<dataset_stem>.ply
        # seq_name = dataset.dataset_path.stem in evaluate.py
        seq_name = frames_dir.resolve().stem
        logs_dir = repo / "logs" / save_as
        expected_ply = logs_dir / f"{seq_name}.ply"
        return self._find_output_ply(logs_dir, expected_ply)

    def _run_monogs(
        self, repo: Path, frames_dir: Path, output_dir: Path
    ) -> Path:
        """Run MonoGS (CVPR 2024 Highlight).

        CLI: python slam.py --config <yaml> --eval
        MonoGS uses a YAML config specifying dataset_path, calibration, etc.
        We generate a minimal config YAML pointing to the frames directory,
        with save_dir set to our output_dir.

        Output: <save_dir>/<path[-3]_path[-2]>/<datetime>/point_cloud/final/point_cloud.ply
        """
        # Generate a temporary MonoGS config file
        monogs_config = self._create_monogs_config(repo, frames_dir, output_dir)

        args = [
            "python", "slam.py",
            "--config", str(monogs_config),
            "--eval",
        ]
        args.extend(self.config.extra_args)
        self._run_subprocess(args, repo, "MonoGS")

        # MonoGS saves to nested directories under save_dir
        expected = output_dir / "point_cloud" / "final" / "point_cloud.ply"
        return self._find_output_ply(output_dir, expected)

    def _create_monogs_config(
        self, repo: Path, frames_dir: Path, output_dir: Path
    ) -> Path:
        """Generate a MonoGS config YAML for our image folder.

        MonoGS expects a config with Dataset.dataset_path, Dataset.type,
        and Results.save_dir. We inherit from the mono base config and
        override the dataset path.
        """
        import yaml

        # Load the base mono config to inherit defaults
        base_config_path = repo / "configs" / "mono" / "tum" / "base_config.yaml"
        if base_config_path.exists():
            with open(base_config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

        # Override dataset and results settings
        cfg.setdefault("Dataset", {})
        cfg["Dataset"]["dataset_path"] = str(frames_dir.resolve())
        cfg["Dataset"]["type"] = "folder"
        cfg["Dataset"]["sensor_type"] = "monocular"

        cfg.setdefault("Results", {})
        cfg["Results"]["save_results"] = True
        cfg["Results"]["save_dir"] = str(output_dir.resolve())
        cfg["Results"]["use_gui"] = False
        cfg["Results"]["eval_rendering"] = False
        cfg["Results"]["use_wandb"] = False

        # Write temporary config
        config_path = output_dir / "monogs_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        logger.info("MonoGS config written to %s", config_path)
        return config_path

    def _run_slam3r(
        self, repo: Path, frames_dir: Path, output_dir: Path
    ) -> Path:
        """Run SLAM3R (CVPR 2025 Highlight).

        CLI: python recon.py --dataset <folder> --test_name <name>
                             --save_dir <dir> [--online]
        Output: <save_dir>/<test_name>/<test_name>_recon.ply

        SLAM3R is feed-forward: no training loop, outputs dense point cloud.
        """
        test_name = "gss_slam"
        save_dir = str(output_dir.resolve())
        args = [
            "python", "recon.py",
            "--dataset", str(frames_dir.resolve()),
            "--test_name", test_name,
            "--save_dir", save_dir,
            "--num_points_save", "2000000",
        ]
        args.extend(self.config.extra_args)
        self._run_subprocess(args, repo, "SLAM3R")

        # SLAM3R saves to <save_dir>/<test_name>/<test_name>_recon.ply
        expected = output_dir / test_name / f"{test_name}_recon.ply"
        return self._find_output_ply(output_dir, expected)

    # ── Subprocess helper ───────────────────────────────────────────

    def _run_subprocess(
        self, args: list[str], cwd: Path, description: str
    ) -> None:
        """Run a command in the SLAM conda environment."""
        conda_env = self.config.conda_env
        cmd = ["conda", "run", "-n", conda_env] + args

        logger.info("[%s] Running: %s", description, " ".join(str(a) for a in cmd))
        logger.info("  cwd: %s", cwd)

        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=self.config.timeout,
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n")[-20:]:
                logger.info("[%s] %s", description, line)

        if result.returncode != 0:
            logger.error("[%s] FAILED (exit code %d)", description, result.returncode)
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-30:]:
                    logger.error("[%s] %s", description, line)
            raise RuntimeError(
                f"{description} failed with exit code {result.returncode}"
            )

        logger.info("[%s] Completed successfully", description)

    # ── Post-processing ─────────────────────────────────────────────

    def _postprocess(self, raw_ply: Path, output_path: Path) -> int:
        """Voxel downsample + normal estimation for s06 compatibility."""
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(raw_ply))
        if not pcd.has_points():
            raise RuntimeError(f"SLAM output has no points: {raw_ply}")

        num_raw = len(pcd.points)
        logger.info("Raw SLAM output: %d points", num_raw)

        # Voxel downsampling
        if self.config.voxel_downsample > 0:
            pcd = pcd.voxel_down_sample(self.config.voxel_downsample)
            logger.info(
                "After voxel downsample (%.3f): %d points",
                self.config.voxel_downsample,
                len(pcd.points),
            )

        # Normal estimation (required for s06 plane extraction)
        if self.config.estimate_normals and not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_radius,
                    max_nn=self.config.normal_max_nn,
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)
            logger.info("Normals estimated and oriented")

        o3d.io.write_point_cloud(str(output_path), pcd)
        num_points = len(pcd.points)
        logger.info("Surface points saved: %d -> %s", num_points, output_path)
        return num_points

    # ── File finders ────────────────────────────────────────────────

    @staticmethod
    def _find_output_ply(output_dir: Path, expected: Path) -> Path:
        """Find the output PLY, with fallback glob search."""
        if expected.exists():
            return expected
        # Fallback: find any PLY in output dir
        plys = list(output_dir.glob("**/*.ply"))
        if plys:
            found = max(plys, key=lambda p: p.stat().st_mtime)
            logger.warning("Expected PLY not at %s, using %s", expected, found)
            return found
        raise FileNotFoundError(
            f"No PLY output found in {output_dir}. "
            "Check SLAM backend logs for errors."
        )

    @staticmethod
    def _find_camera_poses(output_dir: Path, ply_dir: Path) -> Path | None:
        """Look for camera pose files in multiple directories.

        MASt3R-SLAM: logs/<save-as>/<seq_name>.txt (TUM format)
        MonoGS: pose saved internally (no separate file)
        SLAM3R: no camera pose export
        """
        search_dirs = [output_dir]
        if ply_dir != output_dir:
            search_dirs.append(ply_dir)

        candidates = [
            "poses.txt", "cameras.json", "trajectory.txt",
            "keyframes.txt", "poses.json",
        ]
        for d in search_dirs:
            for name in candidates:
                p = d / name
                if p.exists():
                    return p
            # Glob fallback: .txt files with pose-like names
            for pat in ["*pose*", "*traj*"]:
                matches = list(d.glob(pat))
                if matches:
                    return matches[0]
            # MASt3R-SLAM: <seq_name>.txt (TUM trajectory format)
            txt_files = list(d.glob("*.txt"))
            if txt_files:
                return txt_files[0]
        return None


# GitHub URLs for clone instructions
_GITHUB_URLS: dict[str, str] = {
    "mast3r_slam": "rmurai0610/MASt3R-SLAM.git",
    "monogs": "muskie82/MonoGS.git",
    "slam3r": "PKU-VCL-3DV/SLAM3R.git",
}
