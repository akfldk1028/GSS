"""Step 02: COLMAP Structure-from-Motion for camera pose estimation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import ClassVar

from gss.core.step_base import BaseStep
from gss.utils.subprocess_utils import run_command
from .config import ColmapConfig
from .contracts import ColmapInput, ColmapOutput

logger = logging.getLogger(__name__)


def _count_colmap_images(sparse_dir: Path) -> tuple[int, int]:
    """Read COLMAP reconstruction to count registered images and 3D points."""
    try:
        import pycolmap

        recon = pycolmap.Reconstruction(str(sparse_dir))
        return len(recon.images), len(recon.points3D)
    except ImportError:
        pass

    # Fallback: parse images.txt if exists
    images_txt = sparse_dir / "images.txt"
    if images_txt.exists():
        lines = images_txt.read_text().splitlines()
        # images.txt format: comment lines start with #, then pairs of lines per image
        data_lines = [l for l in lines if not l.startswith("#") and l.strip()]
        n_images = len(data_lines) // 2
        return n_images, 0

    # Count .bin files as existence check
    return 0, 0


class ColmapStep(BaseStep[ColmapInput, ColmapOutput, ColmapConfig]):
    name: ClassVar[str] = "colmap"
    input_type: ClassVar = ColmapInput
    output_type: ClassVar = ColmapOutput
    config_type: ClassVar = ColmapConfig

    def validate_inputs(self, inputs: ColmapInput) -> bool:
        if not inputs.frames_dir.exists():
            logger.error(f"Frames directory not found: {inputs.frames_dir}")
            return False
        frames = list(inputs.frames_dir.glob("*.png")) + list(inputs.frames_dir.glob("*.jpg"))
        if len(frames) < 3:
            logger.error(f"Need at least 3 frames, found {len(frames)}")
            return False
        return True

    def run(self, inputs: ColmapInput) -> ColmapOutput:
        output_dir = self.data_root / "interim" / "s02_colmap"
        output_dir.mkdir(parents=True, exist_ok=True)

        database_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # Try pycolmap Python API first, fallback to CLI
        try:
            return self._run_pycolmap(inputs, output_dir, database_path, sparse_dir)
        except ImportError:
            logger.info("pycolmap not available, falling back to COLMAP CLI")
            return self._run_cli(inputs, output_dir, database_path, sparse_dir)

    def _run_pycolmap(
        self, inputs: ColmapInput, output_dir: Path, database_path: Path, sparse_dir: Path
    ) -> ColmapOutput:
        """Run COLMAP via pycolmap Python bindings."""
        import pycolmap

        # 1. Feature extraction
        logger.info("Extracting features with pycolmap...")
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.sift.max_num_features = self.config.max_num_features

        camera_mode = (
            pycolmap.CameraMode.SINGLE if self.config.single_camera else pycolmap.CameraMode.AUTO
        )

        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(inputs.frames_dir),
            camera_mode=camera_mode,
            camera_model=self.config.camera_model,
            extraction_options=extraction_options,
        )

        # 2. Feature matching
        logger.info(f"Matching features ({self.config.matcher})...")
        if self.config.matcher == "sequential":
            pairing_options = pycolmap.SequentialPairingOptions()
            pairing_options.overlap = self.config.match_window
            pycolmap.match_sequential(
                database_path=str(database_path),
                pairing_options=pairing_options,
            )
        elif self.config.matcher == "exhaustive":
            pycolmap.match_exhaustive(database_path=str(database_path))
        elif self.config.matcher == "vocab_tree":
            pycolmap.match_exhaustive(database_path=str(database_path))
            logger.warning("vocab_tree matcher not directly supported, using exhaustive")

        # 3. Incremental mapping
        logger.info("Running incremental mapping...")
        mapper_options = pycolmap.IncrementalPipelineOptions()
        maps = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(inputs.frames_dir),
            output_path=str(sparse_dir),
            options=mapper_options,
        )

        if not maps:
            raise RuntimeError("COLMAP failed to produce any reconstruction")

        # Use the largest reconstruction
        best_idx = max(maps, key=lambda k: maps[k].num_reg_images())
        best = maps[best_idx]

        result_dir = sparse_dir / "0"
        result_dir.mkdir(parents=True, exist_ok=True)
        best.write(result_dir)

        num_registered = best.num_reg_images()
        num_points3d = len(best.points3D)

        logger.info(f"Reconstruction: {num_registered} images, {num_points3d} 3D points")

        return ColmapOutput(
            sparse_dir=result_dir,
            cameras_file=result_dir / "cameras.bin",
            images_file=result_dir / "images.bin",
            num_registered=num_registered,
            num_points3d=num_points3d,
        )

    def _run_cli(
        self, inputs: ColmapInput, output_dir: Path, database_path: Path, sparse_dir: Path
    ) -> ColmapOutput:
        """Run COLMAP via command-line interface."""
        colmap_bin = shutil.which("colmap")
        if colmap_bin is None:
            raise RuntimeError(
                "Neither pycolmap nor COLMAP CLI found. "
                "Install pycolmap: pip install pycolmap, "
                "or install COLMAP: https://colmap.github.io/"
            )

        gpu_flag = "1" if self.config.use_gpu else "0"

        # 1. Feature extraction
        logger.info("Extracting features via COLMAP CLI...")
        run_command([
            colmap_bin, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(inputs.frames_dir),
            "--SiftExtraction.max_num_features", str(self.config.max_num_features),
            "--SiftExtraction.use_gpu", gpu_flag,
            "--ImageReader.camera_model", self.config.camera_model,
            *(["--ImageReader.single_camera", "1"] if self.config.single_camera else []),
        ])

        # 2. Feature matching
        logger.info(f"Matching features via COLMAP CLI ({self.config.matcher})...")
        if self.config.matcher == "sequential":
            run_command([
                colmap_bin, "sequential_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", gpu_flag,
                "--SequentialMatching.overlap", str(self.config.match_window),
            ])
        else:
            run_command([
                colmap_bin, "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", gpu_flag,
            ])

        # 3. Incremental mapping
        logger.info("Running mapper via COLMAP CLI...")

        run_command([
            colmap_bin, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(inputs.frames_dir),
            "--output_path", str(sparse_dir),
        ])

        # Find the best reconstruction
        recon_dirs = sorted(sparse_dir.iterdir())
        if not recon_dirs:
            raise RuntimeError("COLMAP mapper produced no reconstruction")

        result_dir = recon_dirs[0]
        num_registered, num_points3d = _count_colmap_images(result_dir)

        cameras_file = result_dir / "cameras.bin"
        if not cameras_file.exists():
            cameras_file = result_dir / "cameras.txt"
        images_file = result_dir / "images.bin"
        if not images_file.exists():
            images_file = result_dir / "images.txt"

        logger.info(f"CLI Reconstruction: {num_registered} images, {num_points3d} 3D points")

        return ColmapOutput(
            sparse_dir=result_dir,
            cameras_file=cameras_file,
            images_file=images_file,
            num_registered=num_registered,
            num_points3d=num_points3d,
        )
