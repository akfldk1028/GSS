"""3DGRUT-based training — two paths:

1. RGB path:  frames → COLMAP(SfM) → undistort(PINHOLE) → 3DGRUT(colmap_3dgut) → USDZ
2. Depth path: Azure Kinect depth → pointcloud PLY → 3DGRUT(cusfm_3dgut) → USDZ
               Still needs COLMAP images for supervision, but init from depth pointcloud.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from .config import TrainerConfig

logger = logging.getLogger(__name__)


class IncrementalTrainer:
    """Camera frames → COLMAP or Kinect depth → 3DGRUT train → USDZ."""

    def __init__(self, config: TrainerConfig, data_root: Path) -> None:
        self._config = config
        self._data_root = data_root
        self._colmap_db: Path | None = None

    def train(
        self,
        frame_paths: list[Path],
        cycle_dir: Path,
        pointcloud_ply: Path | None = None,
    ) -> Path:
        """Run full pipeline. If pointcloud_ply given, use depth-init path.

        Returns path to exported USDZ file.
        """
        cycle_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = frame_paths[0].parent

        # Step 1: COLMAP SfM (always needed for camera poses + supervision images)
        colmap_dir = cycle_dir / "colmap"
        self._run_colmap(frames_dir, colmap_dir)

        # Step 2: Undistort to PINHOLE (3DGRUT requirement)
        undist_dir = cycle_dir / "colmap_undistorted"
        self._run_undistort(colmap_dir, frames_dir, undist_dir)

        # Step 3: 3DGRUT train + export
        usdz_path = cycle_dir / "scene.nurec.usdz"
        if pointcloud_ply and pointcloud_ply.exists():
            self._run_3dgrut_depth(undist_dir, pointcloud_ply, cycle_dir, usdz_path)
        else:
            self._run_3dgrut(undist_dir, cycle_dir, usdz_path)

        logger.info(f"Training done: {usdz_path}")
        return usdz_path

    def train_depth_only(
        self,
        pointcloud_ply: Path,
        frame_paths: list[Path],
        cycle_dir: Path,
    ) -> Path:
        """Depth-first path: Kinect pointcloud + RGB frames for supervision."""
        return self.train(frame_paths, cycle_dir, pointcloud_ply=pointcloud_ply)

    def _run_colmap(self, frames_dir: Path, output_dir: Path) -> None:
        """Lightweight COLMAP SfM for camera poses (pycolmap 4.0.2)."""
        import pycolmap

        output_dir.mkdir(parents=True, exist_ok=True)
        db_path = output_dir / "database.db"

        if self._config.reuse_poses and self._colmap_db and self._colmap_db.exists():
            shutil.copy2(self._colmap_db, db_path)
            logger.info("Reusing previous COLMAP database")

        extraction_opts = pycolmap.FeatureExtractionOptions()
        extraction_opts.sift.max_num_features = self._config.colmap_max_features
        pycolmap.extract_features(db_path, frames_dir, extraction_options=extraction_opts)

        if self._config.colmap_matcher == "sequential":
            pycolmap.match_sequential(db_path)
        else:
            pycolmap.match_exhaustive(db_path)

        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        reconstructions = pycolmap.incremental_mapping(db_path, frames_dir, sparse_dir)

        if not reconstructions:
            raise RuntimeError("COLMAP reconstruction failed — no maps produced")

        best_key = min(reconstructions.keys())
        best = reconstructions[best_key]
        out_dir = sparse_dir / "0"
        out_dir.mkdir(exist_ok=True)
        best.write(out_dir)
        logger.info(f"COLMAP: {len(best.images)} images, {len(best.points3D)} points")

        self._colmap_db = db_path

    def _run_undistort(
        self, colmap_dir: Path, frames_dir: Path, output_dir: Path
    ) -> None:
        """Undistort images to PINHOLE model (required by 3DGRUT)."""
        import pycolmap

        pycolmap.undistort_images(
            output_path=output_dir,
            input_path=colmap_dir / "sparse" / "0",
            image_path=frames_dir,
        )

        # 3DGRUT needs text format in sparse/0/
        sparse_dir = output_dir / "sparse"
        rec = pycolmap.Reconstruction(sparse_dir)
        text_dir = sparse_dir / "0"
        text_dir.mkdir(exist_ok=True)
        rec.write_text(text_dir)

        logger.info(f"Undistorted to PINHOLE: {output_dir}")

    def _run_3dgrut(
        self, colmap_undist_dir: Path, output_dir: Path, usdz_path: Path
    ) -> None:
        """RGB path: 3DGRUT train with COLMAP points init."""
        self._exec_3dgrut(
            app="colmap_3dgut",
            colmap_dir=colmap_undist_dir,
            output_dir=output_dir,
            usdz_path=usdz_path,
            extra_args=[],
        )

    def _run_3dgrut_depth(
        self,
        colmap_undist_dir: Path,
        pointcloud_ply: Path,
        output_dir: Path,
        usdz_path: Path,
    ) -> None:
        """Depth path: 3DGRUT train with Kinect pointcloud init (cusfm_3dgut)."""
        logger.info(f"Using depth pointcloud: {pointcloud_ply}")
        self._exec_3dgrut(
            app="cusfm_3dgut",
            colmap_dir=colmap_undist_dir,
            output_dir=output_dir,
            usdz_path=usdz_path,
            extra_args=[
                f"initialization.fused_point_cloud_path={pointcloud_ply.resolve()}",
            ],
        )

    def _exec_3dgrut(
        self,
        app: str,
        colmap_dir: Path,
        output_dir: Path,
        usdz_path: Path,
        extra_args: list[str],
    ) -> None:
        """Execute 3DGRUT train.py subprocess."""
        repo = Path(self._config.threedgrut_repo).resolve()
        train_script = repo / "train.py"

        cmd = [
            "conda", "run", "-n", self._config.threedgrut_conda_env,
            "python", str(train_script),
            f"+apps={app}",
            f"path={colmap_dir}",
            f"out_dir={output_dir / '3dgrut_out'}",
            f"n_iterations={self._config.n_iterations}",
            "export_ply.enabled=true",
            f"export_ply.path={output_dir / 'model.ply'}",
            "export_usd.enabled=true",
            "export_usd.format=nurec",
            f"export_usd.path={usdz_path}",
            "with_gui=false",
            "test_last=false",
            "val_frequency=999999",
            *extra_args,
        ]

        logger.info(f"3DGRUT train ({app}): {self._config.n_iterations} iters")
        result = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)

        if result.returncode != 0:
            stderr_tail = "\n".join(result.stderr.strip().split("\n")[-15:])
            logger.error(f"3DGRUT train failed:\n{stderr_tail}")
            raise RuntimeError(f"3DGRUT train failed (exit {result.returncode})")

        if not usdz_path.exists():
            raise RuntimeError(f"3DGRUT finished but USDZ not found: {usdz_path}")

        logger.info(f"3DGRUT export: {usdz_path} ({usdz_path.stat().st_size / 1e6:.1f}MB)")
