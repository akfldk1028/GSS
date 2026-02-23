"""End-to-end pipeline test: Replica room0, s01 → s07.

Usage:
    python scripts/test_full_pipeline.py [--skip-video] [--iterations 3000]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_test")

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
REPLICA_FRAMES = ROOT / "data" / "test" / "replica" / "Replica" / "room0" / "results"
VIDEO_PATH = DATA_ROOT / "raw" / "replica_room0.mp4"


def phase_video(max_frames: int = 200, fps: float = 30.0) -> None:
    """Phase 0.5: Create MP4 from Replica frames."""
    logger.info("=== Creating test video from Replica frames ===")

    from scripts.create_test_video import create_video

    n = create_video(REPLICA_FRAMES, VIDEO_PATH, fps=fps, max_frames=max_frames)
    assert n > 0, f"Video creation failed: wrote {n} frames"
    assert VIDEO_PATH.exists(), f"Video not found at {VIDEO_PATH}"
    logger.info(f"Video created: {n} frames")


def phase_s01() -> Path:
    """Phase 1: Extract frames from video."""
    logger.info("=== Phase 1: s01 Extract Frames ===")

    from gss.steps.s01_extract_frames import ExtractFramesStep, ExtractFramesInput, ExtractFramesConfig

    cfg = ExtractFramesConfig(target_fps=2.0, max_frames=100, blur_threshold=50.0)
    step = ExtractFramesStep(config=cfg, data_root=DATA_ROOT)
    inp = ExtractFramesInput(video_path=VIDEO_PATH)

    assert step.validate_inputs(inp), "s01 validate_inputs failed"

    out = step.run(inp)
    logger.info(f"s01 result: {out.frame_count} frames in {out.frames_dir}")

    assert out.frames_dir.exists()
    assert out.frame_count >= 5, f"Expected >= 5 frames, got {out.frame_count}"

    return out.frames_dir


def phase_s02(frames_dir: Path) -> tuple[Path, int, int]:
    """Phase 2: COLMAP SfM."""
    logger.info("=== Phase 2: s02 COLMAP ===")

    from gss.steps.s02_colmap import ColmapStep, ColmapInput, ColmapConfig

    cfg = ColmapConfig(matcher="exhaustive", camera_model="PINHOLE", single_camera=True)
    step = ColmapStep(config=cfg, data_root=DATA_ROOT)
    inp = ColmapInput(frames_dir=frames_dir)

    assert step.validate_inputs(inp), "s02 validate_inputs failed"

    out = step.run(inp)
    logger.info(f"s02 result: {out.num_registered} images, {out.num_points3d} 3D points")

    assert out.sparse_dir.exists(), f"Sparse dir not found: {out.sparse_dir}"
    assert out.num_registered >= 5, f"Expected >= 5 registered, got {out.num_registered}"
    assert out.num_points3d >= 100, f"Expected >= 100 3D points, got {out.num_points3d}"

    return out.sparse_dir, out.num_registered, out.num_points3d


def phase_s03(frames_dir: Path, sparse_dir: Path, iterations: int = 3000) -> Path:
    """Phase 3: Gaussian Splatting training."""
    logger.info(f"=== Phase 3: s03 Gaussian Splatting ({iterations} iters) ===")

    from gss.steps.s03_gaussian_splatting import (
        GaussianSplattingStep, GaussianSplattingInput, GaussianSplattingConfig,
    )

    cfg = GaussianSplattingConfig(
        method="2dgs",
        iterations=iterations,
        densify_from=300,
        densify_until=min(2000, iterations),
        max_gaussians=100_000,
    )
    step = GaussianSplattingStep(config=cfg, data_root=DATA_ROOT)
    inp = GaussianSplattingInput(frames_dir=frames_dir, sparse_dir=sparse_dir)

    assert step.validate_inputs(inp), "s03 validate_inputs failed"

    out = step.run(inp)
    logger.info(f"s03 result: {out.num_gaussians} gaussians, model at {out.model_path}")

    assert out.model_path.exists(), f"PLY not found: {out.model_path}"
    assert (out.model_path.parent / "model.pt").exists(), "model.pt not found"
    assert out.num_gaussians >= 50, f"Expected >= 50 gaussians, got {out.num_gaussians}"

    return out.model_path


def phase_s04(model_path: Path, sparse_dir: Path) -> tuple[Path, Path]:
    """Phase 4: Depth rendering."""
    logger.info("=== Phase 4: s04 Depth Render ===")

    from gss.steps.s04_depth_render import DepthRenderStep, DepthRenderInput, DepthRenderConfig

    cfg = DepthRenderConfig(num_views=50, render_normals=False, view_selection="uniform")
    step = DepthRenderStep(config=cfg, data_root=DATA_ROOT)
    inp = DepthRenderInput(model_path=model_path, sparse_dir=sparse_dir)

    assert step.validate_inputs(inp), "s04 validate_inputs failed"

    out = step.run(inp)
    logger.info(f"s04 result: {out.num_views} depth maps in {out.depth_dir}")

    assert out.depth_dir.exists()
    assert out.poses_file.exists()
    npy_files = list(out.depth_dir.glob("*.npy"))
    assert len(npy_files) >= 1, f"Expected depth .npy files, found {len(npy_files)}"

    return out.depth_dir, out.poses_file


def phase_s05(depth_dir: Path, poses_file: Path) -> Path:
    """Phase 5: TSDF Fusion."""
    logger.info("=== Phase 5: s05 TSDF Fusion ===")

    from gss.steps.s05_tsdf_fusion import TsdfFusionStep, TsdfFusionInput, TsdfFusionConfig

    # Depths from gsplat are in COLMAP coordinate units (not meters).
    # Replica room0: depth range ~8-39 COLMAP units, so depth_trunc must exceed that.
    cfg = TsdfFusionConfig(voxel_size=0.08, sdf_trunc=0.3, depth_trunc=50.0, depth_scale=1.0)
    step = TsdfFusionStep(config=cfg, data_root=DATA_ROOT)
    inp = TsdfFusionInput(depth_dir=depth_dir, poses_file=poses_file)

    assert step.validate_inputs(inp), "s05 validate_inputs failed"

    out = step.run(inp)
    logger.info(f"s05 result: {out.num_surface_points} surface points")

    assert out.surface_points_path.exists()
    assert out.num_surface_points >= 100, f"Expected >= 100 points, got {out.num_surface_points}"

    return out.surface_points_path


def phase_s06(surface_points_path: Path) -> tuple[Path, Path]:
    """Phase 6: Plane extraction."""
    logger.info("=== Phase 6: s06 Plane Extraction ===")

    from gss.steps.s06_plane_extraction import (
        PlaneExtractionStep, PlaneExtractionInput, PlaneExtractionConfig,
    )

    # COLMAP units are ~3-8x larger than meters; scale thresholds accordingly
    cfg = PlaneExtractionConfig(
        max_planes=20, distance_threshold=0.15, min_inliers=500,
        ransac_iterations=1000, simplify_tolerance=0.2,
    )
    step = PlaneExtractionStep(config=cfg, data_root=DATA_ROOT)

    # metadata_path is required by contract; use s05's metadata.json
    metadata_path = DATA_ROOT / "interim" / "s05_tsdf" / "metadata.json"
    inp = PlaneExtractionInput(
        surface_points_path=surface_points_path,
        metadata_path=metadata_path,
    )

    assert step.validate_inputs(inp), "s06 validate_inputs failed"

    out = step.run(inp)
    logger.info(
        f"s06 result: {out.num_planes} planes "
        f"({out.num_walls} walls, {out.num_floors} floors, {out.num_ceilings} ceilings)"
    )

    assert out.planes_file.exists()
    assert out.boundaries_file.exists()
    assert out.num_planes >= 1, f"Expected >= 1 plane, got {out.num_planes}"

    return out.planes_file, out.boundaries_file


def phase_s07(planes_file: Path, boundaries_file: Path) -> Path:
    """Phase 7: IFC export."""
    logger.info("=== Phase 7: s07 IFC Export ===")

    from gss.steps.s07_ifc_export import IfcExportStep, IfcExportInput, IfcExportConfig

    cfg = IfcExportConfig(project_name="GSS_Replica_room0")
    step = IfcExportStep(config=cfg, data_root=DATA_ROOT)
    inp = IfcExportInput(planes_file=planes_file, boundaries_file=boundaries_file)

    assert step.validate_inputs(inp), "s07 validate_inputs failed"

    out = step.run(inp)
    logger.info(f"s07 result: {out.ifc_path} ({out.num_walls} walls, {out.num_slabs} slabs)")

    assert out.ifc_path.exists(), f"IFC not found: {out.ifc_path}"
    assert (out.num_walls + out.num_slabs) > 0, "No walls or slabs generated"

    return out.ifc_path


def main():
    parser = argparse.ArgumentParser(description="Full pipeline E2E test on Replica room0")
    parser.add_argument("--skip-video", action="store_true", help="Skip video creation (use existing)")
    parser.add_argument("--iterations", type=int, default=3000, help="3DGS training iterations")
    parser.add_argument("--max-video-frames", type=int, default=200, help="Max frames for video")
    args = parser.parse_args()

    # Ensure project root is in path for imports
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    t0 = time.time()
    results = {}

    try:
        # Video creation
        if not args.skip_video or not VIDEO_PATH.exists():
            phase_video(max_frames=args.max_video_frames)
        else:
            logger.info("Skipping video creation (--skip-video)")

        # s01
        t1 = time.time()
        frames_dir = phase_s01()
        results["s01"] = {"time": time.time() - t1, "frames": len(list(frames_dir.glob("*.png")))}

        # s02
        t2 = time.time()
        sparse_dir, num_reg, num_pts = phase_s02(frames_dir)
        results["s02"] = {"time": time.time() - t2, "registered": num_reg, "points3d": num_pts}

        # s03
        t3 = time.time()
        model_path = phase_s03(frames_dir, sparse_dir, iterations=args.iterations)
        results["s03"] = {"time": time.time() - t3}

        # s04
        t4 = time.time()
        depth_dir, poses_file = phase_s04(model_path, sparse_dir)
        results["s04"] = {"time": time.time() - t4, "depths": len(list(depth_dir.glob("*.npy")))}

        # s05
        t5 = time.time()
        surface_path = phase_s05(depth_dir, poses_file)
        results["s05"] = {"time": time.time() - t5}

        # s06
        t6 = time.time()
        planes_file, boundaries_file = phase_s06(surface_path)
        results["s06"] = {"time": time.time() - t6}

        # s07
        t7 = time.time()
        ifc_path = phase_s07(planes_file, boundaries_file)
        results["s07"] = {"time": time.time() - t7}

        total_time = time.time() - t0
        logger.info("=" * 60)
        logger.info("FULL PIPELINE TEST PASSED")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Output: {ifc_path}")
        for step, data in results.items():
            logger.info(f"  {step}: {data['time']:.1f}s {data}")
        logger.info("=" * 60)

    except Exception as e:
        total_time = time.time() - t0
        logger.error(f"PIPELINE FAILED after {total_time:.1f}s: {e}", exc_info=True)
        logger.info("Completed steps:")
        for step, data in results.items():
            logger.info(f"  {step}: {data['time']:.1f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
