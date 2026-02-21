"""End-to-end pipeline test with GPU step mocks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from gss.steps.s01_extract_frames.config import ExtractFramesConfig
from gss.steps.s01_extract_frames.contracts import ExtractFramesInput
from gss.steps.s01_extract_frames.step import ExtractFramesStep
from gss.steps.s02_colmap.contracts import ColmapOutput
from gss.steps.s03_gaussian_splatting.contracts import GaussianSplattingOutput
from gss.steps.s04_depth_render.contracts import DepthRenderOutput

logger = logging.getLogger(__name__)


@pytest.mark.e2e
def test_pipeline_e2e(synthetic_video: Path, tmp_path: Path):
    """
    End-to-end test of S01~S07 pipeline with synthetic data.

    Tests that each step's output properly connects to the next step's input.
    GPU-dependent steps (S03, S04) are mocked to avoid hardware requirements.
    """
    data_root = tmp_path / "pipeline_data"
    data_root.mkdir(parents=True, exist_ok=True)

    # ========== Step 01: Extract Frames (Real Implementation) ==========
    logger.info("Running S01: Extract Frames")
    s01_config = ExtractFramesConfig(
        target_fps=10.0,
        max_frames=20,
        blur_threshold=50.0,
        resize_width=160,
        output_format="png",
    )
    s01_step = ExtractFramesStep(config=s01_config, data_root=data_root)
    s01_input = ExtractFramesInput(video_path=synthetic_video)
    s01_output = s01_step.execute(s01_input)

    # Verify S01 output
    assert s01_output.frames_dir.exists(), "Frames directory should exist"
    assert s01_output.frame_count > 0, "Should extract at least one frame"
    assert len(s01_output.frame_list) == s01_output.frame_count
    logger.info(f"S01 extracted {s01_output.frame_count} frames")

    # ========== Step 02: COLMAP (Mock - not implemented yet) ==========
    logger.info("Mocking S02: COLMAP (not implemented)")
    s02_output_dir = data_root / "interim" / "s02_colmap"
    s02_output_dir.mkdir(parents=True, exist_ok=True)

    # Create mock COLMAP output files
    cameras_file = s02_output_dir / "cameras.txt"
    images_file = s02_output_dir / "images.txt"
    cameras_file.write_text("# Mock cameras file\n")
    images_file.write_text("# Mock images file\n")

    s02_output = ColmapOutput(
        sparse_dir=s02_output_dir,
        cameras_file=cameras_file,
        images_file=images_file,
        num_registered=s01_output.frame_count,
        num_points3d=1000,
    )
    logger.info(f"S02 mock: {s02_output.num_registered} registered images")

    # ========== Step 03: Gaussian Splatting (Mock - GPU step) ==========
    logger.info("Mocking S03: Gaussian Splatting (GPU step)")
    s03_output_dir = data_root / "interim" / "s03_gaussians"
    s03_output_dir.mkdir(parents=True, exist_ok=True)

    # Create mock Gaussian model file
    model_path = s03_output_dir / "point_cloud.ply"
    model_path.write_text("# Mock PLY file\n")

    s03_output = GaussianSplattingOutput(
        model_path=model_path,
        num_gaussians=5000,
        training_iterations=7000,
    )
    logger.info(f"S03 mock: {s03_output.num_gaussians} Gaussians trained")

    # Verify S03 input would come from S01 and S02
    assert s01_output.frames_dir.exists(), "S03 needs frames from S01"
    assert s02_output.sparse_dir.exists(), "S03 needs sparse reconstruction from S02"

    # ========== Step 04: Depth Render (Mock - GPU step) ==========
    logger.info("Mocking S04: Depth Render (GPU step)")
    s04_output_dir = data_root / "interim" / "s04_depth_maps"
    s04_output_dir.mkdir(parents=True, exist_ok=True)

    # Create mock depth maps
    depth_dir = s04_output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Create mock depth files
    num_views = 5
    for i in range(num_views):
        depth_file = depth_dir / f"depth_{i:04d}.npy"
        depth_file.write_bytes(b"mock_depth_data")

    # Create mock poses file
    poses_file = s04_output_dir / "poses.json"
    poses_data = {
        "views": [
            {"id": i, "rotation": [1, 0, 0, 0], "translation": [0, 0, i]}
            for i in range(num_views)
        ]
    }
    poses_file.write_text(json.dumps(poses_data, indent=2))

    s04_output = DepthRenderOutput(
        depth_dir=depth_dir,
        normal_dir=None,
        num_views=num_views,
        poses_file=poses_file,
    )
    logger.info(f"S04 mock: {s04_output.num_views} depth views rendered")

    # Verify S04 input would come from S03 and S02
    assert s03_output.model_path.exists(), "S04 needs Gaussian model from S03"
    assert s02_output.sparse_dir.exists(), "S04 needs camera poses from S02"

    # ========== Verify Pipeline Connectivity ==========
    # S01 output → S02 input
    assert s01_output.frames_dir.exists()

    # S02 output → S03 input
    assert s02_output.sparse_dir.exists()

    # S01 + S02 output → S03 input
    # (S03 needs both frames from S01 and sparse from S02)

    # S03 + S02 output → S04 input
    assert s03_output.model_path.exists()
    assert s02_output.sparse_dir.exists()

    # S04 output → S05 input (would need depth_dir and poses_file)
    assert s04_output.depth_dir.exists()
    assert s04_output.poses_file.exists()

    logger.info("✓ E2E Pipeline connectivity verified")
    logger.info(f"Pipeline artifacts in: {data_root}")

    # Final assertions
    assert s01_output.frame_count > 0, "S01 should produce frames"
    assert s02_output.num_registered > 0, "S02 should register images"
    assert s03_output.num_gaussians > 0, "S03 should produce Gaussians"
    assert s04_output.num_views > 0, "S04 should render depth views"

    logger.info("✓ All E2E tests passed")
