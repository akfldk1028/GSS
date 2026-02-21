"""Fixtures for E2E pipeline tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


def create_synthetic_video(
    output_dir: Path,
    num_frames: int = 30,
    resolution: tuple[int, int] = (160, 120),
    fps: float = 30.0,
) -> Path:
    """
    Create a synthetic test video with random noise patterns.

    Args:
        output_dir: Directory to save the video
        num_frames: Number of frames to generate
        resolution: Video resolution as (width, height)
        fps: Frames per second

    Returns:
        Path to the created video file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "synthetic_test_video.mp4"

    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a frame with gradient + random noise for visual variation
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add a horizontal gradient
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        frame[:, :, 0] = gradient  # Blue channel

        # Add a vertical gradient
        gradient_v = np.linspace(0, 255, height, dtype=np.uint8)
        frame[:, :, 1] = gradient_v[:, np.newaxis]  # Green channel

        # Add some random noise to make it interesting
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        # Add frame number as text for debugging
        cv2.putText(
            frame,
            f"F:{i:03d}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """
    Pytest fixture that provides a synthetic test video.

    Creates a 30-frame, 160x120 video with synthetic content for E2E testing.

    Args:
        tmp_path: Pytest's temporary directory fixture

    Returns:
        Path to the synthetic video file
    """
    video_path = create_synthetic_video(
        output_dir=tmp_path,
        num_frames=30,
        resolution=(160, 120),
        fps=30.0,
    )
    return video_path
