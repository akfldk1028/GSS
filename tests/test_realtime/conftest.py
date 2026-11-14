"""Fixtures for real-time digital twin integration tests.

These tests spin up actual gRPC servers and test the full rendering pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def fake_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal torch checkpoint that _fallback_render can load."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    n_gaussians = 100
    ckpt = {
        "means": torch.randn(n_gaussians, 3),
        "colors": torch.randn(n_gaussians, 3),
        "log_scales": torch.full((n_gaussians, 3), -3.0),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n_gaussians),
        "raw_opacities": torch.zeros(n_gaussians),
        "method": "2dgs",
    }

    ckpt_path = tmp_path / "data" / "test_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    return ckpt_path


@pytest.fixture()
def identity_c2w() -> list[float]:
    """Identity 4x4 camera-to-world matrix (16 floats)."""
    return np.eye(4, dtype=np.float32).flatten().tolist()


@pytest.fixture()
def orbit_cameras() -> list[list[float]]:
    """Generate 8 camera poses orbiting around the origin."""
    cameras = []
    for i in range(8):
        angle = (2 * np.pi * i) / 8
        radius = 3.0
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)

        # Look-at origin
        forward = np.array([-x, 0, -z], dtype=np.float32)
        forward /= np.linalg.norm(forward)

        right = np.cross(np.array([0, 1, 0], dtype=np.float32), forward)
        right /= np.linalg.norm(right)

        up = np.cross(forward, right)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = [x, 0.5, z]

        cameras.append(c2w.flatten().tolist())
    return cameras
