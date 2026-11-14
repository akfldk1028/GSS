"""Tests for splat encoder and spatial index (Phase 2)."""

import numpy as np
import pytest

from server.splat_encoder import (
    GAUSSIAN_SIZE,
    HEADER_SIZE,
    MsgType,
    encode_frame,
    encode_gaussians,
    encode_header,
)
from server.spatial_index import CuboidIndex


class TestSplatEncoder:
    def _random_gaussians(self, n: int):
        return {
            "positions": np.random.randn(n, 3).astype(np.float32),
            "rotations": np.random.randn(n, 4).astype(np.float32),
            "scales": np.random.randn(n, 3).astype(np.float32),
            "opacities": np.random.rand(n).astype(np.float32),
            "sh_dc": np.random.randn(n, 3).astype(np.float32),
        }

    def test_header_size(self):
        h = encode_header(MsgType.FULL, 0, 100)
        assert len(h) == HEADER_SIZE

    def test_gaussian_size(self):
        g = self._random_gaussians(1)
        data = encode_gaussians(**g)
        assert len(data) == GAUSSIAN_SIZE

    def test_frame_size(self):
        n = 50
        g = self._random_gaussians(n)
        frame = encode_frame(MsgType.FULL, 0, **g)
        assert len(frame) == HEADER_SIZE + n * GAUSSIAN_SIZE

    def test_frame_with_sh_rest(self):
        n = 10
        g = self._random_gaussians(n)
        g["sh_rest"] = np.random.randn(n, 7).astype(np.float32)
        frame = encode_frame(MsgType.DELTA_ADD, 1, **g)
        assert len(frame) == HEADER_SIZE + n * GAUSSIAN_SIZE

    def test_zero_gaussians(self):
        g = self._random_gaussians(0)
        frame = encode_frame(MsgType.FULL, 0, **g)
        assert len(frame) == HEADER_SIZE


class TestCuboidIndex:
    def test_construction(self):
        pos = np.random.randn(100, 3).astype(np.float32)
        idx = CuboidIndex(pos, cuboid_size=1.0)
        assert idx.cuboid_count > 0
        assert idx.n == 100

    def test_frustum_query(self):
        pos = np.array([[5, 0, 0], [-5, 0, 0], [0, 5, 0]], dtype=np.float32)
        idx = CuboidIndex(pos, cuboid_size=1.0)

        # Looking in +x direction should see (5,0,0) but not (-5,0,0)
        cam = np.array([0, 0, 0], dtype=np.float32)
        fwd = np.array([1, 0, 0], dtype=np.float32)
        visible = idx.query_frustum(cam, fwd, max_distance=10.0, fov_half_angle=0.5)

        visible_set = set(visible.tolist())
        assert 0 in visible_set  # (5,0,0) is in front
        assert 1 not in visible_set  # (-5,0,0) is behind

    def test_radius_query(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=np.float32)
        idx = CuboidIndex(pos, cuboid_size=1.0)

        center = np.array([0, 0, 0], dtype=np.float32)
        nearby = idx.query_radius(center, 2.0)
        assert len(nearby) == 2  # (0,0,0) and (1,0,0)

    def test_empty_result(self):
        pos = np.array([[100, 100, 100]], dtype=np.float32)
        idx = CuboidIndex(pos, cuboid_size=1.0)

        cam = np.array([0, 0, 0], dtype=np.float32)
        fwd = np.array([1, 0, 0], dtype=np.float32)
        visible = idx.query_frustum(cam, fwd, max_distance=5.0)
        assert len(visible) == 0
