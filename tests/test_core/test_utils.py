"""Tests for shared utility modules: geometry, io."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gss.utils.geometry import qvec2rotmat, rotmat2qvec, make_w2c, make_c2w


class TestQuaternionConversion:
    def test_identity(self):
        """Identity quaternion (1,0,0,0) -> identity rotation."""
        R = qvec2rotmat([1, 0, 0, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90deg_z(self):
        """90-degree rotation around Z axis."""
        import math
        angle = math.pi / 2
        w = math.cos(angle / 2)
        z = math.sin(angle / 2)
        R = qvec2rotmat([w, 0, 0, z])
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_roundtrip(self):
        """qvec -> rotmat -> qvec roundtrip."""
        qvec = np.array([0.5, 0.5, 0.5, 0.5])  # valid unit quaternion
        R = qvec2rotmat(qvec)
        qvec2 = rotmat2qvec(R)
        # Quaternions can differ by sign
        if np.dot(qvec, qvec2) < 0:
            qvec2 = -qvec2
        np.testing.assert_allclose(qvec2, qvec, atol=1e-10)

    def test_orthogonal(self):
        """Result should be orthogonal: R @ R.T = I."""
        s = np.sqrt(0.5)
        qvec = [s, 0.0, s, 0.0]  # exact unit quaternion, 90 deg around Y
        R = qvec2rotmat(qvec)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestMakeW2C:
    def test_identity_pose(self):
        w2c = make_w2c([1, 0, 0, 0], [0, 0, 0])
        np.testing.assert_allclose(w2c, np.eye(4), atol=1e-10)

    def test_translation(self):
        w2c = make_w2c([1, 0, 0, 0], [1.0, 2.0, 3.0])
        assert w2c[0, 3] == pytest.approx(1.0)
        assert w2c[1, 3] == pytest.approx(2.0)
        assert w2c[2, 3] == pytest.approx(3.0)

    def test_inverse_roundtrip(self):
        qvec = [0.5, 0.5, 0.5, 0.5]
        tvec = [1.0, 2.0, 3.0]
        w2c = make_w2c(qvec, tvec)
        c2w = make_c2w(qvec, tvec)
        np.testing.assert_allclose(w2c @ c2w, np.eye(4), atol=1e-10)


class TestPlyIO:
    def test_write_and_read(self, tmp_path: Path):
        from gss.utils.io import write_ply, read_ply_points

        means = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        opacities = np.array([0.9, 0.5], dtype=np.float32)

        ply_path = tmp_path / "test.ply"
        write_ply(ply_path, means, colors, opacities)
        assert ply_path.exists()

        # Read back
        pts = read_ply_points(ply_path)
        assert pts.shape == (2, 3)
        np.testing.assert_allclose(pts, means, atol=1e-5)
