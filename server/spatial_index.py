"""Spatial index — cuboid partitioning for view-adaptive Gaussian streaming.

Divides the 3D scene into axis-aligned cuboids so we can efficiently
query which Gaussians are visible from a given camera pose.
Inspired by SGSS (MMSys 2025) cuboid partitioning.
"""

from __future__ import annotations

import numpy as np


class CuboidIndex:
    """Axis-aligned cuboid spatial index for Gaussian positions."""

    def __init__(self, positions: np.ndarray, cuboid_size: float = 2.0):
        """Build spatial index from Gaussian positions.

        Args:
            positions: (N, 3) float32 array of Gaussian centers.
            cuboid_size: side length of each cuboid cell.
        """
        self.positions = positions
        self.cuboid_size = cuboid_size
        self.n = positions.shape[0]

        # Compute cuboid keys (integer grid coordinates)
        self._keys = np.floor(positions / cuboid_size).astype(np.int32)

        # Build cuboid → gaussian indices mapping
        self._cuboids: dict[tuple[int, int, int], np.ndarray] = {}
        for i in range(self.n):
            key = tuple(self._keys[i].tolist())
            if key not in self._cuboids:
                self._cuboids[key] = []
            self._cuboids[key].append(i)

        # Convert lists to arrays
        for key in self._cuboids:
            self._cuboids[key] = np.array(self._cuboids[key], dtype=np.int32)

    @property
    def cuboid_count(self) -> int:
        return len(self._cuboids)

    def query_frustum(
        self,
        camera_pos: np.ndarray,     # (3,) camera position
        camera_fwd: np.ndarray,     # (3,) forward direction (normalized)
        max_distance: float = 20.0,
        fov_half_angle: float = 1.0,  # radians (~57 degrees)
    ) -> np.ndarray:
        """Return indices of Gaussians visible from the given camera.

        Uses a simple frustum test: cuboid center must be within max_distance
        and within the half-angle cone of the camera forward direction.

        Returns:
            (M,) int32 array of gaussian indices.
        """
        camera_fwd = camera_fwd / (np.linalg.norm(camera_fwd) + 1e-8)
        cos_threshold = np.cos(fov_half_angle)

        visible_indices = []
        for key, indices in self._cuboids.items():
            # Cuboid center
            center = (np.array(key, dtype=np.float32) + 0.5) * self.cuboid_size

            # Direction from camera to cuboid center
            to_cuboid = center - camera_pos
            dist = np.linalg.norm(to_cuboid)

            if dist > max_distance:
                continue

            if dist < 1e-6:
                # Camera inside cuboid
                visible_indices.append(indices)
                continue

            cos_angle = np.dot(to_cuboid / dist, camera_fwd)
            if cos_angle >= cos_threshold:
                visible_indices.append(indices)

        if visible_indices:
            return np.concatenate(visible_indices)
        return np.array([], dtype=np.int32)

    def query_radius(
        self,
        center: np.ndarray,  # (3,)
        radius: float,
    ) -> np.ndarray:
        """Return indices of Gaussians within radius of center point."""
        # Find which cuboids overlap the sphere
        r_cuboids = int(np.ceil(radius / self.cuboid_size))
        center_key = np.floor(center / self.cuboid_size).astype(np.int32)

        visible_indices = []
        for dx in range(-r_cuboids, r_cuboids + 1):
            for dy in range(-r_cuboids, r_cuboids + 1):
                for dz in range(-r_cuboids, r_cuboids + 1):
                    key = (
                        center_key[0] + dx,
                        center_key[1] + dy,
                        center_key[2] + dz,
                    )
                    if key in self._cuboids:
                        visible_indices.append(self._cuboids[key])

        if visible_indices:
            all_idx = np.concatenate(visible_indices)
            # Fine filter: actual distance check
            dists = np.linalg.norm(self.positions[all_idx] - center, axis=1)
            return all_idx[dists <= radius]
        return np.array([], dtype=np.int32)
