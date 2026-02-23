"""I/O utilities: COLMAP binary parsers, PLY reader/writer."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


# ── COLMAP binary readers ────────────────────────────────────────────

def read_colmap_cameras(sparse_dir: Path) -> dict:
    """Read camera intrinsics from COLMAP sparse reconstruction."""
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(str(sparse_dir))
        return {
            cam_id: {
                "model": cam.model.name,
                "width": cam.width,
                "height": cam.height,
                "params": cam.params.tolist(),
            }
            for cam_id, cam in recon.cameras.items()
        }
    except ImportError:
        pass

    cameras_bin = sparse_dir / "cameras.bin"
    if not cameras_bin.exists():
        raise FileNotFoundError(f"cameras.bin not found in {sparse_dir}")

    cameras = {}
    with open(cameras_bin, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            n_params = {0: 3, 1: 4, 2: 4, 4: 4}.get(model_id, 4)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": list(params),
            }
    return cameras


def read_colmap_images(sparse_dir: Path) -> list[dict]:
    """Read image poses from COLMAP sparse reconstruction."""
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(str(sparse_dir))
        images = []
        for img_id, img in recon.images.items():
            cfw = img.cam_from_world()
            quat_xyzw = cfw.rotation.quat  # pycolmap returns (x, y, z, w)
            tvec = cfw.translation
            # Convert to COLMAP convention (w, x, y, z)
            qvec = [float(quat_xyzw[3]), float(quat_xyzw[0]),
                    float(quat_xyzw[1]), float(quat_xyzw[2])]
            images.append({
                "image_id": img_id,
                "camera_id": img.camera_id,
                "name": img.name,
                "qvec": qvec,
                "tvec": tvec.tolist() if hasattr(tvec, "tolist") else list(tvec),
            })
        return images
    except (ImportError, AttributeError):
        pass

    images_bin = sparse_dir / "images.bin"
    if not images_bin.exists():
        raise FileNotFoundError(f"images.bin not found in {sparse_dir}")

    images = []
    with open(images_bin, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_bytes += ch
            name = name_bytes.decode("utf-8")
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)
            images.append({
                "image_id": img_id,
                "camera_id": camera_id,
                "name": name,
                "qvec": list(qvec),
                "tvec": list(tvec),
            })
    return images


def read_colmap_points3d(sparse_dir: Path) -> np.ndarray:
    """Read 3D points from COLMAP sparse reconstruction."""
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(str(sparse_dir))
        pts = np.array([p.xyz for p in recon.points3D.values()])
        return pts
    except ImportError:
        pass

    points3d_bin = sparse_dir / "points3D.bin"
    if not points3d_bin.exists():
        return np.zeros((0, 3))

    points = []
    with open(points3d_bin, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            struct.unpack("<Q", f.read(8))  # point_id
            xyz = struct.unpack("<3d", f.read(24))
            struct.unpack("<3B", f.read(3))  # rgb
            struct.unpack("<d", f.read(8))  # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            points.append(xyz)
    return np.array(points) if points else np.zeros((0, 3))


# ── PLY I/O ──────────────────────────────────────────────────────────

def write_ply(path: Path, means: np.ndarray, colors: np.ndarray, opacities: np.ndarray) -> None:
    """Write Gaussian parameters to PLY file."""
    n = len(means)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float opacity\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        for i in range(n):
            f.write(struct.pack("<3f", *means[i]))
            rgb = np.clip(colors[i] * 255, 0, 255).astype(np.uint8)
            f.write(struct.pack("<3B", *rgb))
            f.write(struct.pack("<f", opacities[i]))


def read_ply_points(path: Path) -> np.ndarray:
    """Read point positions from a PLY file (Open3D fallback to manual parsing)."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points)
    except ImportError:
        pass

    # Manual ASCII/binary PLY reader for positions only
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break
        header_str = header.decode("ascii")

    n_vertices = 0
    # Parse header to calculate vertex record size
    property_sizes = {"float": 4, "double": 8, "uchar": 1, "char": 1,
                      "short": 2, "ushort": 2, "int": 4, "uint": 4}
    vertex_record_size = 0
    in_vertex = False
    for line in header_str.split("\n"):
        line = line.strip()
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element "):
            in_vertex = False
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) >= 3:
                vertex_record_size += property_sizes.get(parts[1], 4)

    if "binary_little_endian" in header_str and vertex_record_size >= 12:
        with open(path, "rb") as f:
            f.read(len(header))
            points = []
            for _ in range(n_vertices):
                data = f.read(vertex_record_size)
                if len(data) < 12:
                    break
                xyz = struct.unpack_from("<3f", data, 0)
                points.append(xyz)
        return np.array(points) if points else np.zeros((0, 3))

    return np.zeros((0, 3))
