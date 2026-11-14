"""Gaussian splat binary encoder for WebSocket streaming.

Encodes Gaussian primitives into compact binary frames for real-time
transmission to WebGL/WebGPU clients.

Frame format:
  Header (12 bytes):
    msg_type   : u8    (0x01=full, 0x02=delta_add, 0x03=delta_remove)
    flags      : u8    (reserved)
    seq        : u32   (sequence number)
    count      : u32   (number of gaussians in this frame)
    padding    : u16   (alignment)

  Per-gaussian (48 bytes):
    position   : f32 x 3  (12 bytes)
    rotation   : f16 x 4  (8 bytes)  — quaternion wxyz
    scale      : f16 x 3  (6 bytes)  — log scale
    opacity    : f16       (2 bytes)
    sh_dc      : f16 x 3  (6 bytes)  — DC spherical harmonic (RGB)
    sh_rest    : f16 x 7  (14 bytes) — first-order SH (truncated)
"""

from __future__ import annotations

import struct
from enum import IntEnum
from pathlib import Path

import numpy as np

HEADER_SIZE = 12
GAUSSIAN_SIZE = 48


class MsgType(IntEnum):
    FULL = 0x01
    DELTA_ADD = 0x02
    DELTA_REMOVE = 0x03


def encode_header(msg_type: MsgType, seq: int, count: int) -> bytes:
    """Encode a frame header (12 bytes)."""
    return struct.pack("<BBIIxx", msg_type, 0, seq, count)


def encode_gaussians(
    positions: np.ndarray,     # (N, 3) float32
    rotations: np.ndarray,     # (N, 4) float32 → f16
    scales: np.ndarray,        # (N, 3) float32 → f16
    opacities: np.ndarray,     # (N,)   float32 → f16
    sh_dc: np.ndarray,         # (N, 3) float32 → f16
    sh_rest: np.ndarray | None = None,  # (N, 7) or None → f16
) -> bytes:
    """Encode N Gaussians into binary payload.

    Returns bytes of length N * GAUSSIAN_SIZE.
    """
    n = positions.shape[0]

    # Ensure correct dtypes
    pos = positions.astype(np.float32)              # 12 B
    rot = rotations.astype(np.float16)              # 8 B
    scl = scales.astype(np.float16)                 # 6 B
    opa = opacities.astype(np.float16).reshape(n, 1)  # 2 B
    dc = sh_dc.astype(np.float16)                   # 6 B

    if sh_rest is not None:
        rest = sh_rest[:, :7].astype(np.float16)    # 14 B
    else:
        rest = np.zeros((n, 7), dtype=np.float16)   # 14 B

    # Pack per-gaussian: 12 + 8 + 6 + 2 + 6 + 14 = 48 bytes
    buf = bytearray(n * GAUSSIAN_SIZE)
    for i in range(n):
        offset = i * GAUSSIAN_SIZE
        struct.pack_into("<3f", buf, offset, *pos[i])
        offset += 12
        buf[offset:offset + 8] = rot[i].tobytes()
        offset += 8
        buf[offset:offset + 6] = scl[i].tobytes()
        offset += 6
        buf[offset:offset + 2] = opa[i].tobytes()
        offset += 2
        buf[offset:offset + 6] = dc[i].tobytes()
        offset += 6
        buf[offset:offset + 14] = rest[i].tobytes()

    return bytes(buf)


def encode_frame(
    msg_type: MsgType,
    seq: int,
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    sh_dc: np.ndarray,
    sh_rest: np.ndarray | None = None,
) -> bytes:
    """Encode a complete frame (header + gaussians)."""
    n = positions.shape[0]
    header = encode_header(msg_type, seq, n)
    body = encode_gaussians(positions, rotations, scales, opacities, sh_dc, sh_rest)
    return header + body


def load_ply_gaussians(ply_path: Path) -> dict[str, np.ndarray]:
    """Load Gaussian attributes from a PLY file (3DGS/gsplat format).

    Returns dict with keys: positions, rotations, scales, opacities, sh_dc, sh_rest
    """
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]

    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)

    # Rotations (quaternion)
    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if all(name in vertex for name in rot_names):
        rotations = np.stack([vertex[r] for r in rot_names], axis=-1).astype(np.float32)
    else:
        n = positions.shape[0]
        rotations = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)

    # Scales
    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(name in vertex for name in scale_names):
        scales = np.stack([vertex[s] for s in scale_names], axis=-1).astype(np.float32)
    else:
        scales = np.zeros((positions.shape[0], 3), dtype=np.float32)

    # Opacity
    if "opacity" in vertex:
        opacities = np.array(vertex["opacity"], dtype=np.float32)
    else:
        opacities = np.ones(positions.shape[0], dtype=np.float32)

    # SH DC
    sh_dc_names = ["f_dc_0", "f_dc_1", "f_dc_2"]
    if all(name in vertex for name in sh_dc_names):
        sh_dc = np.stack([vertex[s] for s in sh_dc_names], axis=-1).astype(np.float32)
    else:
        sh_dc = np.zeros((positions.shape[0], 3), dtype=np.float32)

    # SH rest (first 7 coefficients)
    sh_rest_list = []
    for i in range(7):
        name = f"f_rest_{i}"
        if name in vertex:
            sh_rest_list.append(np.array(vertex[name], dtype=np.float32))
    if sh_rest_list:
        sh_rest = np.stack(sh_rest_list, axis=-1)
    else:
        sh_rest = None

    return {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "opacities": opacities,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
    }
