"""GLB writer — export meshes to GLB (glTF Binary) format.

Uses trimesh to build a Scene and export as GLB, compatible with
all major 3D viewers and game engines (UE5, Unity, Three.js, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._ifc_to_mesh import MeshData

logger = logging.getLogger(__name__)


def _has_trimesh() -> bool:
    """Check if trimesh is available."""
    try:
        import trimesh  # noqa: F401
        return True
    except ImportError:
        return False


def write_glb(meshes: list[MeshData], output_path: Path) -> Path:
    """Write meshes to a GLB file.

    Args:
        meshes: List of MeshData extracted from IFC.
        output_path: Output .glb file path.

    Returns:
        Path to the written GLB file.
    """
    import trimesh

    scene = trimesh.Scene()

    for mesh_data in meshes:
        # Convert RGBA color to per-face colors
        rgba = np.array(mesh_data.color[:4], dtype=np.float64)
        if len(rgba) < 4:
            rgba = np.append(rgba, 1.0)
        rgba_255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

        face_colors = np.tile(rgba_255, (len(mesh_data.faces), 1))

        # IFC is Z-up, glTF is Y-up: (x, y, z) → (x, z, -y)
        verts = mesh_data.vertices.copy()
        verts_converted = np.column_stack([verts[:, 0], verts[:, 2], -verts[:, 1]])

        mesh = trimesh.Trimesh(
            vertices=verts_converted,
            faces=mesh_data.faces,
            face_colors=face_colors,
            process=False,
        )

        # Use unique node name to avoid collisions
        node_name = mesh_data.name.replace(" ", "_").replace("/", "_")
        scene.add_geometry(mesh, node_name=node_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path), file_type="glb")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"GLB exported: {output_path} ({size_mb:.2f} MB, {len(meshes)} meshes)")
    return output_path
