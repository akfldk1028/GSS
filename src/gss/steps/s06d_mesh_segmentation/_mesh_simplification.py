"""Module C: Optional mesh simplification via quadric decimation.

Reduces face count per cluster to keep IFC file sizes manageable.
Uses Open3D's simplify_quadric_decimation.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def simplify_cluster(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_ratio: float = 0.5,
    max_faces: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplify a mesh cluster using quadric decimation.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle indices.
        target_ratio: Fraction of faces to keep.
        max_faces: Hard cap on output faces.

    Returns:
        (simplified_vertices, simplified_faces).
    """
    import open3d as o3d

    target_faces = min(int(len(faces) * target_ratio), max_faces)
    if len(faces) <= target_faces:
        return vertices, faces

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

    out_verts = np.asarray(simplified.vertices)
    out_faces = np.asarray(simplified.triangles)

    logger.debug(
        f"Simplified: {len(faces)} → {len(out_faces)} faces "
        f"({len(vertices)} → {len(out_verts)} vertices)"
    )
    return out_verts, out_faces


def simplify_elements(
    elements: list[dict],
    target_ratio: float = 0.5,
    max_faces: int = 50000,
) -> list[dict]:
    """Simplify all mesh elements that exceed max_faces or benefit from decimation.

    Args:
        elements: mesh_elements list (each with 'vertices' and 'faces').
        target_ratio: Fraction of faces to keep per element.
        max_faces: Hard cap on output faces per element.

    Returns:
        Updated elements list (in-place modification).
    """
    total_before = 0
    total_after = 0

    for elem in elements:
        verts = np.asarray(elem["vertices"], dtype=np.float64)
        faces_arr = np.asarray(elem["faces"], dtype=np.intp)
        total_before += len(faces_arr)

        target = min(int(len(faces_arr) * target_ratio), max_faces)
        if len(faces_arr) > target:
            verts, faces_arr = simplify_cluster(verts, faces_arr, target_ratio, max_faces)
            elem["vertices"] = verts.tolist()
            elem["faces"] = faces_arr.tolist()

        total_after += len(faces_arr)

    if total_before > total_after:
        logger.info(f"Simplification: {total_before} → {total_after} faces total")

    return elements
