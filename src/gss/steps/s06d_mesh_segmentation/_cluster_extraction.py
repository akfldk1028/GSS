"""Module B: Extract connected components from residual (non-planar) faces.

Builds a face adjacency graph via shared edges, then finds connected
components using union-find. Each component becomes one mesh element.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# Distinct colors for cluster visualization (HSV-derived, good contrast)
_CLUSTER_COLORS = [
    [0.90, 0.30, 0.30],  # red
    [0.30, 0.70, 0.90],  # sky blue
    [0.40, 0.85, 0.35],  # green
    [0.95, 0.70, 0.20],  # orange
    [0.60, 0.35, 0.85],  # purple
    [0.85, 0.85, 0.25],  # yellow
    [0.30, 0.85, 0.75],  # teal
    [0.90, 0.45, 0.70],  # pink
]


def _union_find_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find connected components.

    Args:
        n: Number of nodes.
        edges: List of (i, j) edges.

    Returns:
        List of components, each a list of node indices.
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for a, b in edges:
        union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    return list(groups.values())


def extract_clusters(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    min_faces: int = 50,
    min_area: float = 0.01,
    ifc_class: str = "IfcBuildingElementProxy",
    color_by_cluster: bool = True,
) -> tuple[list[dict], int]:
    """Extract connected components from residual faces.

    Args:
        vertices: (V, 3) all mesh vertices.
        faces: (F, 3) all mesh triangles.
        face_labels: (F,) labels from face classification (-1 = residual).
        min_faces: Minimum faces to keep a cluster.
        min_area: Minimum total area (in scene units²) to keep a cluster.
        ifc_class: IFC class for output elements.
        color_by_cluster: Assign distinct colors to each cluster.

    Returns:
        (elements, num_discarded): mesh_elements list + count of discarded faces.
    """
    # Extract residual face indices
    residual_mask = face_labels == -1
    residual_indices = np.where(residual_mask)[0]

    if len(residual_indices) == 0:
        logger.info("No residual faces to cluster")
        return [], 0

    residual_faces = faces[residual_indices]  # (R, 3)

    # Build edge → face adjacency for residual faces only
    # Map: (min_vertex, max_vertex) → list of local face indices
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for local_idx, tri in enumerate(residual_faces):
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            edge = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            edge_to_faces[edge].append(local_idx)

    # Build adjacency edges between local face indices
    edges: list[tuple[int, int]] = []
    for face_list in edge_to_faces.values():
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                edges.append((face_list[i], face_list[j]))

    # Find connected components
    components = _union_find_components(len(residual_faces), edges)
    logger.info(f"Found {len(components)} connected components from {len(residual_faces)} residual faces")

    # Filter and build mesh elements
    elements: list[dict] = []
    num_discarded = 0

    for comp_idx, comp_face_locals in enumerate(components):
        if len(comp_face_locals) < min_faces:
            num_discarded += len(comp_face_locals)
            continue

        # Get the original face indices for this component
        comp_faces = residual_faces[comp_face_locals]  # (C, 3)

        # Find unique vertices used by this component
        unique_verts = np.unique(comp_faces.ravel())
        # Build re-indexing map: old_vertex_idx → new_vertex_idx
        vert_map = {int(old): new for new, old in enumerate(unique_verts)}

        comp_vertices = vertices[unique_verts]  # (V', 3)
        comp_faces_reindexed = np.array(
            [[vert_map[int(v)] for v in tri] for tri in comp_faces],
            dtype=np.intp,
        )

        # Compute total area
        v0 = comp_vertices[comp_faces_reindexed[:, 0]]
        v1 = comp_vertices[comp_faces_reindexed[:, 1]]
        v2 = comp_vertices[comp_faces_reindexed[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area = float(np.sum(np.linalg.norm(cross, axis=1)) * 0.5)

        if area < min_area:
            num_discarded += len(comp_face_locals)
            continue

        color = _CLUSTER_COLORS[len(elements) % len(_CLUSTER_COLORS)] if color_by_cluster else [0.6, 0.6, 0.6]

        elements.append({
            "name": f"Residual_{len(elements)}",
            "ifc_class": ifc_class,
            "vertices": comp_vertices.tolist(),
            "faces": comp_faces_reindexed.tolist(),
            "color": color,
        })

    logger.info(
        f"Extracted {len(elements)} mesh elements, "
        f"discarded {num_discarded} faces from small clusters"
    )
    return elements, num_discarded
