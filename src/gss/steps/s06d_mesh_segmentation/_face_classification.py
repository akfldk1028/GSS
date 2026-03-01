"""Module A: Classify mesh faces as planar (matching RANSAC planes) or residual.

For each triangle face, computes centroid distance and normal angle to every
RANSAC plane. Faces sufficiently close to a plane are labeled with that
plane's ID; the rest are labeled as residual (-1).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def classify_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    planes: list[dict],
    distance_thresh: float = 0.03,
    normal_thresh: float = 0.85,
) -> np.ndarray:
    """Classify each face as belonging to a RANSAC plane or residual.

    All inputs are expected in the same coordinate space (typically Manhattan).

    Args:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) triangle indices.
        planes: List of plane dicts with 'normal' (3,) and 'd' (float).
        distance_thresh: Max |n·c + d| to consider face on-plane.
        normal_thresh: Min |cos(angle)| between face normal and plane normal.

    Returns:
        (F,) int array: -1 = residual, >=0 = matched plane index.
    """
    num_faces = len(faces)
    if num_faces == 0:
        return np.array([], dtype=np.intp)

    # Compute face centroids and normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0  # (F, 3)

    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = np.cross(e1, e2)  # (F, 3)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    # Avoid division by zero for degenerate faces
    norms = np.maximum(norms, 1e-12)
    face_normals = face_normals / norms

    # For each face, find the best matching plane
    labels = np.full(num_faces, -1, dtype=np.intp)
    best_dist = np.full(num_faces, np.inf)

    for i, plane in enumerate(planes):
        pn = np.asarray(plane["normal"], dtype=np.float64)
        pn_norm = np.linalg.norm(pn)
        if pn_norm < 1e-12:
            continue
        pn = pn / pn_norm
        pd = float(plane["d"])

        # Distance from centroid to plane: |n · c + d|
        dist = np.abs(centroids @ pn + pd)  # (F,)

        # Normal alignment: |face_normal · plane_normal|
        cos_angle = np.abs(face_normals @ pn)  # (F,)

        # Condition: close enough AND normals aligned
        match = (dist < distance_thresh) & (cos_angle > normal_thresh)

        # Update only if this plane is a better match (closer)
        better = match & (dist < best_dist)
        labels[better] = i
        best_dist[better] = dist[better]

    n_planar = int(np.sum(labels >= 0))
    n_residual = num_faces - n_planar
    logger.info(
        f"Face classification: {n_planar} planar, {n_residual} residual "
        f"(of {num_faces} total)"
    )
    return labels
