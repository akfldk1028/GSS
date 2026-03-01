"""Step 06d: Mesh segmentation — extract non-planar geometry from TSDF mesh.

Takes the TSDF triangle mesh and RANSAC planes, classifies mesh faces as
planar (matching a RANSAC plane) or residual (non-planar), then clusters
the residual faces into connected components for IFC tessellation.

Pipeline position: s06 → s06d → (parallel with s06b) → s07
  s07 consumes both walls.json (from s06b) and mesh_elements.json (from s06d).

Graceful skip when surface_mesh_path is None (e.g., import pipeline).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

from gss.core.step_base import BaseStep
from .config import MeshSegmentationConfig
from .contracts import MeshSegmentationInput, MeshSegmentationOutput

logger = logging.getLogger(__name__)


def _load_manhattan_rotation(s06_dir: Path) -> np.ndarray | None:
    """Load Manhattan rotation matrix from s06 output."""
    path = s06_dir / "manhattan_alignment.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        R = np.asarray(data["manhattan_rotation"], dtype=float)
        if R.shape != (3, 3):
            return None
        return R
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to load manhattan_alignment.json: {e}")
        return None


class MeshSegmentationStep(
    BaseStep[MeshSegmentationInput, MeshSegmentationOutput, MeshSegmentationConfig]
):
    name: ClassVar[str] = "mesh_segmentation"
    input_type: ClassVar = MeshSegmentationInput
    output_type: ClassVar = MeshSegmentationOutput
    config_type: ClassVar = MeshSegmentationConfig

    def validate_inputs(self, inputs: MeshSegmentationInput) -> bool:
        if inputs.surface_mesh_path is None:
            logger.info("No surface_mesh_path — step will skip gracefully")
            return True
        if not inputs.surface_mesh_path.exists():
            logger.warning(f"surface_mesh_path not found: {inputs.surface_mesh_path}")
            return True  # graceful skip, not failure
        if not inputs.planes_file.exists():
            logger.error(f"planes_file not found: {inputs.planes_file}")
            return False
        return True

    def run(self, inputs: MeshSegmentationInput) -> MeshSegmentationOutput:
        output_dir = self.data_root / "interim" / "s06d_mesh_segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Graceful skip ---
        if inputs.surface_mesh_path is None or not inputs.surface_mesh_path.exists():
            logger.info("No surface mesh available — skipping mesh segmentation")
            return MeshSegmentationOutput()

        # --- Load mesh ---
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("Open3D not installed — skipping mesh segmentation")
            return MeshSegmentationOutput()

        mesh = o3d.io.read_triangle_mesh(str(inputs.surface_mesh_path))
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        if len(faces) == 0:
            logger.info("Mesh has 0 faces — skipping")
            return MeshSegmentationOutput()

        logger.info(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")

        # --- Load planes ---
        with open(inputs.planes_file, encoding="utf-8") as f:
            planes = json.load(f)
        logger.info(f"Loaded {len(planes)} planes")

        # --- Manhattan transform ---
        s06_dir = self.data_root / "interim" / "s06_planes"
        R = _load_manhattan_rotation(s06_dir)

        if R is not None:
            # Transform mesh to Manhattan space
            vertices = vertices @ R.T
            logger.info("Transformed mesh to Manhattan space")

            # Transform plane normals to Manhattan space
            for p in planes:
                n = np.asarray(p["normal"], dtype=np.float64)
                p["normal"] = (R @ n).tolist()
                # d is invariant under orthogonal rotation
        else:
            logger.warning("No Manhattan rotation — processing in original coordinates")

        # --- Module A: Face classification ---
        from ._face_classification import classify_faces

        face_labels = classify_faces(
            vertices, faces, planes,
            distance_thresh=self.config.plane_distance_threshold,
            normal_thresh=self.config.plane_normal_threshold,
        )

        num_planar = int(np.sum(face_labels >= 0))

        if num_planar == len(faces):
            logger.info("All faces are planar — writing empty mesh_elements.json")
            elements_file = output_dir / "mesh_elements.json"
            with open(elements_file, "w", encoding="utf-8") as f:
                json.dump([], f)
            return MeshSegmentationOutput(
                mesh_elements_file=elements_file,
                num_faces_planar=num_planar,
            )

        # --- Module B: Cluster extraction ---
        from ._cluster_extraction import extract_clusters

        elements, num_discarded = extract_clusters(
            vertices, faces, face_labels,
            min_faces=self.config.min_cluster_faces,
            min_area=self.config.min_cluster_area,
            ifc_class=self.config.default_ifc_class,
            color_by_cluster=self.config.color_by_cluster,
        )

        # --- Module C: Simplification ---
        if self.config.enable_simplification and elements:
            from ._mesh_simplification import simplify_elements

            elements = simplify_elements(
                elements,
                target_ratio=self.config.target_face_ratio,
                max_faces=self.config.max_faces_per_element,
            )

        # --- Write output ---
        elements_file = output_dir / "mesh_elements.json"
        with open(elements_file, "w", encoding="utf-8") as f:
            json.dump(elements, f)

        num_residual = sum(len(e["faces"]) for e in elements)

        logger.info(
            f"Mesh segmentation complete: {num_planar} planar faces, "
            f"{num_residual} residual faces in {len(elements)} elements, "
            f"{num_discarded} discarded"
        )

        return MeshSegmentationOutput(
            mesh_elements_file=elements_file,
            num_elements=len(elements),
            num_faces_planar=num_planar,
            num_faces_residual=num_residual,
            num_faces_discarded=num_discarded,
        )
