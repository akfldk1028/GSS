"""Extract triangulated meshes from IFC file.

Uses ifcopenshell.geom to iterate over IFC elements and extract
vertex/face data with classification and color assignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MeshData:
    """Extracted mesh from a single IFC element."""

    name: str
    ifc_class: str
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int32
    color: list[float] = field(default_factory=lambda: [0.8, 0.8, 0.8, 1.0])


def _classify_ifc_class(ifc_class: str, name: str) -> str:
    """Map IFC class + name to a color category key."""
    if ifc_class == "IfcWall":
        return "wall"
    if ifc_class == "IfcSlab":
        return "slab"
    if ifc_class == "IfcDoor":
        return "door"
    if ifc_class == "IfcWindow":
        return "window"
    if ifc_class == "IfcSpace":
        return "space"
    if ifc_class == "IfcOpeningElement":
        return "opening"
    return "default"


def extract_meshes_from_ifc(
    ifc_path: Path,
    *,
    color_map: dict[str, list[float]] | None = None,
    include_spaces: bool = False,
) -> list[MeshData]:
    """Extract all triangulated meshes from an IFC file.

    Args:
        ifc_path: Path to .ifc file.
        color_map: Dict mapping category keys (wall/slab/door/window/space/default)
                   to RGBA color lists.
        include_spaces: Whether to include IfcSpace elements.

    Returns:
        List of MeshData with vertices, faces, and colors.
    """
    import ifcopenshell
    import ifcopenshell.geom

    if color_map is None:
        color_map = {
            "wall": [0.85, 0.85, 0.85, 1.0],
            "slab": [0.7, 0.7, 0.7, 1.0],
            "door": [0.55, 0.35, 0.2, 1.0],
            "window": [0.6, 0.8, 1.0, 0.7],
            "opening": [0.9, 0.9, 0.9, 0.5],
            "space": [0.9, 0.95, 1.0, 0.3],
            "default": [0.8, 0.8, 0.8, 1.0],
        }

    ifc = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    ifc_classes = ["IfcWall", "IfcSlab", "IfcDoor", "IfcWindow"]
    if include_spaces:
        ifc_classes.append("IfcSpace")

    meshes: list[MeshData] = []

    for ifc_class in ifc_classes:
        for element in ifc.by_type(ifc_class):
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
            except (RuntimeError, ValueError):
                logger.debug(f"Failed to create shape for {element.Name or ifc_class}")
                continue

            v = shape.geometry.verts
            f = shape.geometry.faces
            if len(v) < 9 or len(f) < 3:
                continue

            vertices = np.array(v, dtype=np.float64).reshape(-1, 3)
            faces = np.array(f, dtype=np.int32).reshape(-1, 3)

            name = element.Name or f"{ifc_class}_{element.id()}"
            category = _classify_ifc_class(ifc_class, name)
            color = color_map.get(category, color_map.get("default", [0.8, 0.8, 0.8, 1.0]))

            meshes.append(MeshData(
                name=name,
                ifc_class=ifc_class,
                vertices=vertices,
                faces=faces,
                color=list(color),
            ))

    logger.info(
        f"Extracted {len(meshes)} meshes from {ifc_path.name} "
        f"({sum(len(m.vertices) for m in meshes)} verts, "
        f"{sum(len(m.faces) for m in meshes)} faces)"
    )
    return meshes
