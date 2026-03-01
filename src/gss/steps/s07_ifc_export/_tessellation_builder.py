"""Tessellation builder — IfcPolygonalFaceSet for free-form surfaces.

Creates tessellated IFC elements from raw vertex/face mesh data.
Used for stairs, curved surfaces, decorations, and other geometry
that doesn't fit parametric wall/slab/column primitives.

IFC4 pattern (from Cloud2BIM):
  IfcCartesianPointList3D → IfcIndexedPolygonalFace → IfcPolygonalFaceSet
  RepresentationType = "Tessellation"

Coordinate mapping:
  Manhattan (x, y, z) → IFC (x/s, z/s, y/s) where s = coordinate_scale
"""

from __future__ import annotations

import logging
from typing import Any

import ifcopenshell.api
import ifcopenshell.guid

from ._ifc_builder import IfcContext

logger = logging.getLogger(__name__)


def create_tessellated_element(
    ctx: IfcContext,
    vertices: list[list[float]],
    faces: list[list[int]],
    ifc_class: str = "IfcBuildingElementProxy",
    name: str = "Tessellated_0",
    scale: float = 1.0,
    color: list[float] | None = None,
) -> Any | None:
    """Create an IFC element with IfcPolygonalFaceSet representation.

    Args:
        ctx: IFC context.
        vertices: List of [x, y, z] points in Manhattan coordinates.
        faces: List of face index lists (0-based). Each face is [i, j, k, ...].
        ifc_class: IFC entity class (IfcWall, IfcSlab, IfcBuildingElementProxy, etc.).
        name: Element name.
        scale: Coordinate scale divisor.
        color: Optional [r, g, b] color (0-1 range).

    Returns:
        IFC entity, or None if geometry is degenerate.
    """
    if len(vertices) < 3 or len(faces) < 1:
        logger.warning(f"Tessellation '{name}': too few vertices/faces, skipping")
        return None

    ifc = ctx.ifc

    # --- IfcCartesianPointList3D ---
    # Manhattan (x, y, z) → IFC (x/s, z/s, y/s)
    coord_list = []
    for v in vertices:
        coord_list.append([
            float(v[0] / scale),
            float(v[2] / scale) if len(v) > 2 else 0.0,
            float(v[1] / scale) if len(v) > 1 else 0.0,
        ])

    point_list = ifc.createIfcCartesianPointList3D(coord_list)

    # --- IfcIndexedPolygonalFace (1-based indexing!) ---
    ifc_faces = []
    for face in faces:
        # Convert 0-based to 1-based indices
        indices_1based = [int(idx) + 1 for idx in face]
        ifc_face = ifc.createIfcIndexedPolygonalFace(indices_1based)
        ifc_faces.append(ifc_face)

    # --- IfcPolygonalFaceSet ---
    face_set = ifc.createIfcPolygonalFaceSet(
        Coordinates=point_list,
        Closed=False,  # conservative default
        Faces=ifc_faces,
    )

    # --- Shape representation ---
    body_repr = ifc.createIfcShapeRepresentation(
        ctx.body_context, "Body", "Tessellation", [face_set],
    )
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [body_repr])

    # --- Placement at origin ---
    origin = ifc.createIfcCartesianPoint([0.0, 0.0, 0.0])
    placement_3d = ifc.createIfcAxis2Placement3D(origin)
    local_placement = ifc.createIfcLocalPlacement(None, placement_3d)

    # --- Create element ---
    element = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class=ifc_class, name=name,
    )
    element.OwnerHistory = ctx.owner_history
    element.ObjectPlacement = local_placement
    element.Representation = product_shape

    # --- Surface style (color) ---
    if color and len(color) >= 3:
        _apply_surface_color(ctx, element, color, name)

    logger.debug(
        f"Created tessellated {ifc_class} '{name}': "
        f"{len(vertices)} vertices, {len(faces)} faces"
    )
    return element


def _apply_surface_color(
    ctx: IfcContext,
    element: Any,
    color: list[float],
    name: str,
) -> None:
    """Apply a surface color to a tessellated element via IfcSurfaceStyle."""
    ifc = ctx.ifc

    r, g, b = float(color[0]), float(color[1]), float(color[2])

    colour_rgb = ifc.createIfcColourRgb(None, r, g, b)
    rendering = ifc.createIfcSurfaceStyleRendering(
        SurfaceColour=colour_rgb,
        Transparency=0.0,
        ReflectanceMethod="NOTDEFINED",
    )
    style = ifc.createIfcSurfaceStyle(
        Name=f"Style_{name}",
        Side="BOTH",
        Styles=[rendering],
    )

    # Create styled item linking to the face set
    reps = element.Representation.Representations
    if reps:
        body_rep = reps[-1]  # Body representation
        items = body_rep.Items
        if items:
            ifc.createIfcStyledItem(
                Item=items[0],
                Styles=[style],
                Name=f"Color_{name}",
            )
