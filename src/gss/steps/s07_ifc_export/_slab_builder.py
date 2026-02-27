"""Slab builder — floor/ceiling from spaces.json boundary_2d.

Coordinate mapping:
  Manhattan boundary_2d [mx, mz] → IFC XY plane (mx/s, mz/s)
  Manhattan floor_height / ceiling_height → IFC Z axis
"""

from __future__ import annotations

import logging
from typing import Any

import ifcopenshell.api
import ifcopenshell.guid

from ._ifc_builder import IfcContext

logger = logging.getLogger(__name__)


def create_floor_slab(
    ctx: IfcContext,
    space_data: dict,
    scale: float,
    *,
    thickness: float = 0.3,
    material_name: str = "Concrete",
) -> Any | None:
    """Create a floor IfcSlab from space boundary_2d.

    Floor slab at z=floor_height, extruded downward by thickness.
    """
    boundary = space_data.get("boundary_2d", [])
    floor_h = space_data.get("floor_height")
    if floor_h is None or len(boundary) < 3:
        return None

    space_id = space_data.get("id", 0)
    z = floor_h / scale

    return _create_slab(
        ctx,
        boundary,
        scale,
        z_position=z,
        extrude_down=True,
        thickness=thickness / scale,
        material_name=material_name,
        name=f"Floor_Room{space_id}",
    )


def create_ceiling_slab(
    ctx: IfcContext,
    space_data: dict,
    scale: float,
    *,
    thickness: float = 0.3,
    material_name: str = "Concrete",
) -> Any | None:
    """Create a ceiling IfcSlab from space boundary_2d.

    Ceiling slab at z=ceiling_height, extruded upward by thickness.
    """
    boundary = space_data.get("boundary_2d", [])
    ceiling_h = space_data.get("ceiling_height")
    if ceiling_h is None or len(boundary) < 3:
        return None

    space_id = space_data.get("id", 0)
    z = ceiling_h / scale

    return _create_slab(
        ctx,
        boundary,
        scale,
        z_position=z,
        extrude_down=False,
        thickness=thickness / scale,
        material_name=material_name,
        name=f"Ceiling_Room{space_id}",
    )


def _create_slab(
    ctx: IfcContext,
    boundary_2d: list[list[float]],
    scale: float,
    *,
    z_position: float,
    extrude_down: bool,
    thickness: float,
    material_name: str,
    name: str,
) -> Any | None:
    """Internal: create slab from 2D boundary polygon."""
    ifc = ctx.ifc

    # Convert boundary to IFC XY points (scale from Manhattan to meters)
    pts_2d = []
    for pt in boundary_2d:
        pts_2d.append([float(pt[0] / scale), float(pt[1] / scale)])

    # Close polygon if needed
    if pts_2d and pts_2d[0] != pts_2d[-1]:
        pts_2d.append(pts_2d[0])

    if len(pts_2d) < 4:  # need at least 3 unique + 1 closing
        return None

    # Profile from polygon
    profile_pts = [ifc.createIfcCartesianPoint(p) for p in pts_2d]
    polyline = ifc.createIfcPolyline(profile_pts)
    profile = ifc.createIfcArbitraryClosedProfileDef("AREA", "Slab Profile", polyline)

    # Extrusion: floor goes down, ceiling goes up
    if extrude_down:
        ext_dir = ifc.createIfcDirection([0.0, 0.0, -1.0])
    else:
        ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])

    solid = ifc.createIfcExtrudedAreaSolid(profile, None, ext_dir, float(thickness))

    body_repr = ifc.createIfcShapeRepresentation(
        ctx.body_context, "Body", "SweptSolid", [solid]
    )
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [body_repr])

    # Slab placement at z_position
    point = ifc.createIfcCartesianPoint([0.0, 0.0, float(z_position)])
    axis = ifc.createIfcDirection([0.0, 0.0, 1.0])
    ref = ifc.createIfcDirection([1.0, 0.0, 0.0])
    placement_3d = ifc.createIfcAxis2Placement3D(point, axis, ref)
    local_placement = ifc.createIfcLocalPlacement(None, placement_3d)

    slab = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcSlab", name=name
    )
    slab.OwnerHistory = ctx.owner_history
    slab.ObjectPlacement = local_placement
    slab.Representation = product_shape

    # Material
    material = ifc.createIfcMaterial(Name=material_name)
    ifc.createIfcRelAssociatesMaterial(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        RelatedObjects=[slab],
        RelatingMaterial=material,
    )

    return slab
