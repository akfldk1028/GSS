"""Space builder — IfcSpace from spaces.json boundary_2d.

Coordinate mapping:
  Manhattan boundary_2d [mx, mz] → IFC XY plane (mx/s, mz/s)
  Manhattan floor_height / ceiling_height → IFC Z axis (space height)
"""

from __future__ import annotations

import logging
from typing import Any

import ifcopenshell.api
import ifcopenshell.guid

from ._ifc_builder import IfcContext

logger = logging.getLogger(__name__)


def create_space(
    ctx: IfcContext,
    space_data: dict,
    scale: float,
) -> Any | None:
    """Create an IfcSpace from a spaces.json entry.

    The space boundary is extruded from floor_height to ceiling_height.
    Attached to storey via IfcRelAggregates.
    """
    boundary = space_data.get("boundary_2d", [])
    floor_h = space_data.get("floor_height")
    ceiling_h = space_data.get("ceiling_height")
    area = space_data.get("area", 0.0)
    space_id = space_data.get("id", 0)

    if floor_h is None or ceiling_h is None or len(boundary) < 3:
        return None

    height = (ceiling_h - floor_h) / scale
    if height <= 0:
        logger.warning(f"Space {space_id}: non-positive height, skipping")
        return None

    z_floor = floor_h / scale

    ifc = ctx.ifc

    # Convert boundary to IFC XY points
    pts_2d = []
    for pt in boundary:
        pts_2d.append([float(pt[0] / scale), float(pt[1] / scale)])

    # Close polygon if needed
    if pts_2d and pts_2d[0] != pts_2d[-1]:
        pts_2d.append(pts_2d[0])

    if len(pts_2d) < 4:
        return None

    # Profile
    profile_pts = [ifc.createIfcCartesianPoint(p) for p in pts_2d]
    polyline = ifc.createIfcPolyline(profile_pts)
    profile = ifc.createIfcArbitraryClosedProfileDef("AREA", "Space Profile", polyline)

    # Extrude upward
    ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
    solid = ifc.createIfcExtrudedAreaSolid(profile, None, ext_dir, float(height))

    body_repr = ifc.createIfcShapeRepresentation(
        ctx.body_context, "Body", "SweptSolid", [solid]
    )
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [body_repr])

    # Placement at z=floor_height
    point = ifc.createIfcCartesianPoint([0.0, 0.0, float(z_floor)])
    axis_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
    ref_dir = ifc.createIfcDirection([1.0, 0.0, 0.0])
    placement_3d = ifc.createIfcAxis2Placement3D(point, axis_dir, ref_dir)
    local_placement = ifc.createIfcLocalPlacement(None, placement_3d)

    # IfcSpace
    ifc_space = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcSpace", name=f"Room_{space_id}"
    )
    ifc_space.OwnerHistory = ctx.owner_history
    ifc_space.ObjectPlacement = local_placement
    ifc_space.Representation = product_shape

    # Aggregate to storey
    ifcopenshell.api.run(
        "aggregate.assign_object", ifc, products=[ifc_space], relating_object=ctx.storey
    )

    # Property set: GrossFloorArea
    area_m2 = area / (scale * scale)
    props = [
        ifc.createIfcPropertySingleValue(
            Name="GrossFloorArea",
            NominalValue=ifc.create_entity("IfcAreaMeasure", float(area_m2)),
        ),
    ]
    pset = ifc.createIfcPropertySet(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        Name="Pset_SpaceCommon",
        HasProperties=props,
    )
    ifc.createIfcRelDefinesByProperties(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        RelatedObjects=[ifc_space],
        RelatingPropertyDefinition=pset,
    )

    return ifc_space
