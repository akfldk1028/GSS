"""Opening builder — IfcOpeningElement + IfcDoor/IfcWindow.

Standard IFC pattern:
  IfcWall
    └─ IfcRelVoidsElement → IfcOpeningElement (void box)
         └─ IfcRelFillsElement → IfcDoor or IfcWindow

Coordinate mapping:
  Opening position_along_wall → offset along wall axis in meters
  Opening height_range → Z offset from wall base in meters
"""

from __future__ import annotations

import logging
import math
from typing import Any

import ifcopenshell.api
import ifcopenshell.guid

from ._ifc_builder import IfcContext

logger = logging.getLogger(__name__)


def create_openings_for_wall(
    ctx: IfcContext,
    ifc_wall: Any,
    wall_data: dict,
    scale: float,
    *,
    floor_z: float | None = None,
) -> int:
    """Create IfcOpeningElement + IfcDoor/IfcWindow for a wall's openings.

    Args:
        ctx: IFC context.
        ifc_wall: The parent IfcWall entity.
        wall_data: Wall dict that may contain "openings" list.
        scale: Coordinate scale divisor.
        floor_z: Floor Z in meters (for absolute positioning).

    Returns:
        Number of openings created.
    """
    openings = wall_data.get("openings", [])
    if not openings:
        return 0

    center_line = wall_data.get("center_line_2d")
    height_range = wall_data.get("height_range")
    thickness = wall_data.get("thickness", 0.2 * scale)
    wall_id = wall_data.get("id", 0)

    if not center_line or not height_range:
        return 0

    # Wall geometry in meters
    sx, sy = center_line[0][0] / scale, center_line[0][1] / scale
    ex, ey = center_line[1][0] / scale, center_line[1][1] / scale
    wall_thickness = thickness / scale

    # Wall base: the wall builder places the wall at z = floor_z (if given)
    # or height_range[0] / scale. Opening v_start is relative to the per-wall
    # height_range[0], so we need an offset when floor_z differs.
    per_wall_base = height_range[0] / scale
    wall_placement_z = floor_z if floor_z is not None else per_wall_base
    base_offset = per_wall_base - wall_placement_z

    # Wall direction
    dx = ex - sx
    dy = ey - sy
    wall_length = math.sqrt(dx * dx + dy * dy)
    if wall_length < 0.01:
        return 0

    dir_x = dx / wall_length
    dir_y = dy / wall_length

    # Normal direction (perpendicular to wall in XY plane)
    norm_x = -dir_y
    norm_y = dir_x

    ifc = ctx.ifc
    count = 0

    for i, opening_data in enumerate(openings):
        opening_type = opening_data.get("type", "window")
        pos_along = opening_data.get("position_along_wall", [0, 0])
        h_range = opening_data.get("height_range", [0, 0])

        # Convert from scene units to meters
        u_start = pos_along[0] / scale
        u_end = pos_along[1] / scale
        v_start = h_range[0] / scale
        v_end = h_range[1] / scale

        opening_width = u_end - u_start
        opening_height = v_end - v_start

        if opening_width < 0.01 or opening_height < 0.01:
            continue

        # Opening center position along wall
        u_mid = (u_start + u_end) / 2.0

        # 3D position of opening center at base of opening
        # Wall starts at (sx, sy) and goes to (ex, ey)
        # Opening placement is relative to wall placement (which is already
        # at z=wall_placement_z). v_start is relative to per-wall base
        # (height_range[0]/scale), so we add base_offset to compensate
        # when floor_z differs from the per-wall base.
        ox = sx + dir_x * u_mid - norm_x * (wall_thickness / 2.0)
        oy = sy + dir_y * u_mid - norm_y * (wall_thickness / 2.0)
        oz = v_start + base_offset

        # --- Create IfcOpeningElement ---
        opening_name = f"Opening_{wall_id}_{i}"

        # Placement: opening positioned at its location
        point = ifc.createIfcCartesianPoint([float(ox), float(oy), float(oz)])
        axis = ifc.createIfcDirection([0.0, 0.0, 1.0])
        ref = ifc.createIfcDirection([float(dir_x), float(dir_y), 0.0])
        placement_3d = ifc.createIfcAxis2Placement3D(point, axis, ref)
        opening_placement = ifc.createIfcLocalPlacement(
            ifc_wall.ObjectPlacement, placement_3d
        )

        # Opening body: box (width × thickness × height)
        center_2d = ifc.createIfcCartesianPoint([0.0, float(wall_thickness / 2.0)])
        ref_dir_2d = ifc.createIfcDirection([1.0, 0.0])
        placement_2d = ifc.createIfcAxis2Placement2D(center_2d, ref_dir_2d)

        profile = ifc.createIfcRectangleProfileDef(
            "AREA", "Opening Profile", placement_2d,
            float(opening_width), float(wall_thickness),
        )
        ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
        solid = ifc.createIfcExtrudedAreaSolid(
            profile, None, ext_dir, float(opening_height)
        )
        body_repr = ifc.createIfcShapeRepresentation(
            ctx.body_context, "Body", "SweptSolid", [solid]
        )
        product_shape = ifc.createIfcProductDefinitionShape(None, None, [body_repr])

        opening_element = ifcopenshell.api.run(
            "root.create_entity", ifc, ifc_class="IfcOpeningElement", name=opening_name
        )
        opening_element.OwnerHistory = ctx.owner_history
        opening_element.ObjectPlacement = opening_placement
        opening_element.Representation = product_shape

        # Void relationship: Wall → Opening
        ifc.createIfcRelVoidsElement(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            RelatingBuildingElement=ifc_wall,
            RelatedOpeningElement=opening_element,
        )

        # --- Create IfcDoor or IfcWindow ---
        if opening_type == "door":
            fill_class = "IfcDoor"
            fill_name = f"Door_{wall_id}_{i}"
        else:
            fill_class = "IfcWindow"
            fill_name = f"Window_{wall_id}_{i}"

        fill_element = ifcopenshell.api.run(
            "root.create_entity", ifc, ifc_class=fill_class, name=fill_name
        )
        fill_element.OwnerHistory = ctx.owner_history

        # Separate placement for fill element (child of opening)
        fill_origin = ifc.createIfcCartesianPoint([0.0, 0.0, 0.0])
        fill_axis3d = ifc.createIfcAxis2Placement3D(fill_origin)
        fill_placement = ifc.createIfcLocalPlacement(opening_placement, fill_axis3d)
        fill_element.ObjectPlacement = fill_placement

        # IFC requires distinct geometry objects — create separate solid for fill
        fill_center_2d = ifc.createIfcCartesianPoint([0.0, float(wall_thickness / 2.0)])
        fill_ref_dir_2d = ifc.createIfcDirection([1.0, 0.0])
        fill_placement_2d = ifc.createIfcAxis2Placement2D(fill_center_2d, fill_ref_dir_2d)
        fill_profile = ifc.createIfcRectangleProfileDef(
            "AREA", "Fill Profile", fill_placement_2d,
            float(opening_width), float(wall_thickness),
        )
        fill_ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
        fill_solid = ifc.createIfcExtrudedAreaSolid(
            fill_profile, None, fill_ext_dir, float(opening_height)
        )
        fill_body = ifc.createIfcShapeRepresentation(
            ctx.body_context, "Body", "SweptSolid", [fill_solid]
        )
        fill_shape = ifc.createIfcProductDefinitionShape(None, None, [fill_body])
        fill_element.Representation = fill_shape

        # Fill relationship: Opening → Door/Window
        ifc.createIfcRelFillsElement(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            RelatingOpeningElement=opening_element,
            RelatedBuildingElement=fill_element,
        )

        count += 1
        logger.debug(
            f"Created {opening_type} '{fill_name}': "
            f"{opening_width:.2f}m × {opening_height:.2f}m at u={u_mid:.2f}m"
        )

    if count > 0:
        logger.info(f"Wall {wall_id}: created {count} openings")
    return count
