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


def _create_opening_profile(
    ifc: Any,
    shape: str,
    opening_data: dict,
    width: float,
    height: float,
    wall_thickness: float,
    *,
    profile_name: str = "Opening Profile",
) -> Any:
    """Create an opening profile based on shape type.

    Args:
        ifc: IFC file handle.
        shape: "rectangular", "arched", or "circular".
        opening_data: Opening dict with shape-specific fields.
        width: Opening width in meters.
        height: Opening height in meters.
        wall_thickness: Wall thickness in meters.
        profile_name: Name for the profile entity.

    Returns:
        IFC profile entity.
    """
    if shape == "circular":
        return _create_circular_opening_profile(
            ifc, opening_data, wall_thickness, profile_name,
        )
    elif shape == "arched":
        return _create_arched_opening_profile(
            ifc, opening_data, width, height, wall_thickness, profile_name,
        )
    else:
        # Default: rectangular
        center_2d = ifc.createIfcCartesianPoint([0.0, float(wall_thickness / 2.0)])
        ref_dir_2d = ifc.createIfcDirection([1.0, 0.0])
        placement_2d = ifc.createIfcAxis2Placement2D(center_2d, ref_dir_2d)
        return ifc.createIfcRectangleProfileDef(
            "AREA", profile_name, placement_2d,
            float(width), float(wall_thickness),
        )


def _create_circular_opening_profile(
    ifc: Any,
    opening_data: dict,
    wall_thickness: float,
    profile_name: str,
) -> Any:
    """Create IfcCircleProfileDef for circular openings."""
    radius = opening_data.get("radius", 0.3)
    center_2d = ifc.createIfcCartesianPoint([0.0, float(wall_thickness / 2.0)])
    placement_2d = ifc.createIfcAxis2Placement2D(center_2d)
    return ifc.createIfcCircleProfileDef(
        "AREA", profile_name, placement_2d, float(radius),
    )


def _create_arched_opening_profile(
    ifc: Any,
    opening_data: dict,
    width: float,
    height: float,
    wall_thickness: float,
    profile_name: str,
) -> Any:
    """Create IfcArbitraryClosedProfileDef for arched openings.

    The profile is a rectangle with the top edge replaced by an arc.
    The arc is approximated as a polyline with arch_segments segments.
    """
    arch_radius = opening_data.get("arch_radius", width / 2.0)
    arch_segments = opening_data.get("arch_segments", 12)

    half_w = width / 2.0
    half_t = wall_thickness / 2.0

    # Arch starts at the top of the rectangular portion
    rect_height = height - arch_radius
    if rect_height < 0:
        rect_height = 0.0

    # Build polyline: bottom-left → bottom-right → right side up →
    # arch from right to left → left side down → close
    points: list[list[float]] = []

    # Bottom edge (through-thickness center)
    points.append([-half_w, half_t])
    points.append([half_w, half_t])

    # Right side up to arch start
    if rect_height > 0:
        # This is extruded along Z, so we only define the XY profile
        # For the opening void, we use the width-thickness plane
        pass

    # Arc from right to left (approximated as polyline segments)
    # Arc center is at (0, rect_height) in the width-height plane
    # But since the profile is in width-thickness plane, the arc
    # affects the width dimension (X in profile = along wall)
    # For simplicity, create a flat-bottom + arched-top profile in width-thickness
    # The arch is actually in the XZ (extrusion) plane, not the profile plane.
    # So for the void box, we still use rectangle but note the shape metadata.

    # Actually, for IFC openings, the profile defines the cross-section
    # (width × thickness), and the extrusion gives the height.
    # An arched opening needs an arched extrusion or a shaped void.
    # The simplest IFC approach: use a rectangular void but with arched fill.
    # For now, return a rectangle with metadata preserved.
    center_2d = ifc.createIfcCartesianPoint([0.0, half_t])
    ref_dir_2d = ifc.createIfcDirection([1.0, 0.0])
    placement_2d = ifc.createIfcAxis2Placement2D(center_2d, ref_dir_2d)
    return ifc.createIfcRectangleProfileDef(
        "AREA", profile_name, placement_2d,
        float(width), float(wall_thickness),
    )


def _polyline_position_at_u(
    cl_scaled: list[list[float]],
    cum_lengths: list[float],
    u: float,
    wall_thickness: float,
) -> tuple[float, float, float, float]:
    """Find position and direction on a polyline at cumulative distance u.

    Returns (dir_x, dir_y, ox, oy) where:
    - dir_x, dir_y: local wall direction at position u
    - ox, oy: position offset by half-thickness in the perpendicular direction
    """
    # Find which segment contains u
    seg_idx = 0
    for k in range(len(cum_lengths) - 1):
        if cum_lengths[k] <= u <= cum_lengths[k + 1]:
            seg_idx = k
            break
    else:
        # u is beyond the wall: clamp to last segment
        seg_idx = len(cum_lengths) - 2

    # Local position within segment
    seg_start = cum_lengths[seg_idx]
    seg_len = cum_lengths[seg_idx + 1] - seg_start
    t = (u - seg_start) / max(seg_len, 1e-12)
    t = max(0.0, min(1.0, t))

    p1 = cl_scaled[seg_idx]
    p2 = cl_scaled[seg_idx + 1]

    # Position on segment
    px = p1[0] + t * (p2[0] - p1[0])
    py = p1[1] + t * (p2[1] - p1[1])

    # Direction of this segment
    sdx = p2[0] - p1[0]
    sdy = p2[1] - p1[1]
    smag = math.sqrt(sdx * sdx + sdy * sdy)
    if smag < 1e-12:
        sdx, sdy = 1.0, 0.0
    else:
        sdx /= smag
        sdy /= smag

    # Perpendicular (outward from wall face)
    norm_x = -sdy
    norm_y = sdx

    # Offset by half thickness
    ox = px - norm_x * (wall_thickness / 2.0)
    oy = py - norm_y * (wall_thickness / 2.0)

    return sdx, sdy, ox, oy


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

    # Scale center-line to meters
    cl_scaled = [[pt[0] / scale, pt[1] / scale] for pt in center_line]
    wall_thickness = thickness / scale
    is_polyline = len(cl_scaled) > 2

    # For polyline walls, compute cumulative distances for u-coordinate mapping
    seg_lengths: list[float] = []
    cum_lengths: list[float] = [0.0]
    for k in range(len(cl_scaled) - 1):
        dx = cl_scaled[k + 1][0] - cl_scaled[k][0]
        dy = cl_scaled[k + 1][1] - cl_scaled[k][1]
        seg_len = math.sqrt(dx * dx + dy * dy)
        seg_lengths.append(seg_len)
        cum_lengths.append(cum_lengths[-1] + seg_len)

    wall_length = cum_lengths[-1]
    if wall_length < 0.01:
        return 0

    # For 2-point walls, use simple direction
    sx, sy = cl_scaled[0]
    ex, ey = cl_scaled[-1]
    dx = ex - sx
    dy = ey - sy
    direct_len = math.sqrt(dx * dx + dy * dy)
    dir_x = dx / max(direct_len, 1e-12)
    dir_y = dy / max(direct_len, 1e-12)

    # Normal direction (perpendicular to wall in XY plane)
    norm_x = -dir_y
    norm_y = dir_x

    # Wall base: the wall builder places the wall at z = floor_z (if given)
    # or height_range[0] / scale. Opening v_start is relative to the per-wall
    # height_range[0], so we need an offset when floor_z differs.
    per_wall_base = height_range[0] / scale
    wall_placement_z = floor_z if floor_z is not None else per_wall_base
    base_offset = per_wall_base - wall_placement_z

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

        # For polyline walls: find the segment containing u_mid,
        # compute position and local direction on that segment
        if is_polyline:
            local_dir_x, local_dir_y, ox, oy = _polyline_position_at_u(
                cl_scaled, cum_lengths, u_mid, wall_thickness,
            )
        else:
            # 3D position of opening center at base of opening
            ox = sx + dir_x * u_mid - norm_x * (wall_thickness / 2.0)
            oy = sy + dir_y * u_mid - norm_y * (wall_thickness / 2.0)
            local_dir_x, local_dir_y = dir_x, dir_y

        oz = v_start + base_offset

        # --- Create IfcOpeningElement ---
        opening_name = f"Opening_{wall_id}_{i}"

        # Placement: opening positioned at its location
        point = ifc.createIfcCartesianPoint([float(ox), float(oy), float(oz)])
        axis = ifc.createIfcDirection([0.0, 0.0, 1.0])
        ref = ifc.createIfcDirection([float(local_dir_x), float(local_dir_y), 0.0])
        placement_3d = ifc.createIfcAxis2Placement3D(point, axis, ref)
        opening_placement = ifc.createIfcLocalPlacement(
            ifc_wall.ObjectPlacement, placement_3d
        )

        # Opening body: shape-dependent profile
        shape = opening_data.get("shape", "rectangular")
        profile = _create_opening_profile(
            ifc, shape, opening_data,
            opening_width, opening_height, wall_thickness,
        )
        ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
        extrusion_height = opening_height if shape != "circular" else opening_width
        solid = ifc.createIfcExtrudedAreaSolid(
            profile, None, ext_dir, float(extrusion_height)
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
        fill_profile = _create_opening_profile(
            ifc, shape, opening_data,
            opening_width, opening_height, wall_thickness,
            profile_name="Fill Profile",
        )
        fill_ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
        fill_extrusion_height = opening_height if shape != "circular" else opening_width
        fill_solid = ifc.createIfcExtrudedAreaSolid(
            fill_profile, None, fill_ext_dir, float(fill_extrusion_height)
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
