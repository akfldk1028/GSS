"""Wall builder — center-line based wall geometry (Cloud2BIM pattern).

Coordinate mapping:
  Manhattan center_line_2d [mx, mz] → IFC XY plane (mx, mz)
  Manhattan height_range [y_min, y_max] → IFC Z axis (wall height)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import ifcopenshell.api
import ifcopenshell.guid

from ._ifc_builder import IfcContext

logger = logging.getLogger(__name__)


def create_wall_from_centerline(
    ctx: IfcContext,
    wall_data: dict,
    scale: float,
    *,
    floor_z: float | None = None,
    ceiling_z: float | None = None,
    wall_material_name: str = "Concrete",
    default_thickness: float = 0.2,
    create_axis: bool = True,
    create_material_layers: bool = True,
    create_wall_type: bool = True,
    create_property_set: bool = True,
) -> Any | None:
    """Create an IfcWall from a walls.json entry.

    Args:
        ctx: IFC context with file and shared entities.
        wall_data: Dict with center_line_2d, thickness, height_range, etc.
        scale: coordinate_scale divisor for meter conversion.
        floor_z: Uniform floor Z in meters (overrides per-wall height_range min).
        ceiling_z: Uniform ceiling Z in meters (overrides per-wall height_range max).
        Other kwargs control optional IFC features.

    Returns:
        IfcWall entity, or None if geometry is degenerate.
    """
    center_line = wall_data.get("center_line_2d")
    if not center_line or len(center_line) < 2:
        return None

    height_range = wall_data.get("height_range")
    if not height_range or len(height_range) < 2:
        return None

    thickness = wall_data.get("thickness", default_thickness)
    wall_id = wall_data.get("id", 0)
    is_synthetic = wall_data.get("synthetic", False)

    is_polyline = len(center_line) > 2

    # Manhattan coords → IFC (meters)
    cl_scaled = [[pt[0] / scale, pt[1] / scale] for pt in center_line]

    sx, sy = cl_scaled[0]
    ex, ey = cl_scaled[-1]

    # Use uniform storey heights when available (Cloud2BIM: walls span floor→ceiling)
    y_min = floor_z if floor_z is not None else height_range[0] / scale
    y_max = ceiling_z if ceiling_z is not None else height_range[1] / scale
    wall_height = y_max - y_min
    wall_thickness = thickness / scale

    if wall_height < 0.01 or wall_thickness < 0.001:
        logger.warning(f"Wall {wall_id}: degenerate geometry, skipping")
        return None

    # Compute total wall length
    wall_length = 0.0
    for i in range(len(cl_scaled) - 1):
        dx = cl_scaled[i + 1][0] - cl_scaled[i][0]
        dy = cl_scaled[i + 1][1] - cl_scaled[i][1]
        wall_length += math.sqrt(dx * dx + dy * dy)
    if wall_length < 0.01:
        logger.warning(f"Wall {wall_id}: zero length, skipping")
        return None

    ifc = ctx.ifc

    # --- Wall placement at z=y_min ---
    wall_placement = _create_local_placement(ifc, z=y_min)

    # --- Axis representation (center-line as Curve2D) ---
    representations = []
    if create_axis and ctx.axis_context is not None:
        if is_polyline:
            axis_repr = _create_axis_representation_polyline(
                ifc, ctx.axis_context, cl_scaled,
            )
        else:
            axis_repr = _create_axis_representation(
                ifc, ctx.axis_context, sx, sy, ex, ey,
            )
        representations.append(axis_repr)

    # --- Body representation (IfcArbitraryClosedProfileDef polyline + extrusion) ---
    if is_polyline:
        body_repr = _create_body_representation_polyline(
            ifc, ctx.body_context, cl_scaled, wall_thickness, wall_height,
        )
    else:
        body_repr = _create_body_representation(
            ifc, ctx.body_context, sx, sy, ex, ey, wall_thickness, wall_height,
        )
    representations.append(body_repr)

    product_shape = ifc.createIfcProductDefinitionShape(None, None, representations)

    # --- IfcWall entity ---
    wall_name = f"Wall_{wall_id}"
    if is_synthetic:
        wall_name = f"Wall_{wall_id}_Synthetic"

    wall = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcWall", name=wall_name
    )
    wall.OwnerHistory = ctx.owner_history
    wall.ObjectPlacement = wall_placement
    wall.Representation = product_shape

    # --- Material layer set ---
    if create_material_layers:
        _assign_material_layer_set(ctx, wall, wall_thickness, wall_material_name)

    # --- Wall type ---
    if create_wall_type:
        _assign_wall_type(ctx, wall, wall_thickness)

    # --- Property set ---
    if create_property_set:
        _create_property_set(ctx, wall, wall_data)

    return wall


# ---------- Internal helpers ----------


def _create_local_placement(ifc: Any, z: float = 0.0) -> Any:
    """Create IfcLocalPlacement at (0, 0, z)."""
    point = ifc.createIfcCartesianPoint([0.0, 0.0, float(z)])
    axis = ifc.createIfcDirection([0.0, 0.0, 1.0])
    ref = ifc.createIfcDirection([1.0, 0.0, 0.0])
    placement_3d = ifc.createIfcAxis2Placement3D(point, axis, ref)
    return ifc.createIfcLocalPlacement(None, placement_3d)


def _create_axis_representation(
    ifc: Any, axis_ctx: Any, sx: float, sy: float, ex: float, ey: float
) -> Any:
    """Axis representation: center-line polyline in 2D."""
    p1 = ifc.createIfcCartesianPoint([float(sx), float(sy)])
    p2 = ifc.createIfcCartesianPoint([float(ex), float(ey)])
    polyline = ifc.createIfcPolyline([p1, p2])
    return ifc.createIfcShapeRepresentation(
        axis_ctx, "Axis", "Curve2D", [polyline]
    )


def _create_axis_representation_polyline(
    ifc: Any, axis_ctx: Any, cl_scaled: list[list[float]],
) -> Any:
    """Axis representation: N-point center-line polyline in 2D."""
    pts = [ifc.createIfcCartesianPoint([float(p[0]), float(p[1])]) for p in cl_scaled]
    polyline = ifc.createIfcPolyline(pts)
    return ifc.createIfcShapeRepresentation(
        axis_ctx, "Axis", "Curve2D", [polyline]
    )


def _create_body_representation_polyline(
    ifc: Any,
    body_ctx: Any,
    cl_scaled: list[list[float]],
    wall_thickness: float,
    wall_height: float,
) -> Any:
    """Body representation for N-point polyline wall using Shapely buffer.

    Creates a closed polygon by buffering the polyline center-line,
    then extrudes upward.
    """
    try:
        from shapely.geometry import LineString
    except ImportError:
        logger.warning("Shapely not available, falling back to 2-point wall")
        sx, sy = cl_scaled[0]
        ex, ey = cl_scaled[-1]
        return _create_body_representation(
            ifc, body_ctx, sx, sy, ex, ey, wall_thickness, wall_height,
        )

    line = LineString(cl_scaled)
    half_t = wall_thickness / 2.0
    buffered = line.buffer(half_t, cap_style=2, join_style=2)  # flat cap, mitre join

    # Extract exterior polygon coordinates
    polygon_coords = list(buffered.exterior.coords)

    # Create IFC polyline from polygon
    ifc_pts = [
        ifc.createIfcCartesianPoint([float(c[0]), float(c[1])])
        for c in polygon_coords
    ]
    polyline = ifc.createIfcPolyline(ifc_pts)
    profile = ifc.createIfcArbitraryClosedProfileDef("AREA", "Wall Profile", polyline)

    # Extrude upward (z direction)
    ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
    solid = ifc.createIfcExtrudedAreaSolid(profile, None, ext_dir, float(wall_height))

    return ifc.createIfcShapeRepresentation(
        body_ctx, "Body", "SweptSolid", [solid]
    )


def _create_body_representation(
    ifc: Any,
    body_ctx: Any,
    sx: float,
    sy: float,
    ex: float,
    ey: float,
    wall_thickness: float,
    wall_height: float,
) -> Any:
    """Body representation: IfcArbitraryClosedProfileDef polyline extruded upward.

    Constructs a closed rectangular polyline from center-line endpoints offset
    by ±thickness/2 in the perpendicular direction, then extrudes upward.
    This replaces IfcRectangleProfileDef for better compatibility with
    non-axis-aligned and future multi-segment walls.
    """
    # Perpendicular direction (90° CCW from wall direction)
    dx = ex - sx
    dy = ey - sy
    mag = math.sqrt(dx * dx + dy * dy)
    perp_x = float(-dy / mag)
    perp_y = float(dx / mag)
    half_t = wall_thickness / 2.0

    # 4 corners offset from center-line, closed (5 points)
    corners = [
        [sx + perp_x * half_t, sy + perp_y * half_t],  # start-left
        [ex + perp_x * half_t, ey + perp_y * half_t],  # end-left
        [ex - perp_x * half_t, ey - perp_y * half_t],  # end-right
        [sx - perp_x * half_t, sy - perp_y * half_t],  # start-right
        [sx + perp_x * half_t, sy + perp_y * half_t],  # close
    ]

    ifc_pts = [ifc.createIfcCartesianPoint([float(c[0]), float(c[1])]) for c in corners]
    polyline = ifc.createIfcPolyline(ifc_pts)
    profile = ifc.createIfcArbitraryClosedProfileDef("AREA", "Wall Profile", polyline)

    # Extrude upward (z direction)
    ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
    solid = ifc.createIfcExtrudedAreaSolid(profile, None, ext_dir, float(wall_height))

    return ifc.createIfcShapeRepresentation(
        body_ctx, "Body", "SweptSolid", [solid]
    )


def _assign_material_layer_set(
    ctx: IfcContext,
    wall: Any,
    thickness: float,
    material_name: str,
) -> None:
    """Create or reuse IfcMaterialLayerSet and assign to wall."""
    # Round thickness for cache key
    t_key = round(thickness, 4)

    if t_key not in ctx._material_layer_set_cache:
        ifc = ctx.ifc
        material = ifc.createIfcMaterial(Name=material_name)
        layer = ifc.createIfcMaterialLayer(
            Material=material,
            LayerThickness=float(thickness),
            Name="Core",
        )
        thickness_mm = int(round(thickness * 1000))
        layer_set = ifc.createIfcMaterialLayerSet(
            MaterialLayers=[layer],
            LayerSetName=f"{material_name} Wall - {thickness_mm} mm",
        )
        ctx._material_layer_set_cache[t_key] = layer_set

    layer_set = ctx._material_layer_set_cache[t_key]

    ifc = ctx.ifc
    usage = ifc.createIfcMaterialLayerSetUsage(
        ForLayerSet=layer_set,
        LayerSetDirection="AXIS2",
        DirectionSense="POSITIVE",
        OffsetFromReferenceLine=-(thickness / 2.0),
    )
    ifc.createIfcRelAssociatesMaterial(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        RelatedObjects=[wall],
        RelatingMaterial=usage,
    )


def _assign_wall_type(ctx: IfcContext, wall: Any, thickness: float) -> None:
    """Create or reuse IfcWallType for this thickness."""
    t_key = round(thickness, 4)

    if t_key not in ctx._wall_type_cache:
        ifc = ctx.ifc
        thickness_mm = int(round(thickness * 1000))
        wall_type = ifc.createIfcWallType(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            Name=f"Wall_{thickness_mm}mm",
            Description=f"Wall - {thickness_mm} mm",
            PredefinedType="SOLIDWALL",
        )
        # Declare type in project
        ifc.createIfcRelDeclares(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            RelatingContext=ctx.project,
            RelatedDefinitions=[wall_type],
        )
        ctx._wall_type_cache[t_key] = wall_type

    wall_type = ctx._wall_type_cache[t_key]
    ctx.ifc.createIfcRelDefinesByType(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        RelatedObjects=[wall],
        RelatingType=wall_type,
    )


def _create_property_set(ctx: IfcContext, wall: Any, wall_data: dict) -> None:
    """Attach a Pset_WallCommon-like property set."""
    ifc = ctx.ifc
    props = []

    # IsExternal — from s06b exterior classification if available
    is_external = wall_data.get("is_exterior", True)
    props.append(
        ifc.createIfcPropertySingleValue(
            Name="IsExternal",
            NominalValue=ifc.create_entity("IfcBoolean", is_external),
        )
    )

    # Synthetic flag
    if wall_data.get("synthetic", False):
        props.append(
            ifc.createIfcPropertySingleValue(
                Name="Synthetic",
                NominalValue=ifc.create_entity("IfcBoolean", True),
            )
        )

    # Source (detected vs synthetic)
    source = "synthetic" if wall_data.get("synthetic", False) else "detected"
    props.append(
        ifc.createIfcPropertySingleValue(
            Name="Source",
            NominalValue=ifc.create_entity("IfcLabel", source),
        )
    )

    # Normal axis
    normal_axis = wall_data.get("normal_axis", "")
    if normal_axis:
        props.append(
            ifc.createIfcPropertySingleValue(
                Name="NormalAxis",
                NominalValue=ifc.create_entity("IfcLabel", normal_axis),
            )
        )

    pset = ifc.createIfcPropertySet(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        Name="Pset_WallCommon",
        HasProperties=props,
    )
    ifc.createIfcRelDefinesByProperties(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        RelatedObjects=[wall],
        RelatingPropertyDefinition=pset,
    )
