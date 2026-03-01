"""Column builder — IfcColumn from columns.json.

Supports:
- Round columns: IfcCircleProfileDef (Cloud2BIM pattern)
- Rectangular columns: IfcRectangleProfileDef

Coordinate mapping:
  Manhattan center_2d [mx, mz] → IFC XY plane (mx/s, mz/s)
  Manhattan height_range [y_min, y_max] → IFC Z axis
"""

from __future__ import annotations

import logging
import math
from typing import Any

import ifcopenshell.api
import ifcopenshell.guid

from ._ifc_builder import IfcContext

logger = logging.getLogger(__name__)


def create_column(
    ctx: IfcContext,
    column_data: dict,
    scale: float,
    *,
    floor_z: float | None = None,
) -> Any | None:
    """Create an IfcColumn from a columns.json entry.

    Args:
        ctx: IFC context.
        column_data: Dict with column_type, center_2d, height_range, etc.
        scale: Coordinate scale divisor.
        floor_z: Override floor Z in meters.

    Returns:
        IfcColumn entity, or None if geometry is degenerate.
    """
    col_type = column_data.get("column_type", "rectangular")
    center = column_data.get("center_2d")
    height_range = column_data.get("height_range")
    col_id = column_data.get("id", 0)

    if not center or not height_range:
        return None

    # Convert to IFC coordinates (meters)
    cx = center[0] / scale
    cy = center[1] / scale

    y_min = floor_z if floor_z is not None else height_range[0] / scale
    y_max = height_range[1] / scale
    col_height = y_max - y_min

    if col_height < 0.01:
        logger.warning(f"Column {col_id}: degenerate height, skipping")
        return None

    ifc = ctx.ifc

    # --- Profile ---
    if col_type == "round":
        radius = column_data.get("radius", 0.15)
        profile = _create_circle_profile(ifc, radius)
    else:
        width = column_data.get("width", 0.3)
        depth = column_data.get("depth", 0.3)
        direction = column_data.get("direction", [1.0, 0.0])
        profile = _create_rectangle_profile(ifc, width, depth)

    # --- Extrude upward ---
    ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
    solid = ifc.createIfcExtrudedAreaSolid(profile, None, ext_dir, float(col_height))

    body_repr = ifc.createIfcShapeRepresentation(
        ctx.body_context, "Body", "SweptSolid", [solid],
    )
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [body_repr])

    # --- Placement at center, z=y_min ---
    point = ifc.createIfcCartesianPoint([float(cx), float(cy), float(y_min)])
    axis = ifc.createIfcDirection([0.0, 0.0, 1.0])

    # For rectangular columns, orient by direction
    if col_type != "round" and "direction" in column_data:
        d = column_data["direction"]
        ref = ifc.createIfcDirection([float(d[0]), float(d[1]), 0.0])
    else:
        ref = ifc.createIfcDirection([1.0, 0.0, 0.0])

    placement_3d = ifc.createIfcAxis2Placement3D(point, axis, ref)
    local_placement = ifc.createIfcLocalPlacement(None, placement_3d)

    # --- IfcColumn entity ---
    column = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcColumn",
        name=f"Column_{col_id}",
    )
    column.OwnerHistory = ctx.owner_history
    column.ObjectPlacement = local_placement
    column.Representation = product_shape

    # --- Column type ---
    _assign_column_type(ctx, column, col_type)

    # --- Property set ---
    _create_column_pset(ctx, column, column_data)

    logger.debug(f"Created IfcColumn {col_id}: {col_type}")
    return column


def _create_circle_profile(ifc: Any, radius: float) -> Any:
    """Create IfcCircleProfileDef for round columns."""
    center = ifc.createIfcCartesianPoint([0.0, 0.0])
    placement = ifc.createIfcAxis2Placement2D(center)
    return ifc.createIfcCircleProfileDef(
        "AREA", "Column Profile", placement, float(radius),
    )


def _create_rectangle_profile(ifc: Any, width: float, depth: float) -> Any:
    """Create IfcRectangleProfileDef for rectangular columns."""
    center = ifc.createIfcCartesianPoint([0.0, 0.0])
    placement = ifc.createIfcAxis2Placement2D(center)
    return ifc.createIfcRectangleProfileDef(
        "AREA", "Column Profile", placement, float(width), float(depth),
    )


def _assign_column_type(ctx: IfcContext, column: Any, col_type: str) -> None:
    """Create or reuse IfcColumnType."""
    cache_key = col_type
    if not hasattr(ctx, '_column_type_cache'):
        ctx._column_type_cache = {}

    if cache_key not in ctx._column_type_cache:
        ifc = ctx.ifc
        column_type = ifc.createIfcColumnType(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            Name=f"Column_{col_type.capitalize()}",
            PredefinedType="COLUMN",
        )
        ifc.createIfcRelDeclares(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            RelatingContext=ctx.project,
            RelatedDefinitions=[column_type],
        )
        ctx._column_type_cache[cache_key] = column_type

    column_type = ctx._column_type_cache[cache_key]
    ctx.ifc.createIfcRelDefinesByType(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        RelatedObjects=[column],
        RelatingType=column_type,
    )


def _create_column_pset(ctx: IfcContext, column: Any, column_data: dict) -> None:
    """Attach property set to column."""
    ifc = ctx.ifc
    props = []

    col_type = column_data.get("column_type", "rectangular")
    props.append(ifc.createIfcPropertySingleValue(
        Name="ColumnType",
        NominalValue=ifc.create_entity("IfcLabel", col_type),
    ))

    source_wall_id = column_data.get("source_wall_id")
    if source_wall_id is not None:
        props.append(ifc.createIfcPropertySingleValue(
            Name="SourceWallId",
            NominalValue=ifc.create_entity("IfcInteger", int(source_wall_id)),
        ))

    if props:
        pset = ifc.createIfcPropertySet(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            Name="Pset_ColumnCommon",
            HasProperties=props,
        )
        ifc.createIfcRelDefinesByProperties(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            RelatedObjects=[column],
            RelatingPropertyDefinition=pset,
        )
