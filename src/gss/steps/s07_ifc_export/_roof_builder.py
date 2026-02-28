"""Roof builder — IfcRoof container + IfcSlab(ROOF) per roof plane.

IFC hierarchy:
  IfcRoof (container)
    └─ IfcSlab (ROOF) per roof plane

Coordinate mapping:
  Manhattan boundary_3d → IFC XY + Z (same as slab builder)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

import ifcopenshell.api
import ifcopenshell.guid

from ._ifc_builder import IfcContext

logger = logging.getLogger(__name__)


def create_roof(
    ctx: IfcContext,
    roof_planes: list[dict],
    scale: float,
    *,
    material_name: str = "Concrete",
    slab_thickness: float = 0.3,
) -> tuple[Any | None, int]:
    """Create IfcRoof with IfcSlab(ROOF) children from roof planes.

    Args:
        ctx: IFC context.
        roof_planes: List of plane dicts with label="roof".
        scale: Coordinate scale divisor.
        material_name: Roof material name.
        slab_thickness: Default slab thickness for flat roofs.

    Returns:
        (IfcRoof entity or None, number of roof slabs created).
    """
    if not roof_planes:
        return None, 0

    ifc = ctx.ifc

    # Create IfcRoof container
    ifc_roof = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcRoof", name="Roof",
    )
    ifc_roof.OwnerHistory = ctx.owner_history

    # Place roof at origin (children will have their own placements)
    origin = ifc.createIfcCartesianPoint([0.0, 0.0, 0.0])
    placement_3d = ifc.createIfcAxis2Placement3D(origin)
    ifc_roof.ObjectPlacement = ifc.createIfcLocalPlacement(None, placement_3d)

    # Assign roof to top storey
    top_storey = ctx.storey
    if ctx.storeys:
        # Find highest storey
        for name, s in ctx.storeys.items():
            if hasattr(s, 'Elevation') and s.Elevation is not None:
                if not hasattr(top_storey, 'Elevation') or s.Elevation > (top_storey.Elevation or 0):
                    top_storey = s

    ifcopenshell.api.run(
        "spatial.assign_container", ifc,
        products=[ifc_roof], relating_structure=top_storey,
    )

    # Create IfcSlab(ROOF) for each roof plane
    slab_count = 0
    roof_slabs = []

    for rp in roof_planes:
        slab = _create_roof_slab(
            ctx, rp, scale,
            material_name=material_name,
            thickness=slab_thickness,
            index=slab_count,
        )
        if slab is not None:
            roof_slabs.append(slab)
            slab_count += 1

    # Aggregate slabs into roof
    if roof_slabs:
        ifcopenshell.api.run(
            "aggregate.assign_object", ifc,
            products=roof_slabs, relating_object=ifc_roof,
        )

    logger.info(f"Created IfcRoof with {slab_count} roof slabs")
    return ifc_roof, slab_count


def _create_roof_slab(
    ctx: IfcContext,
    plane: dict,
    scale: float,
    *,
    material_name: str,
    thickness: float,
    index: int,
) -> Any | None:
    """Create a single IfcSlab with PredefinedType=ROOF from a plane."""
    boundary = plane.get("boundary_3d")
    if boundary is None or len(boundary) < 3:
        return None

    pts = np.asarray(boundary, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None

    roof_type = plane.get("roof_type", "inclined")

    if roof_type == "flat":
        # Flat roof: use boundary_3d projected to XZ, extrude up
        return _create_flat_roof_slab(
            ctx, pts, scale,
            material_name=material_name,
            thickness=thickness,
            index=index,
        )
    else:
        # Inclined roof: use boundary_3d as-is, create triangulated mesh
        return _create_inclined_roof_slab(
            ctx, pts, plane, scale,
            material_name=material_name,
            thickness=thickness,
            index=index,
        )


def _create_flat_roof_slab(
    ctx: IfcContext,
    pts: np.ndarray,
    scale: float,
    *,
    material_name: str,
    thickness: float,
    index: int,
) -> Any | None:
    """Create a flat roof slab (horizontal, extruded upward)."""
    ifc = ctx.ifc

    # Project to XZ plane (IFC XY)
    z_position = float(pts[:, 1].mean() / scale)  # Y in Manhattan = Z in IFC
    pts_2d = pts[:, [0, 2]] / scale  # X, Z → IFC X, Y

    # Close polygon
    pts_list = pts_2d.tolist()
    if pts_list and pts_list[0] != pts_list[-1]:
        pts_list.append(pts_list[0])

    if len(pts_list) < 4:
        return None

    # Profile
    profile_pts = [ifc.createIfcCartesianPoint(p) for p in pts_list]
    polyline = ifc.createIfcPolyline(profile_pts)
    profile = ifc.createIfcArbitraryClosedProfileDef(
        "AREA", "Roof Profile", polyline,
    )

    # Extrude upward
    ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
    solid = ifc.createIfcExtrudedAreaSolid(
        profile, None, ext_dir, float(thickness),
    )

    body_repr = ifc.createIfcShapeRepresentation(
        ctx.body_context, "Body", "SweptSolid", [solid],
    )
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [body_repr])

    # Placement
    point = ifc.createIfcCartesianPoint([0.0, 0.0, float(z_position)])
    axis = ifc.createIfcDirection([0.0, 0.0, 1.0])
    ref = ifc.createIfcDirection([1.0, 0.0, 0.0])
    placement_3d = ifc.createIfcAxis2Placement3D(point, axis, ref)
    local_placement = ifc.createIfcLocalPlacement(None, placement_3d)

    slab = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcSlab", name=f"Roof_Slab_{index}",
    )
    slab.OwnerHistory = ctx.owner_history
    slab.ObjectPlacement = local_placement
    slab.Representation = product_shape
    slab.PredefinedType = "ROOF"

    # Material
    material = ifc.createIfcMaterial(Name=material_name)
    ifc.createIfcRelAssociatesMaterial(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        RelatedObjects=[slab],
        RelatingMaterial=material,
    )

    return slab


def _create_inclined_roof_slab(
    ctx: IfcContext,
    pts: np.ndarray,
    plane: dict,
    scale: float,
    *,
    material_name: str,
    thickness: float,
    index: int,
) -> Any | None:
    """Create an inclined roof slab using extrusion along the plane normal."""
    ifc = ctx.ifc

    normal = np.asarray(plane["normal"], dtype=float)
    normal_unit = normal / (np.linalg.norm(normal) + 1e-12)

    # Project boundary points onto the plane's local coordinate system
    # Local X = horizontal direction along the plane
    # Local Y = direction perpendicular to normal and vertical
    centroid = pts.mean(axis=0)

    # Build local axes on the inclined plane
    up = np.array([0.0, 1.0, 0.0])
    local_x = np.cross(up, normal_unit)
    lx_norm = np.linalg.norm(local_x)
    if lx_norm < 1e-6:
        # Normal is nearly vertical — use X axis as fallback
        local_x = np.array([1.0, 0.0, 0.0])
    else:
        local_x = local_x / lx_norm
    local_y = np.cross(normal_unit, local_x)
    local_y = local_y / (np.linalg.norm(local_y) + 1e-12)

    # Project points onto local 2D coordinates
    relative = (pts - centroid) / scale
    pts_2d = np.column_stack([
        relative @ local_x,
        relative @ local_y,
    ])

    pts_list = pts_2d.tolist()
    if pts_list and pts_list[0] != pts_list[-1]:
        pts_list.append(pts_list[0])

    if len(pts_list) < 4:
        return None

    # Profile in local coordinates
    profile_pts = [ifc.createIfcCartesianPoint(p) for p in pts_list]
    polyline = ifc.createIfcPolyline(profile_pts)
    profile = ifc.createIfcArbitraryClosedProfileDef(
        "AREA", "Roof Profile", polyline,
    )

    # Extrude along plane normal
    ext_dir = ifc.createIfcDirection([
        float(normal_unit[0]), float(normal_unit[2]), float(normal_unit[1]),
    ])
    solid = ifc.createIfcExtrudedAreaSolid(
        profile, None, ext_dir, float(thickness),
    )

    body_repr = ifc.createIfcShapeRepresentation(
        ctx.body_context, "Body", "SweptSolid", [solid],
    )
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [body_repr])

    # Placement at centroid (Manhattan → IFC: x→x, z→y, y→z)
    cx = float(centroid[0] / scale)
    cy = float(centroid[2] / scale)
    cz = float(centroid[1] / scale)

    point = ifc.createIfcCartesianPoint([cx, cy, cz])
    # Axis along plane normal in IFC coords (x, z, y swap)
    axis_dir = ifc.createIfcDirection([
        float(normal_unit[0]), float(normal_unit[2]), float(normal_unit[1]),
    ])
    # Ref direction = local_x in IFC coords
    ref_dir = ifc.createIfcDirection([
        float(local_x[0]), float(local_x[2]), float(local_x[1]),
    ])
    placement_3d = ifc.createIfcAxis2Placement3D(point, axis_dir, ref_dir)
    local_placement = ifc.createIfcLocalPlacement(None, placement_3d)

    slab = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcSlab", name=f"Roof_Slab_{index}",
    )
    slab.OwnerHistory = ctx.owner_history
    slab.ObjectPlacement = local_placement
    slab.Representation = product_shape
    slab.PredefinedType = "ROOF"

    # Material
    material = ifc.createIfcMaterial(Name=material_name)
    ifc.createIfcRelAssociatesMaterial(
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=ctx.owner_history,
        RelatedObjects=[slab],
        RelatingMaterial=material,
    )

    return slab
