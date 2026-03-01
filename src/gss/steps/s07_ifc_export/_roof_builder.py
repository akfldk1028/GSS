"""Roof builder — IfcRoof container + IfcSlab(ROOF) per roof plane.

Supports two modes:
1. Basic: roof planes from planes.json (label="roof") → flat/inclined slabs
2. Structured: building_context.json roof_structure → PredefinedType + annotations

IFC hierarchy:
  IfcRoof (container, PredefinedType from roof_type)
    +-- IfcSlab (ROOF) per roof plane
    +-- IfcAnnotation per ridge/eave/valley (optional)

Coordinate mapping:
  Manhattan boundary_3d -> IFC XY + Z (same as slab builder)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

import ifcopenshell.api
import ifcopenshell.guid

from ._ifc_builder import IfcContext

logger = logging.getLogger(__name__)

# Mapping from s06c roof_type to IFC PredefinedType
_ROOF_TYPE_MAP = {
    "gable": "GABLE_ROOF",
    "hip": "HIP_ROOF",
    "shed": "SHED_ROOF",
    "flat": "FLAT_ROOF",
    "mixed": "FREEFORM",
    "none": "NOTDEFINED",
}


def create_structured_roof(
    ctx: IfcContext,
    roof_planes: list[dict],
    roof_structure: dict,
    scale: float,
    *,
    material_name: str = "Concrete",
    slab_thickness: float = 0.3,
    create_annotations: bool = True,
) -> tuple[Any | None, int, str]:
    """Create IfcRoof with structured metadata from s06c roof_structure.

    Args:
        ctx: IFC context.
        roof_planes: List of plane dicts with label="roof".
        roof_structure: Dict from building_context.json with
            roof_type, faces, ridges, eaves, valleys.
        scale: Coordinate scale divisor.
        material_name: Roof material name.
        slab_thickness: Default slab thickness.
        create_annotations: Create IfcAnnotation for ridges/eaves/valleys.

    Returns:
        (IfcRoof entity or None, num_roof_slabs, roof_type_str).
    """
    roof_type = roof_structure.get("roof_type", "none")
    faces = roof_structure.get("faces", [])

    if not roof_planes and not faces:
        return None, 0, roof_type

    ifc = ctx.ifc

    # Create IfcRoof container with PredefinedType
    ifc_roof = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcRoof", name="Roof",
    )
    ifc_roof.OwnerHistory = ctx.owner_history
    ifc_roof.PredefinedType = _ROOF_TYPE_MAP.get(roof_type, "NOTDEFINED")

    # Place roof at origin
    origin = ifc.createIfcCartesianPoint([0.0, 0.0, 0.0])
    placement_3d = ifc.createIfcAxis2Placement3D(origin)
    ifc_roof.ObjectPlacement = ifc.createIfcLocalPlacement(None, placement_3d)

    # Assign to top storey
    top_storey = _find_top_storey(ctx)
    ifcopenshell.api.run(
        "spatial.assign_container", ifc,
        products=[ifc_roof], relating_structure=top_storey,
    )

    # Create roof slabs (enriched with per-face metadata)
    slab_count = 0
    roof_slabs = []
    face_lookup = {f["plane_id"]: f for f in faces} if faces else {}

    for rp in roof_planes:
        face = face_lookup.get(rp["id"])
        slab = _create_roof_slab(
            ctx, rp, scale,
            material_name=material_name,
            thickness=slab_thickness,
            index=slab_count,
        )
        if slab is not None:
            # Add per-face property set
            if face:
                _create_roof_face_pset(ctx, slab, face)
            roof_slabs.append(slab)
            slab_count += 1

    # Aggregate slabs into roof
    if roof_slabs:
        ifcopenshell.api.run(
            "aggregate.assign_object", ifc,
            products=roof_slabs, relating_object=ifc_roof,
        )

    # Create annotations for ridges/eaves/valleys
    if create_annotations:
        ridges = roof_structure.get("ridges", [])
        eaves = roof_structure.get("eaves", [])
        valleys = roof_structure.get("valleys", [])

        for i, seg in enumerate(ridges):
            _create_line_annotation(ctx, seg, scale, f"Ridge_{i}", ifc_roof)
        for i, seg in enumerate(eaves):
            _create_line_annotation(ctx, seg, scale, f"Eave_{i}", ifc_roof)
        for i, seg in enumerate(valleys):
            _create_line_annotation(ctx, seg, scale, f"Valley_{i}", ifc_roof)

        total_annot = len(ridges) + len(eaves) + len(valleys)
        if total_annot > 0:
            logger.info(
                f"Roof annotations: {len(ridges)} ridges, "
                f"{len(eaves)} eaves, {len(valleys)} valleys"
            )

    # Pset_RoofCommon
    _create_roof_pset(ctx, ifc_roof, roof_structure, roof_planes, scale)

    logger.info(
        f"Created structured IfcRoof: type={roof_type} "
        f"({_ROOF_TYPE_MAP.get(roof_type, 'NOTDEFINED')}), "
        f"{slab_count} slabs"
    )
    return ifc_roof, slab_count, roof_type


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
    top_storey = _find_top_storey(ctx)
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


# ---------- Internal helpers ----------


def _find_top_storey(ctx: IfcContext) -> Any:
    """Find the highest elevation storey."""
    top_storey = ctx.storey
    if ctx.storeys:
        for name, s in ctx.storeys.items():
            if hasattr(s, 'Elevation') and s.Elevation is not None:
                if not hasattr(top_storey, 'Elevation') or s.Elevation > (top_storey.Elevation or 0):
                    top_storey = s
    return top_storey


def _create_roof_face_pset(ctx: IfcContext, slab: Any, face: dict) -> None:
    """Add per-face property set with slope and aspect."""
    ifc = ctx.ifc
    props = []

    slope = face.get("slope_deg")
    if slope is not None:
        props.append(ifc.createIfcPropertySingleValue(
            Name="SlopeDegrees",
            NominalValue=ifc.create_entity("IfcReal", float(slope)),
        ))

    aspect = face.get("aspect", "")
    if aspect:
        props.append(ifc.createIfcPropertySingleValue(
            Name="Aspect",
            NominalValue=ifc.create_entity("IfcLabel", aspect),
        ))

    sub_type = face.get("sub_type", "")
    if sub_type:
        props.append(ifc.createIfcPropertySingleValue(
            Name="SubType",
            NominalValue=ifc.create_entity("IfcLabel", sub_type),
        ))

    if props:
        pset = ifc.createIfcPropertySet(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            Name="Pset_RoofFace",
            HasProperties=props,
        )
        ifc.createIfcRelDefinesByProperties(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            RelatedObjects=[slab],
            RelatingPropertyDefinition=pset,
        )


def _create_roof_pset(
    ctx: IfcContext,
    ifc_roof: Any,
    roof_structure: dict,
    roof_planes: list[dict],
    scale: float,
) -> None:
    """Create Pset_RoofCommon with TotalArea and ProjectedArea."""
    ifc = ctx.ifc
    props = []

    # Compute total area and projected area from roof planes
    total_area = 0.0
    projected_area = 0.0

    for rp in roof_planes:
        boundary = rp.get("boundary_3d")
        if boundary is None or len(boundary) < 3:
            continue
        pts = np.asarray(boundary, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            continue

        # 3D area via cross products of triangle fan
        centroid = pts.mean(axis=0)
        area_3d = 0.0
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            cross = np.cross(pts[i] - centroid, pts[j] - centroid)
            area_3d += np.linalg.norm(cross) / 2.0
        total_area += area_3d / (scale * scale)

        # Projected area (XZ plane in Manhattan = XY in IFC)
        pts_2d = pts[:, [0, 2]] / scale
        area_2d = 0.0
        for i in range(len(pts_2d)):
            j = (i + 1) % len(pts_2d)
            area_2d += pts_2d[i][0] * pts_2d[j][1] - pts_2d[j][0] * pts_2d[i][1]
        projected_area += abs(area_2d) / 2.0

    if total_area > 0:
        props.append(ifc.createIfcPropertySingleValue(
            Name="TotalArea",
            NominalValue=ifc.create_entity("IfcAreaMeasure", float(total_area)),
        ))
    if projected_area > 0:
        props.append(ifc.createIfcPropertySingleValue(
            Name="ProjectedArea",
            NominalValue=ifc.create_entity("IfcAreaMeasure", float(projected_area)),
        ))

    if props:
        pset = ifc.createIfcPropertySet(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            Name="Pset_RoofCommon",
            HasProperties=props,
        )
        ifc.createIfcRelDefinesByProperties(
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=ctx.owner_history,
            RelatedObjects=[ifc_roof],
            RelatingPropertyDefinition=pset,
        )


def _create_line_annotation(
    ctx: IfcContext,
    segment: list[list[float]],
    scale: float,
    name: str,
    parent: Any,
) -> Any | None:
    """Create IfcAnnotation with IfcPolyline for a ridge/eave/valley line.

    Coordinate mapping: Manhattan [x, y, z] -> IFC [x, z, y].
    """
    if not segment or len(segment) < 2:
        return None

    ifc = ctx.ifc

    # Convert points: Manhattan (x,y,z) -> IFC (x/s, z/s, y/s)
    ifc_pts = []
    for pt in segment:
        ifc_pts.append(ifc.createIfcCartesianPoint([
            float(pt[0] / scale),
            float(pt[2] / scale),
            float(pt[1] / scale),
        ]))

    polyline = ifc.createIfcPolyline(ifc_pts)

    # Curve representation
    curve_repr = ifc.createIfcShapeRepresentation(
        ctx.body_context, "Body", "Curve", [polyline],
    )
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [curve_repr])

    annotation = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcAnnotation", name=name,
    )
    annotation.OwnerHistory = ctx.owner_history
    annotation.Representation = product_shape

    # Place at origin
    origin = ifc.createIfcCartesianPoint([0.0, 0.0, 0.0])
    placement_3d = ifc.createIfcAxis2Placement3D(origin)
    annotation.ObjectPlacement = ifc.createIfcLocalPlacement(None, placement_3d)

    return annotation


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
    pts_2d = pts[:, [0, 2]] / scale  # X, Z -> IFC X, Y

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
    centroid = pts.mean(axis=0)

    # Build local axes on the inclined plane
    up = np.array([0.0, 1.0, 0.0])
    local_x = np.cross(up, normal_unit)
    lx_norm = np.linalg.norm(local_x)
    if lx_norm < 1e-6:
        # Normal is nearly vertical - use X axis as fallback
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

    # Placement at centroid (Manhattan -> IFC: x->x, z->y, y->z)
    cx = float(centroid[0] / scale)
    cy = float(centroid[2] / scale)
    cz = float(centroid[1] / scale)

    point = ifc.createIfcCartesianPoint([cx, cy, cz])
    axis_dir = ifc.createIfcDirection([
        float(normal_unit[0]), float(normal_unit[2]), float(normal_unit[1]),
    ])
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
