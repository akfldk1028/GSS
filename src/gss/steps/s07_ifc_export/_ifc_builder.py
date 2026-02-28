"""IFC file hierarchy builder — Project/Site/Building/Storey + contexts."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid
import ifcopenshell.util.date

logger = logging.getLogger(__name__)


@dataclass
class IfcContext:
    """Holds references to core IFC entities shared by all builders."""

    ifc: Any  # ifcopenshell file
    project: Any = None
    site: Any = None
    building: Any = None
    storey: Any = None  # default (ground) storey — backward-compatible
    owner_history: Any = None
    body_context: Any = None
    axis_context: Any = None

    # Multi-storey: {storey_name: IfcBuildingStorey}
    storeys: dict[str, Any] = field(default_factory=dict)

    # Shared caches to avoid duplicates
    _wall_type_cache: dict[float, Any] = field(default_factory=dict)
    _material_layer_set_cache: dict[float, Any] = field(default_factory=dict)


def create_ifc_file(
    *,
    version: str = "IFC4",
    project_name: str = "GSS_BIM",
    building_name: str = "Building",
    storey_name: str = "Ground Floor",
    author_name: str = "GSS",
    organization_name: str = "GSS Pipeline",
    storeys: list[dict] | None = None,
) -> IfcContext:
    """Create a fully initialized IFC file with spatial hierarchy.

    Args:
        storeys: Optional list of storey dicts from _snap_heights._group_storeys().
            Each dict: {name, floor_height, ceiling_height, elevation}.
            If provided, creates multiple IfcBuildingStorey with elevations.
            If None, creates a single storey (backward-compatible).

    Returns an IfcContext holding references to all shared entities.
    """
    ifc = ifcopenshell.api.run("project.create_file", version=version)

    # --- Owner History ---
    org = ifc.createIfcOrganization(Name=organization_name)
    app = ifc.createIfcApplication(
        ApplicationDeveloper=org,
        Version="1.0",
        ApplicationFullName="GSS Pipeline",
        ApplicationIdentifier="GSS",
    )
    person = ifc.createIfcPerson(GivenName=author_name)
    person_org = ifc.createIfcPersonAndOrganization(ThePerson=person, TheOrganization=org)
    owner_history = ifc.createIfcOwnerHistory(
        OwningUser=person_org,
        OwningApplication=app,
        ChangeAction="NOCHANGE",
        CreationDate=int(
            ifcopenshell.util.date.datetime2ifc(datetime.datetime.now(), "IfcTimeStamp")
        ),
    )

    # --- Project ---
    project = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcProject", name=project_name
    )
    project.OwnerHistory = owner_history

    # --- Units (meters) ---
    ifcopenshell.api.run("unit.assign_unit", ifc, length={"is_metric": True, "raw": "METRE"})

    # --- Geometric Representation Contexts ---
    model_ctx = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
    body_ctx = ifcopenshell.api.run(
        "context.add_context",
        ifc,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model_ctx,
    )
    axis_ctx = ifcopenshell.api.run(
        "context.add_context",
        ifc,
        context_type="Model",
        context_identifier="Axis",
        target_view="GRAPH_VIEW",
        parent=model_ctx,
    )

    # --- Spatial hierarchy: Project > Site > Building > Storey(s) ---
    site = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Site")
    site.OwnerHistory = owner_history
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    building = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcBuilding", name=building_name
    )
    building.OwnerHistory = owner_history
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[building], relating_object=site)

    # Create storey(s)
    storeys_dict: dict[str, Any] = {}

    if storeys and len(storeys) > 0:
        # Multi-storey mode
        for storey_def in storeys:
            s_name = storey_def.get("name", "Ground Floor")
            s_elevation = storey_def.get("elevation", 0.0)

            ifc_storey = ifcopenshell.api.run(
                "root.create_entity", ifc,
                ifc_class="IfcBuildingStorey", name=s_name,
            )
            ifc_storey.OwnerHistory = owner_history
            ifc_storey.Elevation = float(s_elevation)
            ifcopenshell.api.run(
                "aggregate.assign_object", ifc,
                products=[ifc_storey], relating_object=building,
            )
            storeys_dict[s_name] = ifc_storey
            logger.info(
                f"Created IfcBuildingStorey '{s_name}' at elevation={s_elevation:.3f}"
            )

        # Default storey = first (lowest elevation)
        default_storey = storeys_dict[storeys[0]["name"]]
    else:
        # Single storey (backward-compatible)
        default_storey = ifcopenshell.api.run(
            "root.create_entity", ifc,
            ifc_class="IfcBuildingStorey", name=storey_name,
        )
        default_storey.OwnerHistory = owner_history
        ifcopenshell.api.run(
            "aggregate.assign_object", ifc,
            products=[default_storey], relating_object=building,
        )
        storeys_dict[storey_name] = default_storey

    return IfcContext(
        ifc=ifc,
        project=project,
        site=site,
        building=building,
        storey=default_storey,
        owner_history=owner_history,
        body_context=body_ctx,
        axis_context=axis_ctx,
        storeys=storeys_dict,
    )


def set_site_footprint(
    ctx: IfcContext,
    footprint_pts: list[list[float]],
    scale: float = 1.0,
) -> None:
    """Set the building footprint as IfcSite representation.

    Creates a 2D polygon on the IfcSite to represent the building footprint.

    Args:
        ctx: IFC context.
        footprint_pts: List of [x, z] points in Manhattan coordinates.
        scale: Coordinate scale divisor.
    """
    if not footprint_pts or len(footprint_pts) < 3:
        return

    ifc = ctx.ifc

    # Convert to IFC XY coordinates (scaled)
    pts_2d = []
    for pt in footprint_pts:
        pts_2d.append([float(pt[0] / scale), float(pt[1] / scale)])

    # Close polygon if needed
    if pts_2d[0] != pts_2d[-1]:
        pts_2d.append(pts_2d[0])

    if len(pts_2d) < 4:
        return

    # Create polyline profile
    profile_pts = [ifc.createIfcCartesianPoint(p) for p in pts_2d]
    polyline = ifc.createIfcPolyline(profile_pts)
    profile = ifc.createIfcArbitraryClosedProfileDef(
        "AREA", "Site Footprint", polyline,
    )

    # Thin extrusion (0.01m) to make it visible
    ext_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
    solid = ifc.createIfcExtrudedAreaSolid(profile, None, ext_dir, 0.01)

    body_repr = ifc.createIfcShapeRepresentation(
        ctx.body_context, "Body", "SweptSolid", [solid],
    )
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [body_repr])

    # Assign representation to site
    ctx.site.Representation = product_shape

    # Place site at ground level
    origin = ifc.createIfcCartesianPoint([0.0, 0.0, 0.0])
    placement = ifc.createIfcAxis2Placement3D(origin)
    ctx.site.ObjectPlacement = ifc.createIfcLocalPlacement(None, placement)

    logger.info(f"Set building footprint on IfcSite ({len(footprint_pts)} vertices)")


def assign_to_storey(ctx: IfcContext, products: list) -> None:
    """Assign products to the default building storey via spatial containment."""
    if not products:
        return
    ifcopenshell.api.run(
        "spatial.assign_container",
        ctx.ifc,
        products=products,
        relating_structure=ctx.storey,
    )


def get_storey_for_height(
    ctx: IfcContext,
    height: float,
    storey_defs: list[dict],
    scale: float = 1.0,
) -> Any:
    """Find the appropriate IfcBuildingStorey for a given height.

    Matches height to the storey whose floor_height ≤ height < ceiling_height.
    Falls back to the nearest storey if no exact match.

    Args:
        ctx: IFC context with storeys dict.
        height: Height in scene units (before scale division).
        storey_defs: Storey definitions from _snap_heights (with floor_height, ceiling_height).
        scale: Coordinate scale divisor.

    Returns:
        The matching IfcBuildingStorey entity.
    """
    if not storey_defs or not ctx.storeys:
        return ctx.storey

    h_scaled = height / scale

    # Find storey where floor ≤ h < ceiling
    best_storey_name = None
    best_dist = float("inf")

    for sdef in storey_defs:
        s_floor = sdef["floor_height"] / scale
        s_ceil = sdef["ceiling_height"] / scale
        s_name = sdef["name"]

        if s_floor <= h_scaled <= s_ceil:
            return ctx.storeys.get(s_name, ctx.storey)

        # Track nearest for fallback
        dist = min(abs(h_scaled - s_floor), abs(h_scaled - s_ceil))
        if dist < best_dist:
            best_dist = dist
            best_storey_name = s_name

    if best_storey_name and best_storey_name in ctx.storeys:
        return ctx.storeys[best_storey_name]

    return ctx.storey


def assign_products_to_storeys(
    ctx: IfcContext,
    products_with_height: list[tuple[Any, float]],
    storey_defs: list[dict],
    scale: float = 1.0,
) -> None:
    """Assign products to appropriate storeys based on their heights.

    Groups products by storey, then batch-assigns each group.

    Args:
        ctx: IFC context.
        products_with_height: List of (ifc_product, height_in_scene_units).
        storey_defs: Storey definitions from _snap_heights.
        scale: Coordinate scale divisor.
    """
    if not products_with_height:
        return

    # Group by storey
    storey_products: dict[int, list] = {}  # storey entity id → products
    for product, height in products_with_height:
        storey = get_storey_for_height(ctx, height, storey_defs, scale)
        sid = storey.id()
        if sid not in storey_products:
            storey_products[sid] = []
        storey_products[sid].append((product, storey))

    # Batch assign per storey
    for group in storey_products.values():
        target_storey = group[0][1]
        products = [p for p, _ in group]
        ifcopenshell.api.run(
            "spatial.assign_container",
            ctx.ifc,
            products=products,
            relating_structure=target_storey,
        )
