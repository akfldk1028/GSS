"""IFC file hierarchy builder — Project/Site/Building/Storey + contexts."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid
import ifcopenshell.util.date


@dataclass
class IfcContext:
    """Holds references to core IFC entities shared by all builders."""

    ifc: Any  # ifcopenshell file
    project: Any = None
    site: Any = None
    building: Any = None
    storey: Any = None
    owner_history: Any = None
    body_context: Any = None
    axis_context: Any = None

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
) -> IfcContext:
    """Create a fully initialized IFC file with spatial hierarchy.

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

    # --- Spatial hierarchy: Project > Site > Building > Storey ---
    site = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Site")
    site.OwnerHistory = owner_history
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    building = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcBuilding", name=building_name
    )
    building.OwnerHistory = owner_history
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[building], relating_object=site)

    storey = ifcopenshell.api.run(
        "root.create_entity", ifc, ifc_class="IfcBuildingStorey", name=storey_name
    )
    storey.OwnerHistory = owner_history
    ifcopenshell.api.run(
        "aggregate.assign_object", ifc, products=[storey], relating_object=building
    )

    return IfcContext(
        ifc=ifc,
        project=project,
        site=site,
        building=building,
        storey=storey,
        owner_history=owner_history,
        body_context=body_ctx,
        axis_context=axis_ctx,
    )


def assign_to_storey(ctx: IfcContext, products: list) -> None:
    """Assign products to the building storey via spatial containment."""
    if not products:
        return
    ifcopenshell.api.run(
        "spatial.assign_container",
        ctx.ifc,
        products=products,
        relating_structure=ctx.storey,
    )
