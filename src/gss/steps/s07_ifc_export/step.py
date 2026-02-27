"""Step 07: IFC/BIM file generation from planes and boundaries."""

from __future__ import annotations

import json
import logging
from typing import ClassVar

import numpy as np

from gss.core.step_base import BaseStep
from .config import IfcExportConfig
from .contracts import IfcExportInput, IfcExportOutput

logger = logging.getLogger(__name__)


def _to_floats(v) -> list[float]:
    """Ensure value is a list of native Python floats (IfcOpenShell requirement)."""
    return [float(x) for x in v]


def _create_ifc_axis2placement3d(ifc, point=(0.0, 0.0, 0.0), z_dir=(0.0, 0.0, 1.0), x_dir=(1.0, 0.0, 0.0)):
    """Create an IfcAxis2Placement3D."""
    loc = ifc.createIfcCartesianPoint(_to_floats(point))
    z = ifc.createIfcDirection(_to_floats(z_dir))
    x = ifc.createIfcDirection(_to_floats(x_dir))
    return ifc.createIfcAxis2Placement3D(loc, z, x)


def _create_extruded_wall(
    ifc,
    context,
    boundary_3d: list[list[float]],
    normal: list[float],
    thickness: float,
    placement,
):
    """Create an IfcWall with extruded area solid from boundary polygon."""
    if len(boundary_3d) < 3:
        return None

    pts = np.array(boundary_3d)

    # Project boundary to 2D on the plane
    n = np.array(normal)
    n /= np.linalg.norm(n)

    # Build local coordinate system for the wall
    # For walls, normal is roughly horizontal -> extrude vertically
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, up)) > 0.9:
        # This is a floor/ceiling, not a wall
        return None

    # Wall local axes: x along wall, z up
    wall_x = np.cross(up, n)
    wall_x /= np.linalg.norm(wall_x)
    wall_z = up

    centroid = pts.mean(axis=0)
    local = pts - centroid

    # Project to wall-local 2D (x along wall, z up)
    coords_x = local @ wall_x
    coords_z = local @ wall_z

    z_min, z_max = coords_z.min(), coords_z.max()
    x_min, x_max = coords_x.min(), coords_x.max()
    height = z_max - z_min
    width = x_max - x_min

    if height < 0.01 or width < 0.01:
        return None

    # Create a rectangular profile (simplified wall)
    profile_pts = [
        ifc.createIfcCartesianPoint([0.0, 0.0]),
        ifc.createIfcCartesianPoint([float(width), 0.0]),
        ifc.createIfcCartesianPoint([float(width), float(thickness)]),
        ifc.createIfcCartesianPoint([0.0, float(thickness)]),
        ifc.createIfcCartesianPoint([0.0, 0.0]),
    ]
    polyline = ifc.createIfcPolyline(profile_pts)
    profile = ifc.createIfcArbitraryClosedProfileDef("AREA", None, polyline)

    # Extrusion direction (up)
    direction = ifc.createIfcDirection([0.0, 0.0, 1.0])

    # Wall placement: origin at bottom-left of wall
    origin = centroid + x_min * wall_x + z_min * wall_z
    wall_placement = _create_ifc_axis2placement3d(
        ifc,
        point=tuple(origin.tolist()),
        z_dir=tuple(wall_z.tolist()),
        x_dir=tuple(wall_x.tolist()),
    )

    solid = ifc.createIfcExtrudedAreaSolid(
        profile,
        wall_placement,
        direction,
        float(height),
    )

    shape = ifc.createIfcShapeRepresentation(context, "Body", "SweptSolid", [solid])
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [shape])
    return product_shape


def _create_extruded_slab(
    ifc,
    context,
    boundary_3d: list[list[float]],
    normal: list[float],
    thickness: float,
    placement,
):
    """Create an IfcSlab with extruded area solid from boundary polygon."""
    if len(boundary_3d) < 3:
        return None

    pts = np.array(boundary_3d)

    # For slabs (floor/ceiling), project to XY plane
    centroid = pts.mean(axis=0)
    local = pts - centroid

    # Use XY projection
    coords_2d = local[:, :2]
    z_level = centroid[2]

    # Create profile from projected boundary
    profile_pts = [ifc.createIfcCartesianPoint(_to_floats(pt)) for pt in coords_2d]
    if profile_pts:
        profile_pts.append(profile_pts[0])

    polyline = ifc.createIfcPolyline(profile_pts)
    profile = ifc.createIfcArbitraryClosedProfileDef("AREA", None, polyline)

    # Extrusion direction (up for floors, down for ceilings)
    n = np.array(normal)
    ext_dir = [0.0, 0.0, 1.0] if n[2] > 0 else [0.0, 0.0, -1.0]
    direction = ifc.createIfcDirection(ext_dir)

    slab_placement = _create_ifc_axis2placement3d(
        ifc,
        point=[float(centroid[0]), float(centroid[1]), float(z_level)],
    )

    solid = ifc.createIfcExtrudedAreaSolid(
        profile,
        slab_placement,
        direction,
        float(thickness),
    )

    shape = ifc.createIfcShapeRepresentation(context, "Body", "SweptSolid", [solid])
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [shape])
    return product_shape


class IfcExportStep(BaseStep[IfcExportInput, IfcExportOutput, IfcExportConfig]):
    name: ClassVar[str] = "ifc_export"
    input_type: ClassVar = IfcExportInput
    output_type: ClassVar = IfcExportOutput
    config_type: ClassVar = IfcExportConfig

    def validate_inputs(self, inputs: IfcExportInput) -> bool:
        if not (inputs.planes_file.exists() and inputs.boundaries_file.exists()):
            return False
        if inputs.walls_file and not inputs.walls_file.exists():
            logger.warning(f"walls_file specified but not found: {inputs.walls_file}")
        if inputs.spaces_file and not inputs.spaces_file.exists():
            logger.warning(f"spaces_file specified but not found: {inputs.spaces_file}")
        return True

    def run(self, inputs: IfcExportInput) -> IfcExportOutput:
        import ifcopenshell
        import ifcopenshell.api

        output_dir = self.data_root / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load plane data
        with open(inputs.planes_file) as f:
            planes = json.load(f)
        with open(inputs.boundaries_file) as f:
            boundaries = json.load(f)

        # Load optional walls.json for per-wall thickness
        walls_data: list[dict] = []
        if inputs.walls_file and inputs.walls_file.exists():
            with open(inputs.walls_file) as f:
                walls_data = json.load(f)
            logger.info(f"Loaded {len(walls_data)} walls from {inputs.walls_file}")

        # Load optional spaces.json
        spaces_data: list[dict] = []
        if inputs.spaces_file and inputs.spaces_file.exists():
            with open(inputs.spaces_file) as f:
                spaces_raw = json.load(f)
            spaces_data = spaces_raw.get("spaces", [])
            logger.info(f"Loaded {len(spaces_data)} spaces from {inputs.spaces_file}")

        # Build wall thickness lookup: plane_id → thickness
        wall_thickness_map: dict[int, float] = {}
        for w in walls_data:
            thickness = w.get("thickness", self.config.default_wall_thickness)
            for pid in w.get("plane_ids", []):
                wall_thickness_map[pid] = thickness

        # Build boundary lookup
        boundary_map = {b["id"]: b["boundary_3d"] for b in boundaries}

        # Create IFC file
        ifc = ifcopenshell.api.run("project.create_file", version=self.config.ifc_version)

        # Project
        project = ifcopenshell.api.run(
            "root.create_entity", ifc, ifc_class="IfcProject", name=self.config.project_name
        )

        # Unit assignment (meters)
        ifcopenshell.api.run("unit.assign_unit", ifc, length={"is_metric": True, "raw": "METRE"})

        # Context
        context = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
        body = ifcopenshell.api.run(
            "context.add_context",
            ifc,
            context_type="Model",
            context_identifier="Body",
            target_view="MODEL_VIEW",
            parent=context,
        )

        # Site → Building → Storey
        site = ifcopenshell.api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Site")
        ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

        building = ifcopenshell.api.run(
            "root.create_entity", ifc, ifc_class="IfcBuilding", name=self.config.building_name
        )
        ifcopenshell.api.run("aggregate.assign_object", ifc, products=[building], relating_object=site)

        storey = ifcopenshell.api.run(
            "root.create_entity", ifc, ifc_class="IfcBuildingStorey", name="Ground Floor"
        )
        ifcopenshell.api.run("aggregate.assign_object", ifc, products=[storey], relating_object=building)

        num_walls = 0
        num_slabs = 0
        num_spaces = 0

        for plane in planes:
            plane_id = plane["id"]
            label = plane["label"]
            normal = plane["normal"]
            boundary = boundary_map.get(plane_id, [])

            if len(boundary) < 3:
                logger.warning(f"Plane {plane_id} ({label}): insufficient boundary points, skipping")
                continue

            if label == "wall":
                # Use per-wall thickness from walls.json if available
                thickness = wall_thickness_map.get(plane_id, self.config.default_wall_thickness)

                wall = ifcopenshell.api.run(
                    "root.create_entity", ifc, ifc_class="IfcWall", name=f"Wall_{plane_id}"
                )
                product_shape = _create_extruded_wall(
                    ifc,
                    body,
                    boundary,
                    normal,
                    thickness,
                    None,
                )
                if product_shape:
                    wall.Representation = product_shape
                ifcopenshell.api.run(
                    "spatial.assign_container", ifc, products=[wall], relating_structure=storey
                )
                num_walls += 1

            elif label in ("floor", "ceiling"):
                slab = ifcopenshell.api.run(
                    "root.create_entity", ifc, ifc_class="IfcSlab", name=f"Slab_{plane_id}"
                )
                product_shape = _create_extruded_slab(
                    ifc,
                    body,
                    boundary,
                    normal,
                    self.config.default_slab_thickness,
                    None,
                )
                if product_shape:
                    slab.Representation = product_shape
                ifcopenshell.api.run(
                    "spatial.assign_container", ifc, products=[slab], relating_structure=storey
                )
                num_slabs += 1

        # Create IfcSpace objects from spaces.json
        for space in spaces_data:
            space_boundary = space.get("boundary_2d", [])
            if len(space_boundary) < 3:
                continue

            floor_h = space.get("floor_height", 0.0)
            ceiling_h = space.get("ceiling_height", 3.0)
            height = ceiling_h - floor_h
            if height <= 0:
                continue

            ifc_space = ifcopenshell.api.run(
                "root.create_entity", ifc, ifc_class="IfcSpace",
                name=f"Room_{space['id']}",
            )

            # Create space geometry: extrude 2D boundary upward
            # Ensure boundary is closed (first == last coordinate)
            boundary_closed = list(space_boundary)
            if boundary_closed and boundary_closed[0] != boundary_closed[-1]:
                boundary_closed.append(boundary_closed[0])
            profile_pts = [
                ifc.createIfcCartesianPoint([float(c[0]), float(c[1])])
                for c in boundary_closed
            ]

            if len(profile_pts) >= 4:
                polyline = ifc.createIfcPolyline(profile_pts)
                profile = ifc.createIfcArbitraryClosedProfileDef("AREA", None, polyline)
                direction = ifc.createIfcDirection([0.0, 0.0, 1.0])
                space_placement = _create_ifc_axis2placement3d(
                    ifc, point=(0.0, 0.0, float(floor_h)),
                )
                solid = ifc.createIfcExtrudedAreaSolid(
                    profile, space_placement, direction, float(height),
                )
                shape = ifc.createIfcShapeRepresentation(body, "Body", "SweptSolid", [solid])
                product_shape = ifc.createIfcProductDefinitionShape(None, None, [shape])
                ifc_space.Representation = product_shape

            ifcopenshell.api.run(
                "aggregate.assign_object", ifc, products=[ifc_space], relating_object=storey
            )
            num_spaces += 1

        # Write IFC file
        ifc_path = output_dir / f"{self.config.project_name}.ifc"
        ifc.write(str(ifc_path))

        logger.info(
            f"IFC exported: {num_walls} walls, {num_slabs} slabs, "
            f"{num_spaces} spaces -> {ifc_path}"
        )

        return IfcExportOutput(
            ifc_path=ifc_path,
            num_walls=num_walls,
            num_slabs=num_slabs,
            num_spaces=num_spaces,
            ifc_version=self.config.ifc_version,
        )
