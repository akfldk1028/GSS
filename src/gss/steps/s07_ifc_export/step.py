"""Step 07: IFC/BIM export from center-line wall data (Cloud2BIM pattern).

Primary path: walls.json + spaces.json → center-line based IFC
Legacy fallback: planes.json + boundaries.json → boundary_3d based IFC

Cloud2BIM pattern:
  - Walls are extruded from storey floor_height to ceiling_height (uniform).
  - Synthetic walls overlapping detected walls are deduplicated.
"""

from __future__ import annotations

import json
import logging
from typing import ClassVar

import numpy as np

from gss.core.step_base import BaseStep
from .config import IfcExportConfig
from .contracts import IfcExportInput, IfcExportOutput

logger = logging.getLogger(__name__)


def _deduplicate_walls(walls: list[dict]) -> list[dict]:
    """Remove synthetic walls that overlap with detected walls.

    A synthetic wall is considered duplicate if its center-line midpoint
    is within 0.5 units of a detected wall's center-line midpoint.
    """
    detected = [w for w in walls if not w.get("synthetic")]
    synthetic = [w for w in walls if w.get("synthetic")]

    result = list(detected)
    for sw in synthetic:
        s_cl = sw.get("center_line_2d")
        if not s_cl or len(s_cl) != 2:
            result.append(sw)
            continue
        s0, s1 = np.array(s_cl[0]), np.array(s_cl[1])

        is_dup = False
        for dw in detected:
            d_cl = dw.get("center_line_2d")
            if not d_cl or len(d_cl) != 2:
                continue
            d0, d1 = np.array(d_cl[0]), np.array(d_cl[1])
            fwd = np.linalg.norm(s0 - d0) + np.linalg.norm(s1 - d1)
            rev = np.linalg.norm(s0 - d1) + np.linalg.norm(s1 - d0)
            if min(fwd, rev) < 0.5:
                is_dup = True
                break
        if is_dup:
            logger.info(f"Dedup: synthetic wall {sw.get('id')} overlaps detected wall")
        else:
            result.append(sw)
    return result


class IfcExportStep(BaseStep[IfcExportInput, IfcExportOutput, IfcExportConfig]):
    name: ClassVar[str] = "ifc_export"
    input_type: ClassVar = IfcExportInput
    output_type: ClassVar = IfcExportOutput
    config_type: ClassVar = IfcExportConfig

    def validate_inputs(self, inputs: IfcExportInput) -> bool:
        if not inputs.walls_file.exists():
            logger.error(f"walls_file not found: {inputs.walls_file}")
            return False
        if inputs.spaces_file and not inputs.spaces_file.exists():
            logger.warning(f"spaces_file specified but not found: {inputs.spaces_file}")
        return True

    def run(self, inputs: IfcExportInput) -> IfcExportOutput:
        from ._ifc_builder import (
            create_ifc_file, assign_to_storey,
            assign_products_to_storeys, get_storey_for_height,
        )
        from ._wall_builder import create_wall_from_centerline
        from ._slab_builder import create_floor_slab, create_ceiling_slab
        from ._space_builder import create_space
        from ._opening_builder import create_openings_for_wall

        output_dir = self.data_root / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Load data ---
        with open(inputs.walls_file, encoding="utf-8") as f:
            walls_data: list[dict] = json.load(f)
        logger.info(f"Loaded {len(walls_data)} walls from {inputs.walls_file}")

        spaces_data: list[dict] = []
        coordinate_scale = 1.0
        storey_defs: list[dict] = []
        building_footprint: list[list[float]] | None = None
        if inputs.spaces_file and inputs.spaces_file.exists():
            with open(inputs.spaces_file, encoding="utf-8") as f:
                spaces_raw = json.load(f)
            spaces_data = spaces_raw.get("spaces", [])
            coordinate_scale = spaces_raw.get("coordinate_scale", 1.0)
            storey_defs = spaces_raw.get("storeys", [])
            building_footprint = spaces_raw.get("building_footprint")
            logger.info(
                f"Loaded {len(spaces_data)} spaces, "
                f"coordinate_scale={coordinate_scale:.4f}, "
                f"{len(storey_defs)} storeys"
            )

        # Apply scale override if configured
        scale = self.config.scale_override or coordinate_scale
        multi_storey = len(storey_defs) > 1

        # --- Deduplicate synthetic walls overlapping detected walls ---
        walls_data = _deduplicate_walls(walls_data)
        logger.info(f"After dedup: {len(walls_data)} walls")

        # --- Extract storey heights ---
        floor_z: float | None = None
        ceiling_z: float | None = None
        if storey_defs:
            # Use storey definitions for per-storey heights
            floor_z = storey_defs[0]["floor_height"] / scale
            ceiling_z = storey_defs[-1]["ceiling_height"] / scale
            logger.info(
                f"Multi-storey: {len(storey_defs)} storeys, "
                f"overall floor_z={floor_z:.3f}m, ceiling_z={ceiling_z:.3f}m"
            )
        elif spaces_data:
            floor_heights = [s["floor_height"] for s in spaces_data if "floor_height" in s]
            ceiling_heights = [s["ceiling_height"] for s in spaces_data if "ceiling_height" in s]
            if floor_heights:
                floor_z = min(floor_heights) / scale
            if ceiling_heights:
                ceiling_z = max(ceiling_heights) / scale
            logger.info(
                f"Storey heights: floor_z={floor_z:.3f}m, ceiling_z={ceiling_z:.3f}m"
            )

        # --- Create IFC file + hierarchy ---
        ctx = create_ifc_file(
            version=self.config.ifc_version,
            project_name=self.config.project_name,
            building_name=self.config.building_name,
            storey_name=self.config.storey_name,
            author_name=self.config.author_name,
            organization_name=self.config.organization_name,
            storeys=storey_defs if multi_storey else None,
        )

        num_walls = 0
        num_slabs = 0
        num_spaces = 0
        num_openings = 0
        wall_products_with_height: list[tuple] = []
        wall_pairs: list[tuple] = []  # (ifc_wall, wall_data) for opening creation

        # --- Walls (center-line based) ---
        for wall_data in walls_data:
            # Filter synthetic walls if configured
            if wall_data.get("synthetic", False) and not self.config.include_synthetic_walls:
                logger.debug(f"Skipping synthetic wall {wall_data.get('id')}")
                continue

            # For multi-storey, use per-storey heights for the wall
            w_floor_z = floor_z
            w_ceiling_z = ceiling_z
            if multi_storey and storey_defs:
                hr = wall_data.get("height_range", [0, 0])
                wall_mid_h = (hr[0] + hr[1]) / 2.0
                for sdef in storey_defs:
                    sf = sdef["floor_height"]
                    sc = sdef["ceiling_height"]
                    if sf <= wall_mid_h <= sc:
                        w_floor_z = sf / scale
                        w_ceiling_z = sc / scale
                        break

            wall = create_wall_from_centerline(
                ctx,
                wall_data,
                scale,
                floor_z=w_floor_z,
                ceiling_z=w_ceiling_z,
                wall_material_name=self.config.wall_material_name,
                default_thickness=self.config.default_wall_thickness,
                create_axis=self.config.create_axis_representation,
                create_material_layers=self.config.create_material_layers,
                create_wall_type=self.config.create_wall_types,
                create_property_set=self.config.create_property_sets,
            )
            if wall is not None:
                hr = wall_data.get("height_range", [0, 0])
                wall_mid_h = (hr[0] + hr[1]) / 2.0
                wall_products_with_height.append((wall, wall_mid_h))
                wall_pairs.append((wall, wall_data))
                num_walls += 1

        # Assign walls to appropriate storeys
        if multi_storey and storey_defs:
            assign_products_to_storeys(
                ctx, wall_products_with_height, storey_defs, scale,
            )
        else:
            assign_to_storey(ctx, [p for p, _ in wall_products_with_height])

        # --- Openings (doors/windows from s06b opening detection) ---
        for ifc_wall, wall_data in wall_pairs:
            if wall_data.get("openings"):
                w_floor_z = floor_z
                if multi_storey and storey_defs:
                    hr = wall_data.get("height_range", [0, 0])
                    wall_mid_h = (hr[0] + hr[1]) / 2.0
                    for sdef in storey_defs:
                        if sdef["floor_height"] <= wall_mid_h <= sdef["ceiling_height"]:
                            w_floor_z = sdef["floor_height"] / scale
                            break
                n = create_openings_for_wall(
                    ctx, ifc_wall, wall_data, scale, floor_z=w_floor_z,
                )
                num_openings += n

        # --- Slabs (from spaces boundary_2d) ---
        if self.config.create_slabs and spaces_data:
            slab_products_with_height: list[tuple] = []
            for space in spaces_data:
                floor_slab = create_floor_slab(
                    ctx,
                    space,
                    scale,
                    thickness=self.config.default_slab_thickness,
                    material_name=self.config.slab_material_name,
                )
                if floor_slab is not None:
                    slab_products_with_height.append(
                        (floor_slab, space.get("floor_height", 0.0))
                    )
                    num_slabs += 1

                ceiling_slab = create_ceiling_slab(
                    ctx,
                    space,
                    scale,
                    thickness=self.config.default_slab_thickness,
                    material_name=self.config.slab_material_name,
                )
                if ceiling_slab is not None:
                    slab_products_with_height.append(
                        (ceiling_slab, space.get("ceiling_height", 0.0))
                    )
                    num_slabs += 1

            if multi_storey and storey_defs:
                assign_products_to_storeys(
                    ctx, slab_products_with_height, storey_defs, scale,
                )
            else:
                assign_to_storey(ctx, [p for p, _ in slab_products_with_height])

        # --- Spaces ---
        if self.config.create_spaces and spaces_data:
            for space in spaces_data:
                # For multi-storey, assign space to correct storey
                target_storey = ctx.storey
                if multi_storey and storey_defs:
                    space_mid_h = (
                        space.get("floor_height", 0) + space.get("ceiling_height", 0)
                    ) / 2.0
                    target_storey = get_storey_for_height(
                        ctx, space_mid_h, storey_defs, scale,
                    )
                ifc_space = create_space(ctx, space, scale, storey=target_storey)
                if ifc_space is not None:
                    num_spaces += 1

        # --- Roof (from planes.json with label="roof") ---
        num_roof_slabs = 0
        if self.config.create_roof and inputs.planes_file and inputs.planes_file.exists():
            with open(inputs.planes_file, encoding="utf-8") as f:
                all_planes = json.load(f)
            roof_planes = [p for p in all_planes if p.get("label") == "roof"]
            if roof_planes:
                from ._roof_builder import create_roof
                _, num_roof_slabs = create_roof(
                    ctx, roof_planes, scale,
                    material_name=self.config.slab_material_name,
                    slab_thickness=self.config.default_slab_thickness,
                )

        # --- Site footprint ---
        if building_footprint:
            from ._ifc_builder import set_site_footprint
            set_site_footprint(ctx, building_footprint, scale)

        # --- Write IFC file ---
        ifc_path = output_dir / f"{self.config.project_name}.ifc"
        ctx.ifc.write(str(ifc_path))

        logger.info(
            f"IFC exported: {num_walls} walls, {num_slabs} slabs, "
            f"{num_spaces} spaces, {num_openings} openings"
            f"{f', {len(storey_defs)} storeys' if multi_storey else ''}"
            f" → {ifc_path}"
        )

        return IfcExportOutput(
            ifc_path=ifc_path,
            num_walls=num_walls,
            num_slabs=num_slabs,
            num_spaces=num_spaces,
            num_openings=num_openings,
            coordinate_scale=scale,
            ifc_version=self.config.ifc_version,
        )
