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
import math
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
        from ._ifc_builder import create_ifc_file, assign_to_storey
        from ._wall_builder import create_wall_from_centerline
        from ._slab_builder import create_floor_slab, create_ceiling_slab
        from ._space_builder import create_space

        output_dir = self.data_root / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Load data ---
        with open(inputs.walls_file, encoding="utf-8") as f:
            walls_data: list[dict] = json.load(f)
        logger.info(f"Loaded {len(walls_data)} walls from {inputs.walls_file}")

        spaces_data: list[dict] = []
        coordinate_scale = 1.0
        if inputs.spaces_file and inputs.spaces_file.exists():
            with open(inputs.spaces_file, encoding="utf-8") as f:
                spaces_raw = json.load(f)
            spaces_data = spaces_raw.get("spaces", [])
            coordinate_scale = spaces_raw.get("coordinate_scale", 1.0)
            logger.info(
                f"Loaded {len(spaces_data)} spaces, "
                f"coordinate_scale={coordinate_scale:.4f}"
            )

        # Apply scale override if configured
        scale = self.config.scale_override or coordinate_scale

        # --- Deduplicate synthetic walls overlapping detected walls ---
        walls_data = _deduplicate_walls(walls_data)
        logger.info(f"After dedup: {len(walls_data)} walls")

        # --- Extract uniform storey heights from spaces ---
        floor_z: float | None = None
        ceiling_z: float | None = None
        if spaces_data:
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
        )

        num_walls = 0
        num_slabs = 0
        num_spaces = 0
        wall_products = []

        # --- Walls (center-line based) ---
        for wall_data in walls_data:
            # Filter synthetic walls if configured
            if wall_data.get("synthetic", False) and not self.config.include_synthetic_walls:
                logger.debug(f"Skipping synthetic wall {wall_data.get('id')}")
                continue

            wall = create_wall_from_centerline(
                ctx,
                wall_data,
                scale,
                floor_z=floor_z,
                ceiling_z=ceiling_z,
                wall_material_name=self.config.wall_material_name,
                default_thickness=self.config.default_wall_thickness,
                create_axis=self.config.create_axis_representation,
                create_material_layers=self.config.create_material_layers,
                create_wall_type=self.config.create_wall_types,
                create_property_set=self.config.create_property_sets,
            )
            if wall is not None:
                wall_products.append(wall)
                num_walls += 1

        # Assign all walls to storey
        assign_to_storey(ctx, wall_products)

        # --- Slabs (from spaces boundary_2d) ---
        if self.config.create_slabs and spaces_data:
            slab_products = []
            for space in spaces_data:
                floor_slab = create_floor_slab(
                    ctx,
                    space,
                    scale,
                    thickness=self.config.default_slab_thickness,
                    material_name=self.config.slab_material_name,
                )
                if floor_slab is not None:
                    slab_products.append(floor_slab)
                    num_slabs += 1

                ceiling_slab = create_ceiling_slab(
                    ctx,
                    space,
                    scale,
                    thickness=self.config.default_slab_thickness,
                    material_name=self.config.slab_material_name,
                )
                if ceiling_slab is not None:
                    slab_products.append(ceiling_slab)
                    num_slabs += 1

            assign_to_storey(ctx, slab_products)

        # --- Spaces ---
        if self.config.create_spaces and spaces_data:
            for space in spaces_data:
                ifc_space = create_space(ctx, space, scale)
                if ifc_space is not None:
                    num_spaces += 1

        # --- Write IFC file ---
        ifc_path = output_dir / f"{self.config.project_name}.ifc"
        ctx.ifc.write(str(ifc_path))

        logger.info(
            f"IFC exported: {num_walls} walls, {num_slabs} slabs, "
            f"{num_spaces} spaces → {ifc_path}"
        )

        return IfcExportOutput(
            ifc_path=ifc_path,
            num_walls=num_walls,
            num_slabs=num_slabs,
            num_spaces=num_spaces,
            coordinate_scale=scale,
            ifc_version=self.config.ifc_version,
        )
