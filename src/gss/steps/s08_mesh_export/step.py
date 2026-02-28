"""Step 08: Mesh export — IFC → GLB + USD for digital twin platforms.

Reads the IFC file from s07 and exports to:
- GLB (glTF Binary) — universal 3D format for viewers and game engines
- USDC (Universal Scene Description) — NVIDIA Omniverse / Isaac Sim
- USDZ (optional) — Apple Vision Pro / ARKit
"""

from __future__ import annotations

import logging
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import MeshExportConfig
from .contracts import MeshExportInput, MeshExportOutput

logger = logging.getLogger(__name__)


class MeshExportStep(BaseStep[MeshExportInput, MeshExportOutput, MeshExportConfig]):
    name: ClassVar[str] = "mesh_export"
    input_type: ClassVar = MeshExportInput
    output_type: ClassVar = MeshExportOutput
    config_type: ClassVar = MeshExportConfig

    def validate_inputs(self, inputs: MeshExportInput) -> bool:
        if not inputs.ifc_path.exists():
            logger.error(f"IFC file not found: {inputs.ifc_path}")
            return False
        return True

    def run(self, inputs: MeshExportInput) -> MeshExportOutput:
        from ._ifc_to_mesh import extract_meshes_from_ifc

        output_dir = self.data_root / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Build color map from config ---
        if self.config.color_scheme == "uniform":
            # Use default color for all element types
            uniform = self.config.color_default
            color_map = {
                "wall": uniform, "slab": uniform, "door": uniform,
                "window": uniform, "space": uniform, "default": uniform,
            }
        else:
            color_map = {
                "wall": self.config.color_wall,
                "slab": self.config.color_slab,
                "door": self.config.color_door,
                "window": self.config.color_window,
                "space": self.config.color_space,
                "default": self.config.color_default,
            }

        # --- Extract meshes from IFC ---
        meshes = extract_meshes_from_ifc(
            inputs.ifc_path,
            color_map=color_map,
            include_spaces=self.config.include_spaces,
        )

        if not meshes:
            logger.warning("No meshes extracted from IFC file")
            return MeshExportOutput()

        # Stats
        num_meshes = len(meshes)
        num_vertices = sum(len(m.vertices) for m in meshes)
        num_faces = sum(len(m.faces) for m in meshes)

        logger.info(
            f"Extracted {num_meshes} meshes "
            f"({num_vertices} vertices, {num_faces} faces)"
        )

        # --- Derive output stem from IFC filename ---
        stem = inputs.ifc_path.stem  # e.g., "GSS_BIM"

        glb_path = None
        usd_path = None
        usdz_path = None

        # --- GLB export ---
        if self.config.export_glb:
            from ._glb_writer import _has_trimesh, write_glb

            if _has_trimesh():
                glb_path = output_dir / f"{stem}.glb"
                write_glb(meshes, glb_path)
            else:
                logger.warning(
                    "trimesh not installed — skipping GLB export. "
                    "Install with: pip install trimesh"
                )

        # --- USD export ---
        if self.config.export_usd:
            from ._usd_writer import _has_pxr

            if _has_pxr():
                from ._usd_writer import write_usd

                usd_path = output_dir / f"{stem}.usdc"
                write_usd(
                    meshes,
                    usd_path,
                    up_axis=self.config.usd_up_axis,
                    meters_per_unit=self.config.usd_meters_per_unit,
                )

                # --- USDZ packaging ---
                if self.config.export_usdz:
                    from ._usd_writer import write_usdz

                    usdz_path = output_dir / f"{stem}.usdz"
                    write_usdz(usd_path, usdz_path)
            else:
                logger.warning(
                    "usd-core not installed — skipping USD export. "
                    "Install with: pip install usd-core"
                )

        logger.info(
            f"Mesh export complete: "
            f"GLB={'yes' if glb_path else 'no'}, "
            f"USD={'yes' if usd_path else 'no'}, "
            f"USDZ={'yes' if usdz_path else 'no'}"
        )

        return MeshExportOutput(
            glb_path=glb_path,
            usd_path=usd_path,
            usdz_path=usdz_path,
            num_meshes=num_meshes,
            num_vertices=num_vertices,
            num_faces=num_faces,
        )
