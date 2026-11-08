"""NuRec scene manifest generator for Isaac Sim integration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_nurec_manifest(
    usdz_path: Path,
    model_path: Path,
    num_gaussians: int,
    output_path: Path,
    up_axis: str = "Y",
    meters_per_unit: float = 1.0,
) -> None:
    """Write scene_manifest_nurec.json for Isaac Sim NuRec scene loading.

    This manifest is consumed by soarm_stack's scene_loader to set up
    the simulation environment with NuRec volume rendering.
    """
    manifest = {
        "format": "nurec",
        "version": "1.0",
        "usdz_path": str(usdz_path),
        "source_ply": str(model_path),
        "num_gaussians": num_gaussians,
        "stage": {
            "up_axis": up_axis,
            "meters_per_unit": meters_per_unit,
        },
        "rendering": {
            "mode": "volume",
            "engine": "nurec",
            "requires_isaac_sim": "5.1.0",
        },
        "spawn_points": [
            {
                "name": "default",
                "position": [0.0, 0.0, 0.0],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            }
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"NuRec manifest written: {output_path}")
