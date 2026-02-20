"""Step 02: COLMAP Structure-from-Motion for camera pose estimation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import ColmapConfig
from .contracts import ColmapInput, ColmapOutput

logger = logging.getLogger(__name__)


class ColmapStep(BaseStep[ColmapInput, ColmapOutput, ColmapConfig]):
    name: ClassVar[str] = "colmap"
    input_type: ClassVar = ColmapInput
    output_type: ClassVar = ColmapOutput
    config_type: ClassVar = ColmapConfig

    def validate_inputs(self, inputs: ColmapInput) -> bool:
        if not inputs.frames_dir.exists():
            logger.error(f"Frames directory not found: {inputs.frames_dir}")
            return False
        return True

    def run(self, inputs: ColmapInput) -> ColmapOutput:
        # TODO: Implement COLMAP execution via pycolmap or subprocess
        # MVP: pycolmap.extract_features() + pycolmap.match_sequential() + pycolmap.incremental_mapping()
        # Future: hloc (SuperPoint + LightGlue) frontend
        output_dir = self.data_root / "interim" / "s02_colmap"
        output_dir.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError("COLMAP step not yet implemented")
