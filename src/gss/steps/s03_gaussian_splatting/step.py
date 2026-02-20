"""Step 03: 3D Gaussian Splatting training (gsplat / 2DGS)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import GaussianSplattingConfig
from .contracts import GaussianSplattingInput, GaussianSplattingOutput

logger = logging.getLogger(__name__)


class GaussianSplattingStep(
    BaseStep[GaussianSplattingInput, GaussianSplattingOutput, GaussianSplattingConfig]
):
    name: ClassVar[str] = "gaussian_splatting"
    input_type: ClassVar = GaussianSplattingInput
    output_type: ClassVar = GaussianSplattingOutput
    config_type: ClassVar = GaussianSplattingConfig

    def validate_inputs(self, inputs: GaussianSplattingInput) -> bool:
        if not inputs.frames_dir.exists():
            logger.error(f"Frames directory not found: {inputs.frames_dir}")
            return False
        if not inputs.sparse_dir.exists():
            logger.error(f"COLMAP sparse dir not found: {inputs.sparse_dir}")
            return False
        return True

    def run(self, inputs: GaussianSplattingInput) -> GaussianSplattingOutput:
        # TODO: Implement gsplat 2DGS training
        # Uses: gsplat library with 2DGS mode for flat-disk Gaussians
        output_dir = self.data_root / "interim" / "s03_gaussians"
        output_dir.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError("Gaussian Splatting step not yet implemented")
