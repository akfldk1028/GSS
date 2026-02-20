"""Step 04: Render depth/normal maps from trained Gaussians."""

from __future__ import annotations

import logging
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import DepthRenderConfig
from .contracts import DepthRenderInput, DepthRenderOutput

logger = logging.getLogger(__name__)


class DepthRenderStep(BaseStep[DepthRenderInput, DepthRenderOutput, DepthRenderConfig]):
    name: ClassVar[str] = "depth_render"
    input_type: ClassVar = DepthRenderInput
    output_type: ClassVar = DepthRenderOutput
    config_type: ClassVar = DepthRenderConfig

    def validate_inputs(self, inputs: DepthRenderInput) -> bool:
        return inputs.model_path.exists() and inputs.sparse_dir.exists()

    def run(self, inputs: DepthRenderInput) -> DepthRenderOutput:
        # TODO: gsplat native depth rendering from trained Gaussians
        # View sampling: uniform or coverage-based selection
        output_dir = self.data_root / "interim" / "s04_depth_maps"
        output_dir.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError("Depth render step not yet implemented")
