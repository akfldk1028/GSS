"""Step 05: TSDF fusion from depth maps using Open3D."""

from __future__ import annotations

import logging
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import TsdfFusionConfig
from .contracts import TsdfFusionInput, TsdfFusionOutput

logger = logging.getLogger(__name__)


class TsdfFusionStep(BaseStep[TsdfFusionInput, TsdfFusionOutput, TsdfFusionConfig]):
    name: ClassVar[str] = "tsdf_fusion"
    input_type: ClassVar = TsdfFusionInput
    output_type: ClassVar = TsdfFusionOutput
    config_type: ClassVar = TsdfFusionConfig

    def validate_inputs(self, inputs: TsdfFusionInput) -> bool:
        return inputs.depth_dir.exists() and inputs.poses_file.exists()

    def run(self, inputs: TsdfFusionInput) -> TsdfFusionOutput:
        # TODO: Open3D ScalableTSDFVolume integration
        # 1. Load depth maps + poses
        # 2. volume.integrate(rgbd, intrinsic, extrinsic) for each view
        # 3. Extract surface points via volume.extract_point_cloud()
        # 4. Save surface_points.ply + metadata.json
        output_dir = self.data_root / "interim" / "s05_tsdf"
        output_dir.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError("TSDF fusion step not yet implemented")
