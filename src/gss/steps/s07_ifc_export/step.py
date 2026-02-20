"""Step 07: IFC/BIM file generation from planes and boundaries."""

from __future__ import annotations

import logging
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import IfcExportConfig
from .contracts import IfcExportInput, IfcExportOutput

logger = logging.getLogger(__name__)


class IfcExportStep(BaseStep[IfcExportInput, IfcExportOutput, IfcExportConfig]):
    name: ClassVar[str] = "ifc_export"
    input_type: ClassVar = IfcExportInput
    output_type: ClassVar = IfcExportOutput
    config_type: ClassVar = IfcExportConfig

    def validate_inputs(self, inputs: IfcExportInput) -> bool:
        return inputs.planes_file.exists() and inputs.boundaries_file.exists()

    def run(self, inputs: IfcExportInput) -> IfcExportOutput:
        # TODO: Implement via IfcOpenShell
        # 1. Load planes.json + boundaries.json
        # 2. Create IFC project/site/building/storey hierarchy
        # 3. For each wall plane: IfcWall with extruded solid
        # 4. For each floor/ceiling: IfcSlab with extruded solid
        # 5. Write .ifc file
        output_dir = self.data_root / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError("IFC export step not yet implemented")
