"""Step 06: Plane extraction and boundary polyline generation."""

from __future__ import annotations

import logging
from typing import ClassVar

from gss.core.step_base import BaseStep
from .config import PlaneExtractionConfig
from .contracts import PlaneExtractionInput, PlaneExtractionOutput

logger = logging.getLogger(__name__)


class PlaneExtractionStep(
    BaseStep[PlaneExtractionInput, PlaneExtractionOutput, PlaneExtractionConfig]
):
    name: ClassVar[str] = "plane_extraction"
    input_type: ClassVar = PlaneExtractionInput
    output_type: ClassVar = PlaneExtractionOutput
    config_type: ClassVar = PlaneExtractionConfig

    def validate_inputs(self, inputs: PlaneExtractionInput) -> bool:
        return inputs.surface_points_path.exists()

    def run(self, inputs: PlaneExtractionInput) -> PlaneExtractionOutput:
        # TODO: Implement
        # 1. Open3D segment_plane() iterative RANSAC
        # 2. Normal-based classification (wall/floor/ceiling)
        # 3. alphashape boundary extraction per plane
        # 4. Shapely Douglas-Peucker simplification
        # 5. Save planes.json + boundaries.json
        output_dir = self.data_root / "interim" / "s06_planes"
        output_dir.mkdir(parents=True, exist_ok=True)
        raise NotImplementedError("Plane extraction step not yet implemented")
