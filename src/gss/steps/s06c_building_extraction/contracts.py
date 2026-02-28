"""I/O contracts for Step 06c: Building extraction."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class BuildingExtractionInput(BaseModel):
    planes_file: Path = Field(..., description="Path to planes.json from s06")
    boundaries_file: Path = Field(..., description="Path to boundaries.json from s06")
    surface_points_file: Optional[Path] = Field(
        None, description="Path to surface_points.ply (optional, for building segmentation)"
    )


class BuildingExtractionOutput(BaseModel):
    planes_file: Path = Field(..., description="Path to updated planes.json (with ground/exterior labels)")
    boundaries_file: Path = Field(..., description="Path to updated boundaries.json")
    building_context_file: Path = Field(
        ..., description="Path to building_context.json (footprint, facades, roof, storeys)"
    )
    building_points_file: Optional[Path] = Field(
        None, description="Path to building_points.ply (segmented point cloud)"
    )
    num_facades: int = Field(0)
    num_roof_faces: int = Field(0)
    num_storeys: int = Field(0)
