"""I/O contracts for Step 06b: Plane regularization."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class PlaneRegularizationInput(BaseModel):
    planes_file: Path = Field(..., description="Path to planes.json from s06")
    boundaries_file: Path = Field(..., description="Path to boundaries.json from s06")


class PlaneRegularizationOutput(BaseModel):
    planes_file: Path = Field(..., description="Path to regularized planes.json")
    boundaries_file: Path = Field(..., description="Path to regularized boundaries.json")
    walls_file: Path = Field(..., description="Path to walls.json with center-lines and thickness")
    spaces_file: Optional[Path] = Field(None, description="Path to spaces.json with room polygons")
    num_walls: int = Field(0)
    num_spaces: int = Field(0)
