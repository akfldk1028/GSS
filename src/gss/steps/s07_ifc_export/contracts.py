"""I/O contracts for Step 07: IFC/BIM export."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class IfcExportInput(BaseModel):
    planes_file: Path = Field(..., description="Path to planes.json")
    boundaries_file: Path = Field(..., description="Path to boundaries.json")
    walls_file: Optional[Path] = Field(None, description="Path to walls.json from s06b (center-lines + thickness)")
    spaces_file: Optional[Path] = Field(None, description="Path to spaces.json from s06b (room polygons)")


class IfcExportOutput(BaseModel):
    ifc_path: Path = Field(..., description="Path to generated .ifc file")
    num_walls: int = Field(0, description="Number of IfcWall objects created")
    num_slabs: int = Field(0, description="Number of IfcSlab objects created")
    num_spaces: int = Field(0, description="Number of IfcSpace objects created")
    ifc_version: str = Field("IFC4", description="IFC schema version used")
