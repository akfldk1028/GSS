"""I/O contracts for Step 07: IFC/BIM export."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class IfcExportInput(BaseModel):
    walls_file: Path = Field(..., description="Path to walls.json from s06b (center-lines + thickness)")
    spaces_file: Optional[Path] = Field(None, description="Path to spaces.json from s06b (room polygons)")
    planes_file: Optional[Path] = Field(None, description="Path to planes.json (legacy fallback)")
    boundaries_file: Optional[Path] = Field(None, description="Path to boundaries.json (legacy fallback)")


class IfcExportOutput(BaseModel):
    ifc_path: Path = Field(..., description="Path to generated .ifc file")
    num_walls: int = Field(0, description="Number of IfcWall objects created")
    num_slabs: int = Field(0, description="Number of IfcSlab objects created")
    num_spaces: int = Field(0, description="Number of IfcSpace objects created")
    coordinate_scale: float = Field(1.0, description="Scale factor applied (coordinate_scale from spaces.json)")
    ifc_version: str = Field("IFC4", description="IFC schema version used")
