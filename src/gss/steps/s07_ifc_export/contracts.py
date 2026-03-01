"""I/O contracts for Step 07: IFC/BIM export."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class IfcExportInput(BaseModel):
    walls_file: Path = Field(..., description="Path to walls.json from s06b (center-lines + thickness)")
    spaces_file: Optional[Path] = Field(None, description="Path to spaces.json from s06b (room polygons)")
    planes_file: Optional[Path] = Field(None, description="Path to planes.json (for roof planes)")
    boundaries_file: Optional[Path] = Field(None, description="Path to boundaries.json (legacy fallback)")
    building_context_file: Optional[Path] = Field(None, description="Path to building_context.json from s06c")
    columns_file: Optional[Path] = Field(None, description="Path to columns.json from s06b")
    mesh_elements_file: Optional[Path] = Field(None, description="Path to mesh_elements.json for tessellation")


class IfcExportOutput(BaseModel):
    ifc_path: Path = Field(..., description="Path to generated .ifc file")
    num_walls: int = Field(0, description="Number of IfcWall objects created")
    num_slabs: int = Field(0, description="Number of IfcSlab objects created")
    num_spaces: int = Field(0, description="Number of IfcSpace objects created")
    num_openings: int = Field(0, description="Number of IfcOpeningElement objects created")
    num_roof_slabs: int = Field(0, description="Number of IfcSlab(ROOF) objects created")
    roof_type: str = Field("none", description="Detected roof type (flat/gable/hip/shed/none)")
    num_columns: int = Field(0, description="Number of IfcColumn objects created")
    num_tessellated: int = Field(0, description="Number of tessellated elements created")
    coordinate_scale: float = Field(1.0, description="Scale factor applied (coordinate_scale from spaces.json)")
    ifc_version: str = Field("IFC4", description="IFC schema version used")
