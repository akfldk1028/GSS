"""I/O contracts for Step 00: Import external PLY file."""

from pathlib import Path

from pydantic import BaseModel, Field


class ImportPlyInput(BaseModel):
    ply_path: Path = Field(..., description="Path to external 3DGS PLY or plain point cloud PLY")


class ImportPlyOutput(BaseModel):
    surface_points_path: Path = Field(..., description="Path to surface_points.ply (s06 compatible)")
    metadata_path: Path = Field(..., description="Path to metadata.json (s06 compatible)")
    num_surface_points: int = Field(..., description="Number of surface points in output")
