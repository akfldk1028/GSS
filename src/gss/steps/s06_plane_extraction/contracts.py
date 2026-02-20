"""I/O contracts for Step 06: Plane extraction + boundary polylines."""

from pathlib import Path
from pydantic import BaseModel, Field


class PlaneExtractionInput(BaseModel):
    surface_points_path: Path = Field(..., description="Path to surface_points.ply")
    metadata_path: Path = Field(..., description="Path to TSDF metadata.json")


class DetectedPlane(BaseModel):
    id: int
    normal: list[float] = Field(..., min_length=3, max_length=3)
    d: float
    label: str = Field(..., description="wall|floor|ceiling|other")
    num_inliers: int
    boundary_3d: list[list[float]] = Field(default_factory=list, description="Polyline vertices [[x,y,z],...]")


class PlaneExtractionOutput(BaseModel):
    planes_file: Path = Field(..., description="Path to planes.json")
    boundaries_file: Path = Field(..., description="Path to boundaries.json")
    num_planes: int = Field(..., description="Total planes detected")
    num_walls: int = Field(0)
    num_floors: int = Field(0)
    num_ceilings: int = Field(0)
