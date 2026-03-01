"""I/O contracts for Step 06d: Mesh segmentation."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class MeshSegmentationInput(BaseModel):
    surface_mesh_path: Optional[Path] = Field(
        None, description="TSDF triangle mesh (.ply) from s05 or s03_planargs"
    )
    planes_file: Path = Field(..., description="planes.json from s06")


class MeshSegmentationOutput(BaseModel):
    mesh_elements_file: Optional[Path] = Field(
        None, description="mesh_elements.json for s07 tessellation"
    )
    num_elements: int = Field(0, description="Number of mesh element clusters")
    num_faces_planar: int = Field(0, description="Faces matched to RANSAC planes")
    num_faces_residual: int = Field(0, description="Non-planar faces retained")
    num_faces_discarded: int = Field(0, description="Tiny clusters removed")
