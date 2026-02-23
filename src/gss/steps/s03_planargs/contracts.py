"""I/O contracts for Step 03-alt: PlanarGS (NeurIPS 2025) subprocess wrapper."""

from pathlib import Path
from pydantic import BaseModel, Field


class PlanarGSInput(BaseModel):
    frames_dir: Path = Field(..., description="Directory of training images (s01 output)")
    sparse_dir: Path = Field(..., description="COLMAP sparse/0/ directory (s02 output)")


class PlanarGSOutput(BaseModel):
    surface_points_path: Path = Field(..., description="Mesh vertices as PLY (s06 compatible)")
    mesh_path: Path = Field(..., description="TSDF fusion mesh (tsdf_fusion_post.ply)")
    metadata_path: Path = Field(..., description="Metadata JSON for s06")
    num_surface_points: int = Field(..., description="Number of surface points extracted")
