"""I/O contracts for Step 05: TSDF fusion."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class TsdfFusionInput(BaseModel):
    depth_dir: Path = Field(..., description="Directory of depth maps (.npy)")
    poses_file: Path = Field(..., description="JSON file with camera poses")


class TsdfFusionOutput(BaseModel):
    tsdf_dir: Path = Field(..., description="Directory containing TSDF volume data")
    surface_points_path: Path = Field(..., description="Path to surface_points.ply")
    num_surface_points: int = Field(..., description="Number of extracted surface points")
    metadata_path: Path = Field(..., description="Path to metadata.json")
    surface_mesh_path: Optional[Path] = Field(None, description="TSDF triangle mesh (.ply)")
    num_mesh_vertices: int = Field(0, description="Number of mesh vertices")
    num_mesh_faces: int = Field(0, description="Number of mesh faces")
