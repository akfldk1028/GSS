"""I/O contracts for Step 02-alt: GS-SLAM (replaces COLMAP + 3DGS)."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class SlamInput(BaseModel):
    frames_dir: Path = Field(..., description="Directory of training images (s01 output)")


class SlamOutput(BaseModel):
    surface_points_path: Path = Field(..., description="Dense point cloud PLY (s06 compatible)")
    metadata_path: Path = Field(..., description="Metadata JSON for s06")
    camera_poses_path: Optional[Path] = Field(None, description="Camera poses (optional)")
    num_surface_points: int = Field(..., description="Number of surface points extracted")
