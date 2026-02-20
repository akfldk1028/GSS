"""I/O contracts for Step 02: COLMAP SfM."""

from pathlib import Path
from pydantic import BaseModel, Field


class ColmapInput(BaseModel):
    frames_dir: Path = Field(..., description="Directory of extracted frames")


class ColmapOutput(BaseModel):
    sparse_dir: Path = Field(..., description="COLMAP sparse reconstruction directory")
    cameras_file: Path = Field(..., description="Path to cameras.bin/txt")
    images_file: Path = Field(..., description="Path to images.bin/txt")
    num_registered: int = Field(..., description="Number of successfully registered images")
    num_points3d: int = Field(..., description="Number of 3D points in sparse cloud")
