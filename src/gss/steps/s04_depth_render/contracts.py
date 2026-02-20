"""I/O contracts for Step 04: Depth/Normal map rendering from 3DGS."""

from pathlib import Path
from pydantic import BaseModel, Field


class DepthRenderInput(BaseModel):
    model_path: Path = Field(..., description="Path to trained Gaussian model")
    sparse_dir: Path = Field(..., description="COLMAP sparse dir (for camera poses)")


class DepthRenderOutput(BaseModel):
    depth_dir: Path = Field(..., description="Directory of rendered depth maps (.npy)")
    normal_dir: Path | None = Field(None, description="Directory of rendered normal maps")
    num_views: int = Field(..., description="Number of views rendered")
    poses_file: Path = Field(..., description="JSON file with camera poses used")
