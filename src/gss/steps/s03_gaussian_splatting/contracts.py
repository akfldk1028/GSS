"""I/O contracts for Step 03: 3D Gaussian Splatting training."""

from pathlib import Path
from pydantic import BaseModel, Field


class GaussianSplattingInput(BaseModel):
    frames_dir: Path = Field(..., description="Directory of training images")
    sparse_dir: Path = Field(..., description="COLMAP sparse reconstruction")


class GaussianSplattingOutput(BaseModel):
    model_path: Path = Field(..., description="Path to trained Gaussian model (.ply)")
    num_gaussians: int = Field(..., description="Number of Gaussians in the model")
    training_iterations: int = Field(..., description="Number of training iterations completed")
