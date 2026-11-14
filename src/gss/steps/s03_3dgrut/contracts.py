"""I/O contracts for Step 03: 3DGRUT training (NVIDIA 3D Gaussian Ray Tracing)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class S03_3dgrutInput(BaseModel):
    frames_dir: Path = Field(..., description="Directory of training images (s01 output)")
    sparse_dir: Path = Field(..., description="COLMAP sparse reconstruction (s02 output)")


class S03_3dgrutOutput(BaseModel):
    model_path: Path = Field(..., description="Exported PLY for downstream pipeline (s04+)")
    checkpoint_path: Path = Field(..., description="3DGRUT checkpoint for gRPC rendering")
    usdz_path: Optional[Path] = Field(None, description="USDZ export (Omniverse/ARKit)")
    num_gaussians: int = Field(..., description="Number of Gaussian primitives")
    training_iterations: int = Field(..., description="Completed training iterations")
