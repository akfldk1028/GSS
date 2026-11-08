"""I/O contracts for Step 09: 3DGS → USDZ export (NuRec)."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ThreeDGSExportInput(BaseModel):
    model_path: Path = Field(..., description="Path to Gaussian model PLY (from s03 or s00)")


class ThreeDGSExportOutput(BaseModel):
    usdz_path: Optional[Path] = Field(None, description="Path to NuRec .usdz file")
    nurec_manifest_path: Optional[Path] = Field(
        None, description="Path to scene_manifest_nurec.json"
    )
    num_gaussians: int = Field(0, description="Number of Gaussians in exported model")
