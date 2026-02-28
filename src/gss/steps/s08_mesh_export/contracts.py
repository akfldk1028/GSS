"""I/O contracts for Step 08: Mesh export (IFC → GLB/USD)."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class MeshExportInput(BaseModel):
    ifc_path: Path = Field(..., description="Path to .ifc file from s07")


class MeshExportOutput(BaseModel):
    glb_path: Optional[Path] = Field(None, description="Path to exported .glb file")
    usd_path: Optional[Path] = Field(None, description="Path to exported .usdc file")
    usdz_path: Optional[Path] = Field(None, description="Path to exported .usdz file")
    num_meshes: int = Field(0, description="Number of mesh elements exported")
    num_vertices: int = Field(0, description="Total vertex count across all meshes")
    num_faces: int = Field(0, description="Total face count across all meshes")
