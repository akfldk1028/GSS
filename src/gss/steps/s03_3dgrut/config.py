"""Configuration for Step 03: 3DGRUT (NVIDIA 3D Gaussian Ray Tracing)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class S03_3dgrutConfig(BaseModel):
    method: str = Field(
        "3dgut",
        description="3DGRUT method: '3dgut' (rasterization) or '3dgrt' (ray tracing)",
    )
    config_name: str = Field(
        "apps/colmap_3dgut",
        description="Hydra config path under clone/3dgrut/configs/",
    )
    num_iterations: int = Field(30000, description="Training iterations")
    export_ply: bool = Field(True, description="Export PLY for downstream pipeline")
    export_usdz: bool = Field(True, description="Export USDZ via 3DGRUT's built-in exporter")
    threedgrut_path: Path = Field(
        Path("clone/3dgrut"),
        description="Path to cloned 3DGRUT repository",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Additional Hydra overrides (e.g. ['trainer.lr=0.001'])",
    )
