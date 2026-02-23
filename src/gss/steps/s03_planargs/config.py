"""Configuration for Step 03-alt: PlanarGS subprocess wrapper."""

from pathlib import Path
from pydantic import BaseModel, Field


class PlanarGSConfig(BaseModel):
    planargs_repo: Path = Field(Path("clone/PlanarGS"), description="Path to PlanarGS repository")
    conda_env: str = Field("planargs", description="Conda environment name for PlanarGS")
    group_size: int = Field(25, description="DUSt3R group size (25 for 16GB VRAM, 40 for 24GB)")
    text_prompts: str = Field(
        "wall. floor. door. screen. window. ceiling. table",
        description="GroundedSAM text prompts for planar segmentation",
    )
    iterations: int = Field(30000, description="Gaussian splatting training iterations")
    voxel_size: float = Field(0.02, description="TSDF voxel size for mesh extraction")
    max_depth: float = Field(100.0, description="Maximum depth for TSDF fusion")
    skip_geomprior: bool = Field(False, description="Skip DUSt3R geometric prior generation")
    skip_lp3: bool = Field(False, description="Skip GroundedSAM planar mask generation")
    skip_train: bool = Field(False, description="Skip training (use existing checkpoint)")
